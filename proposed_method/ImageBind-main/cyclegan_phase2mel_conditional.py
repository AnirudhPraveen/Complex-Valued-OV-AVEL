"""
Conditional CycleGAN: IF phase spectrogram -> mel-like magnitude spectrogram,
conditioned on the corresponding real mel spectrogram.

Idea:
    - Translate IF phase (257x204) into a mel-like image (128x204) so the
      pretrained ImageBind audio encoder can consume it, while preserving
      phase information via cycle consistency.
    - Condition both generators on the PAIRED mel spectrogram so the network
      has content guidance (the mel is always available at inference since
      it is derived directly from the waveform).

Generators:
    G_IF2Mel(IF, cond_mel) -> fake_mel      [1, 128, 204]
    G_Mel2IF(Mel, cond_IF) -> fake_IF       [1, 257, 204]

Conditioning is concatenated as a second input channel after resizing to the
source spatial resolution -- simple, stable, and adds minimal parameters.

Cycle:
    G_Mel2IF(G_IF2Mel(IF, mel), IF) ~ IF
    G_IF2Mel(G_Mel2IF(Mel, IF), Mel) ~ Mel

Usage:
    python cyclegan_phase2mel_conditional.py --data_dir ./spectrograms --epochs 60
    python cyclegan_phase2mel_conditional.py --resume checkpoints/cyclegan_cond/cyclegan_best.pt --eval_only
"""

import os
import argparse
import itertools
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter


# ============================================================================
# Paired dataset (same filename in both subdirs)
# ============================================================================

class PairedSpectrogramDataset(Dataset):
    def __init__(self, data_dir, split='train',
                 if_range=None, mel_range=None):
        self.if_dir = os.path.join(data_dir, 'phase_if')
        self.mel_dir = os.path.join(data_dir, 'magnitude')
        files = sorted(os.listdir(self.if_dir))
        if split:
            files = [f for f in files if f.startswith(split + '_')]
        files = [f for f in files if f.endswith('.npy')]
        mel_set = set(os.listdir(self.mel_dir))
        self.filenames = [f for f in files if f in mel_set]
        assert len(self.filenames) > 0, f"No paired files for split={split}"
        self.if_range = if_range
        self.mel_range = mel_range

    @staticmethod
    def _norm(x, rng):
        vmin, vmax = rng
        return 2.0 * (x - vmin) / (vmax - vmin + 1e-6) - 1.0

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        name = self.filenames[idx]
        ifx = np.load(os.path.join(self.if_dir, name))
        mel = np.load(os.path.join(self.mel_dir, name))
        ifx = torch.from_numpy(ifx).float().unsqueeze(0)
        mel = torch.from_numpy(mel).float().unsqueeze(0)
        if self.if_range is not None:
            ifx = self._norm(ifx, self.if_range)
        if self.mel_range is not None:
            mel = self._norm(mel, self.mel_range)
        return ifx, mel


def compute_domain_stats(data_dir, subdir, split='train', num_samples=500):
    dir_path = os.path.join(data_dir, subdir)
    files = sorted(os.listdir(dir_path))
    if split:
        files = [f for f in files if f.startswith(split + '_')]
    files = [f for f in files if f.endswith('.npy')][:num_samples]
    samples = np.concatenate([np.load(os.path.join(dir_path, f)).flatten() for f in files])
    vmin = float(np.percentile(samples, 1))
    vmax = float(np.percentile(samples, 99))
    return vmin, vmax


# ============================================================================
# Replay buffer
# ============================================================================

class ReplayBuffer:
    def __init__(self, max_size=50):
        self.max_size = max_size
        self.data = []

    def push_and_pop(self, data):
        out = []
        for el in data:
            el = el.unsqueeze(0)
            if len(self.data) < self.max_size:
                self.data.append(el)
                out.append(el)
            else:
                if random.random() > 0.5:
                    i = random.randint(0, self.max_size - 1)
                    out.append(self.data[i].clone())
                    self.data[i] = el
                else:
                    out.append(el)
        return torch.cat(out, dim=0)


# ============================================================================
# Conditional ResNet generator: input = source + cond (concat channels)
# ============================================================================

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1), nn.Conv2d(channels, channels, 3),
            nn.InstanceNorm2d(channels), nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1), nn.Conv2d(channels, channels, 3),
            nn.InstanceNorm2d(channels),
        )

    def forward(self, x):
        return x + self.block(x)


class ConditionalResNetGenerator(nn.Module):
    """
    Generator that takes a source spectrogram and a conditioning spectrogram.
    The condition is resized to the source freq/time and concatenated as
    extra input channels.
    """
    def __init__(self, in_freq, out_freq, cond_freq,
                 in_channels=1, cond_channels=1, out_channels=1,
                 n_residual=9, features=64):
        super().__init__()
        self.in_freq = in_freq
        self.out_freq = out_freq
        self.cond_freq = cond_freq

        total_in = in_channels + cond_channels

        # Adapt source freq to out_freq before the main network
        if in_freq > out_freq:
            self.freq_adapt = nn.Conv2d(total_in, total_in,
                                        kernel_size=(in_freq - out_freq + 1, 1))
        elif out_freq > in_freq:
            self.freq_adapt = nn.ConvTranspose2d(total_in, total_in,
                                                 kernel_size=(out_freq - in_freq + 1, 1))
        else:
            self.freq_adapt = nn.Identity()

        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(total_in, features, 7),
            nn.InstanceNorm2d(features),
            nn.ReLU(inplace=True),
        ]
        in_f = features
        for _ in range(2):
            out_f = in_f * 2
            model += [nn.Conv2d(in_f, out_f, 3, stride=2, padding=1),
                      nn.InstanceNorm2d(out_f), nn.ReLU(inplace=True)]
            in_f = out_f
        for _ in range(n_residual):
            model += [ResidualBlock(in_f)]
        for _ in range(2):
            out_f = in_f // 2
            model += [nn.ConvTranspose2d(in_f, out_f, 3, stride=2, padding=1, output_padding=1),
                      nn.InstanceNorm2d(out_f), nn.ReLU(inplace=True)]
            in_f = out_f
        model += [nn.ReflectionPad2d(3), nn.Conv2d(features, out_channels, 7), nn.Tanh()]
        self.model = nn.Sequential(*model)

    def forward(self, x, cond):
        # x: [B, 1, in_freq, T]; cond: [B, 1, cond_freq, T_c]
        T = x.size(3)
        if cond.size(2) != self.in_freq or cond.size(3) != T:
            cond = F.interpolate(cond, size=(self.in_freq, T),
                                 mode='bilinear', align_corners=False)
        h = torch.cat([x, cond], dim=1)    # [B, 2, in_freq, T]
        h = self.freq_adapt(h)             # [B, 2, out_freq, T]
        out = self.model(h)
        out = out[:, :, :self.out_freq, :T]
        if out.size(2) < self.out_freq:
            out = F.pad(out, [0, 0, 0, self.out_freq - out.size(2)])
        if out.size(3) < T:
            out = F.pad(out, [0, T - out.size(3)])
        return out


# ============================================================================
# Discriminator (optionally conditional — here we keep it unconditional
# to avoid the GAN collapsing into trivially copying the condition)
# ============================================================================

class PatchDiscriminator(nn.Module):
    def __init__(self, in_channels=1, use_spectral_norm=True):
        super().__init__()
        norm_fn = nn.utils.spectral_norm if use_spectral_norm else (lambda m: m)

        def block(i, o, stride=2, normalize=True):
            layers = [norm_fn(nn.Conv2d(i, o, 4, stride=stride, padding=1))]
            if normalize:
                layers.append(nn.InstanceNorm2d(o))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(in_channels, 64, normalize=False),
            *block(64, 128), *block(128, 256),
            *block(256, 512, stride=1),
            norm_fn(nn.Conv2d(512, 1, 4, padding=1)),
        )

    def forward(self, x):
        if self.training:
            x = x + 0.05 * torch.randn_like(x)
        return self.model(x)


# ============================================================================
# Distribution matching loss
# ============================================================================

class DistributionMatchingLoss(nn.Module):
    def forward(self, fake, real):
        fm = fake.mean(dim=(0, 3)); rm = real.mean(dim=(0, 3))
        fs = fake.std(dim=(0, 3)) + 1e-6; rs = real.std(dim=(0, 3)) + 1e-6
        mean_loss = F.l1_loss(fm, rm)
        std_loss = F.l1_loss(fs, rs)
        ff = fake.reshape(-1); rf = real.reshape(-1)
        fn = (ff - ff.mean()) / (ff.std() + 1e-6)
        rn = (rf - rf.mean()) / (rf.std() + 1e-6)
        skew = F.l1_loss(fn.pow(3).mean(), rn.pow(3).mean())
        kurt = F.l1_loss(fn.pow(4).mean(), rn.pow(4).mean())
        return mean_loss + std_loss + 0.5 * (skew + kurt)


# ============================================================================
# Training
# ============================================================================

def build_models(device, n_residual):
    # G_IF2Mel: IF (257) + cond mel (128) -> mel-like (128)
    G_IF2Mel = ConditionalResNetGenerator(
        in_freq=257, out_freq=128, cond_freq=128, n_residual=n_residual).to(device)
    # G_Mel2IF: mel (128) + cond IF (257) -> IF-like (257)
    G_Mel2IF = ConditionalResNetGenerator(
        in_freq=128, out_freq=257, cond_freq=257, n_residual=n_residual).to(device)
    D_Mel = PatchDiscriminator(in_channels=1).to(device)
    D_IF = PatchDiscriminator(in_channels=1).to(device)
    return G_IF2Mel, G_Mel2IF, D_Mel, D_IF


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    print("Computing domain statistics...")
    if_range = compute_domain_stats(args.data_dir, 'phase_if', 'train')
    mel_range = compute_domain_stats(args.data_dir, 'magnitude', 'train')
    print(f"IF range={if_range}  Mel range={mel_range}")

    train_ds = PairedSpectrogramDataset(args.data_dir, split='train',
                                        if_range=if_range, mel_range=mel_range)
    val_ds = PairedSpectrogramDataset(args.data_dir, split='val',
                                      if_range=if_range, mel_range=mel_range)
    print(f"Train pairs: {len(train_ds)}  Val pairs: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers)

    G_IF2Mel, G_Mel2IF, D_Mel, D_IF = build_models(device, args.n_residual)

    criterion_adv = nn.MSELoss()
    criterion_cycle = nn.L1Loss()
    criterion_identity = nn.L1Loss()
    criterion_dist = DistributionMatchingLoss()

    opt_G = torch.optim.Adam(
        itertools.chain(G_IF2Mel.parameters(), G_Mel2IF.parameters()),
        lr=args.lr_g, betas=(0.5, 0.999))
    opt_D = torch.optim.Adam(
        itertools.chain(D_Mel.parameters(), D_IF.parameters()),
        lr=args.lr_d, betas=(0.5, 0.999))
    sched_G = CosineAnnealingLR(opt_G, T_max=args.epochs, eta_min=args.lr_g * 0.01)
    sched_D = CosineAnnealingLR(opt_D, T_max=args.epochs, eta_min=args.lr_d * 0.01)

    fake_mel_buf = ReplayBuffer(50)
    fake_if_buf = ReplayBuffer(50)

    start_epoch = 0
    best_val = float('inf')
    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        G_IF2Mel.load_state_dict(ckpt['G_IF2Mel'])
        G_Mel2IF.load_state_dict(ckpt['G_Mel2IF'])
        D_Mel.load_state_dict(ckpt['D_Mel'])
        D_IF.load_state_dict(ckpt['D_IF'])
        opt_G.load_state_dict(ckpt['opt_G'])
        opt_D.load_state_dict(ckpt['opt_D'])
        start_epoch = ckpt.get('epoch', 0) + 1
        best_val = ckpt.get('best_val', float('inf'))
        print(f"Resumed epoch {start_epoch}  best_val={best_val:.4f}")

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join(args.checkpoint_dir, 'tb_logs'))
    step = start_epoch * len(train_loader)
    no_improve = 0

    for epoch in range(start_epoch, args.epochs):
        G_IF2Mel.train(); G_Mel2IF.train(); D_Mel.train(); D_IF.train()
        epoch_g, epoch_d = 0.0, 0.0

        for bi, (real_if, real_mel) in enumerate(train_loader):
            real_if = real_if.to(device)    # [B,1,257,204]
            real_mel = real_mel.to(device)  # [B,1,128,204]
            B = real_if.size(0)
            real_lbl = torch.full((B, 1, 1, 1), 0.9, device=device)
            fake_lbl = torch.full((B, 1, 1, 1), 0.1, device=device)

            # ---- Generators ----
            opt_G.zero_grad()
            # Identity: feed target as source with target as cond -> should stay
            idt_mel = G_IF2Mel(F.interpolate(real_mel, size=(257, real_mel.size(3)),
                                             mode='bilinear', align_corners=False),
                               real_mel)
            loss_idt_mel = criterion_identity(idt_mel, real_mel) * args.lambda_identity
            idt_if = G_Mel2IF(F.interpolate(real_if, size=(128, real_if.size(3)),
                                            mode='bilinear', align_corners=False),
                              real_if)
            loss_idt_if = criterion_identity(idt_if, real_if) * args.lambda_identity

            # Forward translations
            fake_mel = G_IF2Mel(real_if, real_mel)       # IF + cond(mel) -> mel-like
            fake_if = G_Mel2IF(real_mel, real_if)        # mel + cond(IF) -> IF-like

            loss_adv_IF2Mel = criterion_adv(D_Mel(fake_mel),
                                            real_lbl.expand_as(D_Mel(fake_mel)))
            loss_adv_Mel2IF = criterion_adv(D_IF(fake_if),
                                            real_lbl.expand_as(D_IF(fake_if)))

            # Cycles: reuse the SAME conditioning (it is always available)
            recovered_if = G_Mel2IF(fake_mel, real_if)
            recovered_mel = G_IF2Mel(fake_if, real_mel)
            loss_cyc_if = criterion_cycle(recovered_if, real_if) * args.lambda_cycle
            loss_cyc_mel = criterion_cycle(recovered_mel, real_mel) * args.lambda_cycle

            loss_dist = criterion_dist(fake_mel, real_mel) * args.lambda_dist

            loss_G = (loss_adv_IF2Mel + loss_adv_Mel2IF
                      + loss_cyc_if + loss_cyc_mel
                      + loss_idt_mel + loss_idt_if
                      + loss_dist)
            loss_G.backward()
            opt_G.step()

            # ---- Discriminators ----
            opt_D.zero_grad()
            fmb = fake_mel_buf.push_and_pop(fake_mel.detach())
            fib = fake_if_buf.push_and_pop(fake_if.detach())
            pr_m = D_Mel(real_mel); pf_m = D_Mel(fmb)
            pr_i = D_IF(real_if); pf_i = D_IF(fib)
            loss_D_Mel = 0.5 * (criterion_adv(pr_m, real_lbl.expand_as(pr_m))
                                + criterion_adv(pf_m, fake_lbl.expand_as(pf_m)))
            loss_D_IF = 0.5 * (criterion_adv(pr_i, real_lbl.expand_as(pr_i))
                               + criterion_adv(pf_i, fake_lbl.expand_as(pf_i)))
            loss_D = loss_D_Mel + loss_D_IF
            if loss_D.item() > 0.1:
                loss_D.backward()
                opt_D.step()

            epoch_g += loss_G.item(); epoch_d += loss_D.item()

            writer.add_scalars('Loss/G', {
                'total': loss_G.item(),
                'adv_IF2Mel': loss_adv_IF2Mel.item(),
                'adv_Mel2IF': loss_adv_Mel2IF.item(),
                'cyc_IF': loss_cyc_if.item(),
                'cyc_Mel': loss_cyc_mel.item(),
                'idt_Mel': loss_idt_mel.item(),
                'idt_IF': loss_idt_if.item(),
                'dist': loss_dist.item(),
            }, step)
            writer.add_scalars('Loss/D', {
                'total': loss_D.item(),
                'D_Mel': loss_D_Mel.item(),
                'D_IF': loss_D_IF.item(),
            }, step)
            step += 1

            if bi % 50 == 0:
                print(f"  [{epoch}/{args.epochs}] {bi}/{len(train_loader)}  "
                      f"G={loss_G.item():.3f} D={loss_D.item():.3f}  "
                      f"adv={loss_adv_IF2Mel.item():.2f}+{loss_adv_Mel2IF.item():.2f}  "
                      f"cyc={loss_cyc_if.item():.2f}+{loss_cyc_mel.item():.2f}  "
                      f"dist={loss_dist.item():.2f}")

        sched_G.step(); sched_D.step()

        # ---- Val ----
        val_loss = validate(G_IF2Mel, G_Mel2IF, val_loader, criterion_cycle, device)
        avg_g = epoch_g / len(train_loader); avg_d = epoch_d / len(train_loader)
        print(f"Epoch {epoch}/{args.epochs}  G={avg_g:.3f} D={avg_d:.3f}  val_cyc={val_loss:.4f}")
        writer.add_scalars('Epoch', {'G': avg_g, 'D': avg_d, 'val_cycle': val_loss}, epoch)

        ckpt_data = {
            'epoch': epoch, 'best_val': best_val,
            'G_IF2Mel': G_IF2Mel.state_dict(),
            'G_Mel2IF': G_Mel2IF.state_dict(),
            'D_Mel': D_Mel.state_dict(), 'D_IF': D_IF.state_dict(),
            'opt_G': opt_G.state_dict(), 'opt_D': opt_D.state_dict(),
            'n_residual': args.n_residual,
            'domain_stats': {'if_range': if_range, 'mel_range': mel_range},
        }
        torch.save(ckpt_data, os.path.join(args.checkpoint_dir, 'cyclegan_latest.pt'))
        if val_loss < best_val:
            best_val = val_loss; no_improve = 0
            torch.save(ckpt_data, os.path.join(args.checkpoint_dir, 'cyclegan_best.pt'))
            print(f"  -> new best val_cyc={best_val:.4f}")
        else:
            no_improve += 1

        if (epoch + 1) % 5 == 0 or epoch == 0:
            save_visualizations(G_IF2Mel, G_Mel2IF, val_ds, device, args, epoch=epoch)

        if no_improve >= args.patience:
            print(f"Early stop at epoch {epoch}")
            break

    writer.close()
    print(f"Done. best val_cyc={best_val:.4f}")


def validate(G_IF2Mel, G_Mel2IF, loader, criterion_cycle, device):
    G_IF2Mel.eval(); G_Mel2IF.eval()
    tot = 0.0; n = 0
    with torch.no_grad():
        for real_if, real_mel in loader:
            real_if = real_if.to(device); real_mel = real_mel.to(device)
            fake_mel = G_IF2Mel(real_if, real_mel)
            rec_if = G_Mel2IF(fake_mel, real_if)
            tot += criterion_cycle(rec_if, real_if).item()
            n += 1
    G_IF2Mel.train(); G_Mel2IF.train()
    return tot / max(n, 1)


# ============================================================================
# Visualization
# ============================================================================

def save_visualizations(G_IF2Mel, G_Mel2IF, ds, device, args, epoch=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    G_IF2Mel.eval(); G_Mel2IF.eval()
    out_dir = os.path.join(args.checkpoint_dir, 'visualizations')
    os.makedirs(out_dir, exist_ok=True)
    n = min(6, len(ds))
    idxs = np.random.choice(len(ds), size=n, replace=False)
    for i, ix in enumerate(idxs):
        real_if, real_mel = ds[ix]
        real_if = real_if.unsqueeze(0).to(device)
        real_mel = real_mel.unsqueeze(0).to(device)
        with torch.no_grad():
            fake_mel = G_IF2Mel(real_if, real_mel)
            rec_if = G_Mel2IF(fake_mel, real_if)
        fig, ax = plt.subplots(1, 4, figsize=(20, 4))
        ax[0].imshow(real_if.cpu().squeeze().numpy(), aspect='auto', origin='lower', cmap='RdBu_r')
        ax[0].set_title('IF input [257x204]')
        ax[1].imshow(real_mel.cpu().squeeze().numpy(), aspect='auto', origin='lower', cmap='magma')
        ax[1].set_title('Mel condition [128x204]')
        ax[2].imshow(fake_mel.cpu().squeeze().numpy(), aspect='auto', origin='lower', cmap='magma')
        ax[2].set_title('Generated mel-like')
        ax[3].imshow(rec_if.cpu().squeeze().numpy(), aspect='auto', origin='lower', cmap='RdBu_r')
        ax[3].set_title('Recovered IF (cycle)')
        tag = f"_epoch{epoch}" if epoch is not None else ""
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f'cond_cycle{tag}_s{i}.png'), dpi=140)
        plt.close()
    G_IF2Mel.train(); G_Mel2IF.train()


# ============================================================================
# Inference helper
# ============================================================================

@torch.no_grad()
def load_trained(checkpoint_path, device='cuda'):
    ckpt = torch.load(checkpoint_path, map_location=device)
    n_res = ckpt.get('n_residual', 9)
    G_IF2Mel, G_Mel2IF, _, _ = build_models(device, n_res)
    G_IF2Mel.load_state_dict(ckpt['G_IF2Mel'])
    G_Mel2IF.load_state_dict(ckpt['G_Mel2IF'])
    G_IF2Mel.eval(); G_Mel2IF.eval()
    s = ckpt['domain_stats']
    return G_IF2Mel, G_Mel2IF, s['if_range'], s['mel_range']


@torch.no_grad()
def translate_phase_to_mel(G_IF2Mel, if_range, mel_range,
                            phase_if, mel_cond, device='cuda'):
    """phase_if: [B,1,257,204] raw; mel_cond: [B,1,128,204] raw."""
    ivmin, ivmax = if_range; mvmin, mvmax = mel_range
    x = phase_if.to(device); c = mel_cond.to(device)
    x = 2.0 * (x - ivmin) / (ivmax - ivmin + 1e-6) - 1.0
    c = 2.0 * (c - mvmin) / (mvmax - mvmin + 1e-6) - 1.0
    out = G_IF2Mel(x, c)
    return (out + 1.0) / 2.0 * (mvmax - mvmin + 1e-6) + mvmin


# ============================================================================
# Eval
# ============================================================================

def evaluate(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    G_IF2Mel, G_Mel2IF, if_range, mel_range = load_trained(args.resume, device)
    val_ds = PairedSpectrogramDataset(args.data_dir, split='val',
                                      if_range=if_range, mel_range=mel_range)
    loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers)
    fakes, reals, cyc = [], [], []
    with torch.no_grad():
        for real_if, real_mel in loader:
            real_if = real_if.to(device); real_mel = real_mel.to(device)
            fake = G_IF2Mel(real_if, real_mel)
            rec = G_Mel2IF(fake, real_if)
            cyc.append(F.l1_loss(rec, real_if).item())
            mvmin, mvmax = mel_range
            fake_d = (fake + 1.0) / 2.0 * (mvmax - mvmin + 1e-6) + mvmin
            real_d = (real_mel + 1.0) / 2.0 * (mvmax - mvmin + 1e-6) + mvmin
            fakes.append(fake_d.cpu()); reals.append(real_d.cpu())
    F_ = torch.cat(fakes); R = torch.cat(reals)
    print(f"\nVal ({len(F_)} samples)")
    print(f"  real: mean={R.mean():.3f} std={R.std():.3f} min={R.min():.3f} max={R.max():.3f}")
    print(f"  fake: mean={F_.mean():.3f} std={F_.std():.3f} min={F_.min():.3f} max={F_.max():.3f}")
    print(f"  cycle L1 (IF->mel->IF): {np.mean(cyc):.4f}")
    save_visualizations(G_IF2Mel, G_Mel2IF, val_ds, device, args)


# ============================================================================
# Main
# ============================================================================

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', type=str, default='./spectrograms')
    p.add_argument('--checkpoint_dir', type=str, default='./checkpoints/cyclegan_cond')
    p.add_argument('--resume', type=str, default=None)
    p.add_argument('--eval_only', action='store_true')
    p.add_argument('--epochs', type=int, default=60)
    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--lr_g', type=float, default=2e-4)
    p.add_argument('--lr_d', type=float, default=1e-4)
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--patience', type=int, default=15)
    p.add_argument('--n_residual', type=int, default=9)
    p.add_argument('--lambda_cycle', type=float, default=10.0)
    p.add_argument('--lambda_identity', type=float, default=2.0)
    p.add_argument('--lambda_dist', type=float, default=2.0)
    args = p.parse_args()

    if args.eval_only:
        assert args.resume, "--resume required for --eval_only"
        evaluate(args)
    else:
        train(args)


if __name__ == '__main__':
    main()
