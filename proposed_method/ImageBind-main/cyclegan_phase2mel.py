"""
CycleGAN: Translate IF phase spectrograms → mel-like spectrograms (unpaired).

Goal: Make IF phase spectrograms LOOK like mel spectrograms so the pretrained
ImageBind audio encoder can extract features, while PRESERVING the phase
information through cycle consistency.

Key difference from Pix2Pix:
    - No paired supervision — learns domain transfer, not pixel reconstruction
    - Cycle consistency: IF → mel-like → IF (must recover original)
    - Phase information is preserved because the reverse mapping must work
    - Identity loss: mel → G_IF2Mel(mel) ≈ mel (don't change what's already mel-like)

Architecture:
    G_IF2Mel:  ResNet generator (257×204 → 128×204)
    G_Mel2IF:  ResNet generator (128×204 → 257×204)
    D_Mel:     PatchGAN discriminator (judges mel-domain realism)
    D_IF:      PatchGAN discriminator (judges IF-domain realism)

Losses:
    1. Adversarial (LSGAN): Domain realism
    2. Cycle consistency: IF → mel → IF ≈ original, mel → IF → mel ≈ original
    3. Identity: G_IF2Mel(mel) ≈ mel, G_Mel2IF(IF) ≈ IF
    4. Distribution matching: Output follows mel statistics

Usage:
    python cyclegan_phase2mel.py --data_dir ./spectrograms --epochs 30

    # Evaluate / visualize
    python cyclegan_phase2mel.py --data_dir ./spectrograms --eval_only --resume checkpoints/cyclegan/cyclegan_best.pt
    python cyclegan_phase2mel.py --data_dir ./spectrograms --visualize --resume checkpoints/cyclegan/cyclegan_best.pt
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


# =============================================================================
# Datasets (unpaired — separate loaders for each domain)
# =============================================================================

class SpectrogramDataset(Dataset):
    """Load spectrograms from a single directory (one domain)."""
    def __init__(self, data_dir, subdir, split='train', normalize_range=None):
        self.dir = os.path.join(data_dir, subdir)
        files = sorted(os.listdir(self.dir))
        if split:
            files = [f for f in files if f.startswith(split + '_')]
        self.filenames = [f for f in files if f.endswith('.npy')]
        assert len(self.filenames) > 0, f"No files found in {self.dir} for split='{split}'"
        self.normalize_range = normalize_range  # (min, max) for normalization to [-1, 1]

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        data = np.load(os.path.join(self.dir, self.filenames[idx]))
        data = torch.from_numpy(data).float().unsqueeze(0)  # [1, F, T]
        if self.normalize_range is not None:
            vmin, vmax = self.normalize_range
            data = 2.0 * (data - vmin) / (vmax - vmin + 1e-6) - 1.0
        return data


def compute_domain_stats(data_dir, subdir, split='train', num_samples=500):
    """Compute min/max from a domain for normalization."""
    dir_path = os.path.join(data_dir, subdir)
    files = sorted(os.listdir(dir_path))
    if split:
        files = [f for f in files if f.startswith(split + '_')]
    files = [f for f in files if f.endswith('.npy')]

    samples = []
    for f in files[:num_samples]:
        samples.append(np.load(os.path.join(dir_path, f)))
    samples = np.concatenate([s.flatten() for s in samples])

    # Use percentile-based range for robustness
    vmin = float(np.percentile(samples, 1))
    vmax = float(np.percentile(samples, 99))
    mean = float(samples.mean())
    std = float(samples.std())
    return vmin, vmax, mean, std


# =============================================================================
# Replay Buffer (stabilizes GAN training)
# =============================================================================

class ReplayBuffer:
    """
    Stores previously generated images and randomly returns them
    instead of the latest ones. Prevents D from overfitting to
    the generator's current output.
    """
    def __init__(self, max_size=50):
        self.max_size = max_size
        self.data = []

    def push_and_pop(self, data):
        result = []
        for element in data:
            element = element.unsqueeze(0)
            if len(self.data) < self.max_size:
                self.data.append(element)
                result.append(element)
            else:
                if random.random() > 0.5:
                    idx = random.randint(0, self.max_size - 1)
                    result.append(self.data[idx].clone())
                    self.data[idx] = element
                else:
                    result.append(element)
        return torch.cat(result, dim=0)


# =============================================================================
# Generator: ResNet-based (better than U-Net for unpaired transfer)
# =============================================================================

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, 3),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, 3),
            nn.InstanceNorm2d(channels),
        )

    def forward(self, x):
        return x + self.block(x)


class ResNetGenerator(nn.Module):
    """
    ResNet generator for domain transfer.

    Architecture:
        1. Frequency projection (adaptive pool to target freq size)
        2. Initial conv (7×7 with reflection padding)
        3. 2 downsampling layers
        4. N residual blocks (default 9)
        5. 2 upsampling layers
        6. Output conv (7×7) + Tanh

    ResNet generators are preferred over U-Net for CycleGAN because
    skip connections in U-Net encourage copying input details, while
    ResNet blocks learn transformations through residuals.
    """
    def __init__(self, in_freq, out_freq, in_channels=1, out_channels=1,
                 n_residual=9, features=64):
        super().__init__()
        self.in_freq = in_freq
        self.out_freq = out_freq

        # Learned frequency axis projection (instead of lossy avg pooling)
        self.freq_adapt = nn.Conv2d(in_channels, in_channels, kernel_size=(in_freq - out_freq + 1, 1),
                                     padding=0) if in_freq > out_freq else (
            nn.ConvTranspose2d(in_channels, in_channels, kernel_size=(out_freq - in_freq + 1, 1),
                               padding=0) if out_freq > in_freq else nn.Identity()
        )

        # Initial convolution
        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels, features, 7),
            nn.InstanceNorm2d(features),
            nn.ReLU(inplace=True),
        ]

        # Downsampling (2 layers: features → 2*features → 4*features)
        in_f = features
        for _ in range(2):
            out_f = in_f * 2
            model += [
                nn.Conv2d(in_f, out_f, 3, stride=2, padding=1),
                nn.InstanceNorm2d(out_f),
                nn.ReLU(inplace=True),
            ]
            in_f = out_f

        # Residual blocks
        for _ in range(n_residual):
            model += [ResidualBlock(in_f)]

        # Upsampling (2 layers: 4*features → 2*features → features)
        for _ in range(2):
            out_f = in_f // 2
            model += [
                nn.ConvTranspose2d(in_f, out_f, 3, stride=2, padding=1, output_padding=1),
                nn.InstanceNorm2d(out_f),
                nn.ReLU(inplace=True),
            ]
            in_f = out_f

        # Output layer
        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(features, out_channels, 7),
            nn.Tanh(),
        ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        # x: [B, 1, in_freq, T]
        T = x.size(3)
        # Adapt frequency axis
        x = self.freq_adapt(x)  # [B, 1, out_freq, T]
        # Ensure time dimension is preserved after up/downsampling
        out = self.model(x)
        # Crop/pad to exact output size
        out = out[:, :, :self.out_freq, :T]
        if out.size(2) < self.out_freq:
            out = F.pad(out, [0, 0, 0, self.out_freq - out.size(2)])
        if out.size(3) < T:
            out = F.pad(out, [0, T - out.size(3)])
        return out


# =============================================================================
# Discriminator: PatchGAN
# =============================================================================

class PatchDiscriminator(nn.Module):
    """PatchGAN discriminator (70×70 receptive field) with spectral normalization."""
    def __init__(self, in_channels=1, use_spectral_norm=True):
        super().__init__()

        norm_fn = nn.utils.spectral_norm if use_spectral_norm else (lambda x: x)

        def block(in_ch, out_ch, stride=2, normalize=True):
            layers = [norm_fn(nn.Conv2d(in_ch, out_ch, 4, stride=stride, padding=1))]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_ch))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(in_channels, 64, normalize=False),
            *block(64, 128),
            *block(128, 256),
            *block(256, 512, stride=1),
            norm_fn(nn.Conv2d(512, 1, 4, padding=1)),
        )

    def forward(self, x):
        # Instance noise for training stability
        if self.training:
            x = x + 0.05 * torch.randn_like(x)
        return self.model(x)


# =============================================================================
# Distribution Matching Loss (same as pix2pix version)
# =============================================================================

class DistributionMatchingLoss(nn.Module):
    """Match per-frequency-bin statistics and global moments."""
    def forward(self, fake, real):
        # Per-frequency-bin mean and std
        fake_mean = fake.mean(dim=(0, 3))
        real_mean = real.mean(dim=(0, 3))
        fake_std = fake.std(dim=(0, 3)) + 1e-6
        real_std = real.std(dim=(0, 3)) + 1e-6

        mean_loss = F.l1_loss(fake_mean, real_mean)
        std_loss = F.l1_loss(fake_std, real_std)

        # Global skewness and kurtosis
        fake_flat = fake.reshape(-1)
        real_flat = real.reshape(-1)
        fake_norm = (fake_flat - fake_flat.mean()) / (fake_flat.std() + 1e-6)
        real_norm = (real_flat - real_flat.mean()) / (real_flat.std() + 1e-6)

        skew_loss = F.l1_loss(fake_norm.pow(3).mean(), real_norm.pow(3).mean())
        kurt_loss = F.l1_loss(fake_norm.pow(4).mean(), real_norm.pow(4).mean())

        return mean_loss + std_loss + 0.5 * (skew_loss + kurt_loss)


# =============================================================================
# Training
# =============================================================================

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # ---------- Compute domain statistics ----------
    print("Computing domain statistics...")
    if_vmin, if_vmax, if_mean, if_std = compute_domain_stats(args.data_dir, 'phase_if', 'train')
    mel_vmin, mel_vmax, mel_mean, mel_std = compute_domain_stats(args.data_dir, 'magnitude', 'train')
    print(f"IF phase:  min={if_vmin:.3f}, max={if_vmax:.3f}, mean={if_mean:.3f}, std={if_std:.3f}")
    print(f"Mel spec:  min={mel_vmin:.3f}, max={mel_vmax:.3f}, mean={mel_mean:.3f}, std={mel_std:.3f}")

    # ---------- Datasets (unpaired!) ----------
    if_train = SpectrogramDataset(args.data_dir, 'phase_if', split='train',
                                  normalize_range=(if_vmin, if_vmax))
    mel_train = SpectrogramDataset(args.data_dir, 'magnitude', split='train',
                                   normalize_range=(mel_vmin, mel_vmax))
    if_val = SpectrogramDataset(args.data_dir, 'phase_if', split='val',
                                normalize_range=(if_vmin, if_vmax))
    mel_val = SpectrogramDataset(args.data_dir, 'magnitude', split='val',
                                 normalize_range=(mel_vmin, mel_vmax))

    print(f"IF train: {len(if_train)}, Mel train: {len(mel_train)}")
    print(f"IF val: {len(if_val)}, Mel val: {len(mel_val)}")

    if_loader = DataLoader(if_train, batch_size=args.batch_size, shuffle=True,
                           num_workers=args.num_workers, pin_memory=True, drop_last=True)
    mel_loader = DataLoader(mel_train, batch_size=args.batch_size, shuffle=True,
                            num_workers=args.num_workers, pin_memory=True, drop_last=True)

    # ---------- Models ----------
    # G_IF2Mel: IF [1,257,204] → mel-like [1,128,204]
    G_IF2Mel = ResNetGenerator(in_freq=257, out_freq=128, n_residual=args.n_residual).to(device)
    # G_Mel2IF: mel [1,128,204] → IF-like [1,257,204]
    G_Mel2IF = ResNetGenerator(in_freq=128, out_freq=257, n_residual=args.n_residual).to(device)
    # D_Mel: judges if [1,128,204] looks like real mel
    D_Mel = PatchDiscriminator(in_channels=1).to(device)
    # D_IF: judges if [1,257,204] looks like real IF
    D_IF = PatchDiscriminator(in_channels=1).to(device)

    # ---------- Losses ----------
    criterion_adv = nn.MSELoss()  # LSGAN
    criterion_cycle = nn.L1Loss()
    criterion_identity = nn.L1Loss()
    criterion_dist = DistributionMatchingLoss()

    # ---------- Optimizers ----------
    opt_G = torch.optim.Adam(
        itertools.chain(G_IF2Mel.parameters(), G_Mel2IF.parameters()),
        lr=args.lr_g, betas=(0.5, 0.999)
    )
    opt_D = torch.optim.Adam(
        itertools.chain(D_Mel.parameters(), D_IF.parameters()),
        lr=args.lr_d, betas=(0.5, 0.999)
    )

    sched_G = CosineAnnealingLR(opt_G, T_max=args.epochs, eta_min=args.lr_g * 0.01)
    sched_D = CosineAnnealingLR(opt_D, T_max=args.epochs, eta_min=args.lr_d * 0.01)

    # Replay buffers
    fake_mel_buffer = ReplayBuffer(max_size=50)
    fake_if_buffer = ReplayBuffer(max_size=50)

    # ---------- Resume ----------
    start_epoch = 0
    best_val_loss = float('inf')
    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        G_IF2Mel.load_state_dict(ckpt['G_IF2Mel'])
        G_Mel2IF.load_state_dict(ckpt['G_Mel2IF'])
        D_Mel.load_state_dict(ckpt['D_Mel'])
        D_IF.load_state_dict(ckpt['D_IF'])
        opt_G.load_state_dict(ckpt['opt_G'])
        opt_D.load_state_dict(ckpt['opt_D'])
        start_epoch = ckpt.get('epoch', 0) + 1
        best_val_loss = ckpt.get('best_val_loss', float('inf'))
        print(f"Resumed from epoch {start_epoch}, best_val_loss={best_val_loss:.4f}")

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # ---------- TensorBoard ----------
    tb_dir = os.path.join(args.checkpoint_dir, 'tb_logs')
    writer = SummaryWriter(log_dir=tb_dir)
    print(f"TensorBoard logs: {tb_dir}")
    print(f"  -> Run: tensorboard --logdir {tb_dir}")
    global_step = start_epoch * min(len(if_loader), len(mel_loader))

    # ---------- Training loop ----------
    epochs_no_improve = 0
    for epoch in range(start_epoch, args.epochs):
        G_IF2Mel.train()
        G_Mel2IF.train()
        D_Mel.train()
        D_IF.train()

        epoch_g_loss = 0.0
        epoch_d_loss = 0.0
        n_batches = min(len(if_loader), len(mel_loader))

        for batch_idx, (real_if, real_mel) in enumerate(zip(if_loader, mel_loader)):
            real_if = real_if.to(device)    # [B, 1, 257, 204] normalized to [-1,1]
            real_mel = real_mel.to(device)  # [B, 1, 128, 204] normalized to [-1,1]

            real_label = torch.ones(real_if.size(0), 1, 1, 1).to(device) * 0.9
            fake_label = torch.ones(real_if.size(0), 1, 1, 1).to(device) * 0.1

            # ================================================================
            # Train Generators
            # ================================================================
            opt_G.zero_grad()

            # --- Identity loss ---
            # G_IF2Mel should be identity on mel input (don't change what's already mel-like)
            idt_mel = G_IF2Mel(F.adaptive_avg_pool2d(real_mel, (257, 204)))  # project mel→257 first
            loss_idt_mel = criterion_identity(idt_mel, real_mel) * args.lambda_identity

            idt_if = G_Mel2IF(F.adaptive_avg_pool2d(real_if, (128, 204)))  # project IF→128 first
            loss_idt_if = criterion_identity(idt_if, real_if) * args.lambda_identity

            # --- Adversarial loss ---
            fake_mel = G_IF2Mel(real_if)        # IF → mel-like [B, 1, 128, 204]
            pred_fake_mel = D_Mel(fake_mel)
            loss_adv_IF2Mel = criterion_adv(pred_fake_mel,
                                             real_label.expand_as(pred_fake_mel))

            fake_if = G_Mel2IF(real_mel)        # mel → IF-like [B, 1, 257, 204]
            pred_fake_if = D_IF(fake_if)
            loss_adv_Mel2IF = criterion_adv(pred_fake_if,
                                             real_label.expand_as(pred_fake_if))

            # --- Cycle consistency loss ---
            # IF → mel-like → IF (must recover original IF)
            recovered_if = G_Mel2IF(fake_mel)   # [B, 1, 257, 204]
            loss_cycle_if = criterion_cycle(recovered_if, real_if) * args.lambda_cycle

            # mel → IF-like → mel (must recover original mel)
            recovered_mel = G_IF2Mel(fake_if)   # [B, 1, 128, 204]
            loss_cycle_mel = criterion_cycle(recovered_mel, real_mel) * args.lambda_cycle

            # --- Distribution matching loss (on fake mel only) ---
            loss_dist = criterion_dist(fake_mel, real_mel) * args.lambda_dist

            # Total generator loss
            loss_G = (loss_adv_IF2Mel + loss_adv_Mel2IF +
                      loss_cycle_if + loss_cycle_mel +
                      loss_idt_mel + loss_idt_if +
                      loss_dist)

            loss_G.backward()
            opt_G.step()

            # ================================================================
            # Train Discriminators
            # ================================================================
            opt_D.zero_grad()

            # D_Mel: real mel vs fake mel
            fake_mel_buffered = fake_mel_buffer.push_and_pop(fake_mel.detach())
            pred_real_mel = D_Mel(real_mel)
            pred_fake_mel = D_Mel(fake_mel_buffered)
            loss_D_Mel = 0.5 * (
                criterion_adv(pred_real_mel, real_label.expand_as(pred_real_mel)) +
                criterion_adv(pred_fake_mel, fake_label.expand_as(pred_fake_mel))
            )

            # D_IF: real IF vs fake IF
            fake_if_buffered = fake_if_buffer.push_and_pop(fake_if.detach())
            pred_real_if = D_IF(real_if)
            pred_fake_if = D_IF(fake_if_buffered)
            loss_D_IF = 0.5 * (
                criterion_adv(pred_real_if, real_label.expand_as(pred_real_if)) +
                criterion_adv(pred_fake_if, fake_label.expand_as(pred_fake_if))
            )

            loss_D = loss_D_Mel + loss_D_IF

            # Only update D if not already too strong (raised threshold — D was too passive)
            if loss_D.item() > 0.1:
                loss_D.backward()
                opt_D.step()

            epoch_g_loss += loss_G.item()
            epoch_d_loss += loss_D.item()

            # --- TensorBoard: per-step losses ---
            writer.add_scalars('Loss/Discriminator', {
                'D_total': loss_D.item(),
                'D_Mel': loss_D_Mel.item(),
                'D_IF': loss_D_IF.item(),
            }, global_step)
            writer.add_scalars('Loss/Generator', {
                'G_total': loss_G.item(),
                'G_adv_IF2Mel': loss_adv_IF2Mel.item(),
                'G_adv_Mel2IF': loss_adv_Mel2IF.item(),
            }, global_step)
            writer.add_scalars('Loss/Cycle', {
                'cycle_IF': loss_cycle_if.item(),
                'cycle_Mel': loss_cycle_mel.item(),
            }, global_step)
            writer.add_scalars('Loss/Regularization', {
                'identity_Mel': loss_idt_mel.item(),
                'identity_IF': loss_idt_if.item(),
                'distribution': loss_dist.item(),
            }, global_step)
            # D health: how well D distinguishes real vs fake (LSGAN: real→1, fake→0)
            writer.add_scalars('Health/D_predictions', {
                'D_on_real': (pred_real_mel.mean().item() + pred_real_if.mean().item()) / 2,
                'D_on_fake': (pred_fake_mel.mean().item() + pred_fake_if.mean().item()) / 2,
            }, global_step)
            global_step += 1

            if batch_idx % 50 == 0:
                print(f"  [{epoch}/{args.epochs}] batch {batch_idx}/{n_batches}  "
                      f"D={loss_D.item():.4f}  G={loss_G.item():.4f}  "
                      f"(adv={loss_adv_IF2Mel.item():.3f}+{loss_adv_Mel2IF.item():.3f} "
                      f"cyc={loss_cycle_if.item():.3f}+{loss_cycle_mel.item():.3f} "
                      f"idt={loss_idt_mel.item():.3f}+{loss_idt_if.item():.3f} "
                      f"dist={loss_dist.item():.3f})")

        sched_G.step()
        sched_D.step()

        avg_g = epoch_g_loss / n_batches
        avg_d = epoch_d_loss / n_batches

        # ---------- Validation ----------
        val_loss = validate(G_IF2Mel, G_Mel2IF, if_val, mel_val,
                            criterion_cycle, device, args)

        print(f"Epoch {epoch}/{args.epochs}  "
              f"Train G={avg_g:.4f} D={avg_d:.4f}  Val cycle={val_loss:.4f}")

        # --- TensorBoard: per-epoch summary ---
        writer.add_scalars('Epoch/Summary', {
            'G_loss': avg_g,
            'D_loss': avg_d,
            'Val_cycle': val_loss,
        }, epoch)
        writer.add_scalar('Epoch/LR_G', sched_G.get_last_lr()[0], epoch)
        writer.add_scalar('Epoch/LR_D', sched_D.get_last_lr()[0], epoch)

        # Log sample images every 5 epochs
        if (epoch + 1) % 5 == 0 or epoch == 0:
            G_IF2Mel.eval()
            G_Mel2IF.eval()
            with torch.no_grad():
                sample_if = if_val[0].unsqueeze(0).to(device)
                sample_mel = mel_val[0].unsqueeze(0).to(device)
                sample_fake_mel = G_IF2Mel(sample_if)
                sample_recovered_if = G_Mel2IF(sample_fake_mel)
                # Normalize to [0,1] for TensorBoard
                def tb_norm(x): return (x - x.min()) / (x.max() - x.min() + 1e-8)
                writer.add_images('Samples/IF_input', tb_norm(sample_if), epoch)
                writer.add_images('Samples/Fake_mel', tb_norm(sample_fake_mel), epoch)
                writer.add_images('Samples/Real_mel', tb_norm(sample_mel), epoch)
                writer.add_images('Samples/Recovered_IF', tb_norm(sample_recovered_if), epoch)
            G_IF2Mel.train()
            G_Mel2IF.train()

        # ---------- Checkpointing ----------
        ckpt_data = {
            'epoch': epoch,
            'G_IF2Mel': G_IF2Mel.state_dict(),
            'G_Mel2IF': G_Mel2IF.state_dict(),
            'D_Mel': D_Mel.state_dict(),
            'D_IF': D_IF.state_dict(),
            'opt_G': opt_G.state_dict(),
            'opt_D': opt_D.state_dict(),
            'best_val_loss': best_val_loss,
            'domain_stats': {
                'if_range': (if_vmin, if_vmax),
                'mel_range': (mel_vmin, mel_vmax),
            },
            'n_residual': args.n_residual,
        }

        torch.save(ckpt_data, os.path.join(args.checkpoint_dir, 'cyclegan_latest.pt'))

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(ckpt_data, os.path.join(args.checkpoint_dir, 'cyclegan_best.pt'))
            print(f"  -> New best val cycle loss: {best_val_loss:.4f}")
        else:
            epochs_no_improve += 1

        if (epoch + 1) % 10 == 0:
            torch.save(ckpt_data,
                       os.path.join(args.checkpoint_dir, f'cyclegan_epoch{epoch}.pt'))

        # Save visualizations every 5 epochs
        if (epoch + 1) % 5 == 0 or epoch == 0:
            save_visualizations(G_IF2Mel, G_Mel2IF, if_val, mel_val,
                                device, args, epoch=epoch)

        if epochs_no_improve >= args.patience:
            print(f"\nEarly stopping at epoch {epoch} "
                  f"(no improvement for {args.patience} epochs)")
            break

    writer.close()
    print(f"\nTraining complete. Best val cycle loss: {best_val_loss:.4f}")


def validate(G_IF2Mel, G_Mel2IF, if_dataset, mel_dataset, criterion_cycle, device, args):
    """Validate cycle consistency on val set."""
    G_IF2Mel.eval()
    G_Mel2IF.eval()

    if_loader = DataLoader(if_dataset, batch_size=args.batch_size, shuffle=False,
                           num_workers=args.num_workers)
    mel_loader = DataLoader(mel_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers)

    total_cycle = 0.0
    count = 0

    with torch.no_grad():
        for real_if, real_mel in zip(if_loader, mel_loader):
            real_if = real_if.to(device)
            real_mel = real_mel.to(device)

            fake_mel = G_IF2Mel(real_if)
            recovered_if = G_Mel2IF(fake_mel)
            total_cycle += criterion_cycle(recovered_if, real_if).item()
            count += 1

    avg_cycle = total_cycle / max(count, 1)
    print(f"  Val: cycle_loss={avg_cycle:.4f}")
    G_IF2Mel.train()
    G_Mel2IF.train()
    return avg_cycle


# =============================================================================
# Visualization
# =============================================================================

def save_visualizations(G_IF2Mel, G_Mel2IF, if_dataset, mel_dataset,
                        device, args, epoch=None):
    """Save side-by-side: IF input → generated mel → real mel (random samples)."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    G_IF2Mel.eval()
    G_Mel2IF.eval()

    save_dir = os.path.join(args.checkpoint_dir, 'visualizations')
    os.makedirs(save_dir, exist_ok=True)

    n_samples = min(8, len(if_dataset), len(mel_dataset))
    if_indices = np.random.choice(len(if_dataset), size=n_samples, replace=False)
    mel_indices = np.random.choice(len(mel_dataset), size=n_samples, replace=False)

    for i in range(n_samples):
        real_if = if_dataset[if_indices[i]].unsqueeze(0).to(device)
        real_mel = mel_dataset[mel_indices[i]].unsqueeze(0).to(device)

        with torch.no_grad():
            fake_mel = G_IF2Mel(real_if)
            recovered_if = G_Mel2IF(fake_mel)
            fake_if = G_Mel2IF(real_mel)
            recovered_mel = G_IF2Mel(fake_if)

        fig, axes = plt.subplots(2, 3, figsize=(18, 8))

        # Row 1: IF → fake mel → recovered IF
        axes[0, 0].imshow(real_if.cpu().squeeze().numpy(), aspect='auto',
                          origin='lower', cmap='RdBu_r')
        axes[0, 0].set_title('Real IF [257×204]')

        axes[0, 1].imshow(fake_mel.cpu().squeeze().numpy(), aspect='auto',
                          origin='lower', cmap='magma')
        axes[0, 1].set_title('Generated Mel-like [128×204]')

        axes[0, 2].imshow(recovered_if.cpu().squeeze().numpy(), aspect='auto',
                          origin='lower', cmap='RdBu_r')
        axes[0, 2].set_title('Recovered IF (cycle) [257×204]')

        # Row 2: real mel → fake IF → recovered mel
        axes[1, 0].imshow(real_mel.cpu().squeeze().numpy(), aspect='auto',
                          origin='lower', cmap='magma')
        axes[1, 0].set_title('Real Mel [128×204]')

        axes[1, 1].imshow(fake_if.cpu().squeeze().numpy(), aspect='auto',
                          origin='lower', cmap='RdBu_r')
        axes[1, 1].set_title('Generated IF-like [257×204]')

        axes[1, 2].imshow(recovered_mel.cpu().squeeze().numpy(), aspect='auto',
                          origin='lower', cmap='magma')
        axes[1, 2].set_title('Recovered Mel (cycle) [128×204]')

        for ax in axes.flat:
            ax.set_xlabel('Time')
            ax.set_ylabel('Frequency')

        epoch_str = f"_epoch{epoch}" if epoch is not None else ""
        fig.suptitle(f'CycleGAN IF↔Mel{epoch_str}  (Row 1: IF→Mel→IF, Row 2: Mel→IF→Mel)')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'cycle{epoch_str}_sample{i}.png'), dpi=150)
        plt.close()

    print(f"  Saved {n_samples} visualizations to {save_dir}/")
    G_IF2Mel.train()
    G_Mel2IF.train()


# =============================================================================
# Inference: translate IF → mel-like for downstream pipeline
# =============================================================================

@torch.no_grad()
def translate_phase_to_mel(generator, if_range, mel_range, phase_if_tensor, device='cuda'):
    """
    Translate IF phase spectrograms to mel-like spectrograms.

    Args:
        generator: trained G_IF2Mel model
        if_range: (vmin, vmax) for IF normalization
        mel_range: (vmin, vmax) for mel denormalization
        phase_if_tensor: [B, 1, 257, 204] raw IF values in [-π, π]
        device: torch device

    Returns:
        mel_like: [B, 1, 128, 204] in mel spectrogram value range
    """
    generator.eval()
    if_vmin, if_vmax = if_range
    mel_vmin, mel_vmax = mel_range

    # Normalize IF to [-1, 1]
    x = phase_if_tensor.to(device)
    x = 2.0 * (x - if_vmin) / (if_vmax - if_vmin + 1e-6) - 1.0

    # Generate mel-like
    fake_mel_norm = generator(x)

    # Denormalize to mel range
    mel_like = (fake_mel_norm + 1.0) / 2.0 * (mel_vmax - mel_vmin + 1e-6) + mel_vmin
    return mel_like


def load_trained_cyclegan(checkpoint_path, device='cuda'):
    """
    Load trained CycleGAN generator for inference.

    Returns:
        generator: G_IF2Mel in eval mode
        if_range: (vmin, vmax) for IF normalization
        mel_range: (vmin, vmax) for mel denormalization
    """
    ckpt = torch.load(checkpoint_path, map_location=device)

    n_residual = ckpt.get('n_residual', 9)
    generator = ResNetGenerator(in_freq=257, out_freq=128,
                                n_residual=n_residual).to(device)
    generator.load_state_dict(ckpt['G_IF2Mel'])
    generator.eval()

    stats = ckpt['domain_stats']
    if_range = stats['if_range']
    mel_range = stats['mel_range']

    return generator, if_range, mel_range


# =============================================================================
# Evaluation
# =============================================================================

def evaluate(args):
    """Load best model, compare distributions, and save visualizations."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ckpt = torch.load(args.resume, map_location=device)
    stats = ckpt['domain_stats']
    if_range = stats['if_range']
    mel_range = stats['mel_range']
    n_residual = ckpt.get('n_residual', 9)

    G_IF2Mel = ResNetGenerator(in_freq=257, out_freq=128, n_residual=n_residual).to(device)
    G_Mel2IF = ResNetGenerator(in_freq=128, out_freq=257, n_residual=n_residual).to(device)
    G_IF2Mel.load_state_dict(ckpt['G_IF2Mel'])
    G_Mel2IF.load_state_dict(ckpt['G_Mel2IF'])
    G_IF2Mel.eval()
    G_Mel2IF.eval()

    if_val = SpectrogramDataset(args.data_dir, 'phase_if', split='val',
                                normalize_range=if_range)
    mel_val = SpectrogramDataset(args.data_dir, 'magnitude', split='val',
                                 normalize_range=mel_range)

    # Generate fake mels and compare distribution
    if_loader = DataLoader(if_val, batch_size=args.batch_size, shuffle=False,
                           num_workers=args.num_workers)
    mel_loader = DataLoader(mel_val, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers)

    all_fake_mel = []
    all_real_mel = []
    all_cycle_loss = []

    with torch.no_grad():
        for real_if, real_mel in zip(if_loader, mel_loader):
            real_if = real_if.to(device)
            real_mel = real_mel.to(device)

            fake_mel = G_IF2Mel(real_if)
            recovered_if = G_Mel2IF(fake_mel)

            cycle_loss = F.l1_loss(recovered_if, real_if).item()
            all_cycle_loss.append(cycle_loss)

            # Denormalize for distribution comparison
            mel_vmin, mel_vmax = mel_range
            fake_mel_denorm = (fake_mel + 1.0) / 2.0 * (mel_vmax - mel_vmin + 1e-6) + mel_vmin
            real_mel_denorm = (real_mel + 1.0) / 2.0 * (mel_vmax - mel_vmin + 1e-6) + mel_vmin

            all_fake_mel.append(fake_mel_denorm.cpu())
            all_real_mel.append(real_mel_denorm.cpu())

    all_fake = torch.cat(all_fake_mel)
    all_real = torch.cat(all_real_mel)

    print(f"\n{'='*60}")
    print(f"CycleGAN Distribution Comparison (Val Set, {len(all_fake)} spectrograms)")
    print(f"{'='*60}")
    print(f"{'Metric':<20} {'Real Mel':>12} {'Generated':>12} {'Diff':>12}")
    print(f"{'-'*60}")

    for name, real_val, fake_val in [
        ('Mean', all_real.mean(), all_fake.mean()),
        ('Std', all_real.std(), all_fake.std()),
        ('Min', all_real.min(), all_fake.min()),
        ('Max', all_real.max(), all_fake.max()),
        ('Median', all_real.median(), all_fake.median()),
    ]:
        diff = abs(fake_val - real_val)
        print(f"{name:<20} {real_val.item():>12.4f} {fake_val.item():>12.4f} {diff.item():>12.4f}")

    real_freq_mean = all_real.mean(dim=(0, 1, 3))
    fake_freq_mean = all_fake.mean(dim=(0, 1, 3))
    freq_mae = (real_freq_mean - fake_freq_mean).abs().mean()
    print(f"\nPer-frequency-bin MAE: {freq_mae:.4f}")
    print(f"Average cycle consistency (IF→mel→IF): {np.mean(all_cycle_loss):.4f}")

    real_flat = all_real.reshape(-1)
    fake_flat = all_fake.reshape(-1)
    real_norm = (real_flat - real_flat.mean()) / (real_flat.std() + 1e-6)
    fake_norm = (fake_flat - fake_flat.mean()) / (fake_flat.std() + 1e-6)
    print(f"Skewness  — Real: {real_norm.pow(3).mean():.4f}, "
          f"Generated: {fake_norm.pow(3).mean():.4f}")
    print(f"Kurtosis  — Real: {real_norm.pow(4).mean():.4f}, "
          f"Generated: {fake_norm.pow(4).mean():.4f}")

    # Save visualizations
    save_visualizations(G_IF2Mel, G_Mel2IF, if_val, mel_val, device, args)


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='CycleGAN: IF Phase ↔ Mel Spectrogram')
    parser.add_argument('--data_dir', type=str, default='./spectrograms')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints/cyclegan')
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--eval_only', action='store_true')
    parser.add_argument('--visualize', action='store_true')

    # Training
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr_g', type=float, default=2e-4)
    parser.add_argument('--lr_d', type=float, default=1e-4,
                        help='D learns slower than G (raised from 5e-5 to push G harder)')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--patience', type=int, default=20)

    # Architecture
    parser.add_argument('--n_residual', type=int, default=9,
                        help='Number of residual blocks in generator')

    # Loss weights
    parser.add_argument('--lambda_cycle', type=float, default=10.0,
                        help='Cycle consistency loss weight (high = preserve info)')
    parser.add_argument('--lambda_identity', type=float, default=2.0,
                        help='Identity loss weight (stabilizes early training, reduced from 5.0)')
    parser.add_argument('--lambda_dist', type=float, default=2.0,
                        help='Distribution matching loss weight')

    args = parser.parse_args()

    if args.visualize:
        assert args.resume, "--resume required for --visualize"
        ckpt = torch.load(args.resume, map_location='cpu')
        stats = ckpt['domain_stats']
        n_residual = ckpt.get('n_residual', 9)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        G_IF2Mel = ResNetGenerator(in_freq=257, out_freq=128, n_residual=n_residual).to(device)
        G_Mel2IF = ResNetGenerator(in_freq=128, out_freq=257, n_residual=n_residual).to(device)
        G_IF2Mel.load_state_dict(ckpt['G_IF2Mel'])
        G_Mel2IF.load_state_dict(ckpt['G_Mel2IF'])
        if_val = SpectrogramDataset(args.data_dir, 'phase_if', split='val',
                                    normalize_range=stats['if_range'])
        mel_val = SpectrogramDataset(args.data_dir, 'magnitude', split='val',
                                     normalize_range=stats['mel_range'])
        save_visualizations(G_IF2Mel, G_Mel2IF, if_val, mel_val, device, args)
    elif args.eval_only:
        assert args.resume, "--resume required for --eval_only"
        evaluate(args)
    else:
        train(args)


if __name__ == '__main__':
    main()
