"""
Pix2Pix GAN: Translate IF phase spectrograms → mel-like spectrograms.

Goal: Learn a mapping from IF phase domain [1, 257, 204] to mel spectrogram
domain [1, 128, 204] so that the pretrained ImageBind audio encoder can
extract meaningful features from phase information without LoRA or a
separate encoder.

Architecture:
    Generator:  U-Net with skip connections (257×204 → 128×204)
    Discriminator: PatchGAN (70×70 receptive field)

Losses:
    1. Adversarial (LSGAN): Discriminator enforces perceptual quality
    2. L1 reconstruction: Pixel-wise match to target mel spectrogram
    3. Distribution matching: Penalizes divergence in per-channel mean/std
       so the output follows the mel spectrogram's log-energy distribution

Usage:
    # Train on extracted spectrograms
    python pix2pix_phase2mel.py --data_dir ./spectrograms --epochs 200

    # Resume training
    python pix2pix_phase2mel.py --data_dir ./spectrograms --resume checkpoints/pix2pix_latest.pt

    # Inference only
    python pix2pix_phase2mel.py --data_dir ./spectrograms --eval_only --resume checkpoints/pix2pix_best.pt
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR


# =============================================================================
# Dataset
# =============================================================================

class PhaseToMelDataset(Dataset):
    """
    Loads paired IF phase spectrograms and mel spectrograms from .npy files.

    Expects:
        data_dir/phase_if/   — [257, 204] .npy files
        data_dir/magnitude/  — [128, 204] .npy files
    File names must match between the two directories.
    """
    def __init__(self, data_dir, split='train', normalize_mel=True):
        self.phase_dir = os.path.join(data_dir, 'phase_if')
        self.mel_dir = os.path.join(data_dir, 'magnitude')
        self.normalize_mel = normalize_mel

        # Find matched pairs
        phase_files = set(os.listdir(self.phase_dir))
        mel_files = set(os.listdir(self.mel_dir))
        common = sorted(phase_files & mel_files)

        # Filter by split prefix if present in filenames
        if split:
            common = [f for f in common if f.startswith(split + '_')]

        self.filenames = common
        assert len(self.filenames) > 0, (
            f"No matched pairs found for split='{split}' in {data_dir}. "
            f"Phase files: {len(phase_files)}, Mel files: {len(mel_files)}"
        )

        # Compute mel statistics from training set for distribution matching
        self.mel_mean, self.mel_std = self._compute_mel_stats()

    def _compute_mel_stats(self):
        """Compute per-frequency-bin mean and std from the mel spectrograms."""
        # Sample up to 500 files for efficiency
        sample_files = self.filenames[:500]
        mels = []
        for f in sample_files:
            mel = np.load(os.path.join(self.mel_dir, f))  # [128, 204]
            mels.append(mel)
        mels = np.stack(mels)  # [N, 128, 204]
        mean = mels.mean(axis=(0, 2), keepdims=True)  # [1, 128, 1]
        std = mels.std(axis=(0, 2), keepdims=True) + 1e-6
        return (
            torch.from_numpy(mean.squeeze()).float(),   # [128]
            torch.from_numpy(std.squeeze()).float(),     # [128]
        )

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        fname = self.filenames[idx]

        phase_if = np.load(os.path.join(self.phase_dir, fname))   # [257, 204]
        mel = np.load(os.path.join(self.mel_dir, fname))          # [128, 204]

        # Normalize IF phase from [-π, π] to [-1, 1]
        phase_if = phase_if / np.pi

        # Convert to tensors with channel dim
        phase_if = torch.from_numpy(phase_if).float().unsqueeze(0)  # [1, 257, 204]
        mel = torch.from_numpy(mel).float().unsqueeze(0)            # [1, 128, 204]

        return phase_if, mel


# =============================================================================
# Generator: U-Net (257×204 → 128×204)
# =============================================================================

class UNetDown(nn.Module):
    """Encoder block: Conv → [BN] → LeakyReLU"""
    def __init__(self, in_ch, out_ch, normalize=True, kernel_size=4, stride=2, padding=1):
        super().__init__()
        layers = [nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, bias=not normalize)]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_ch))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class UNetUp(nn.Module):
    """Decoder block: ConvTranspose → BN → [Dropout] → ReLU + skip connection"""
    def __init__(self, in_ch, out_ch, dropout=0.0):
        super().__init__()
        layers = [
            nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(out_ch),
        ]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.ReLU(inplace=True))
        self.model = nn.Sequential(*layers)

    def forward(self, x, skip):
        x = self.model(x)
        # Handle size mismatch from odd dimensions
        diff_h = skip.size(2) - x.size(2)
        diff_w = skip.size(3) - x.size(3)
        if diff_h != 0 or diff_w != 0:
            x = F.pad(x, [0, diff_w, 0, diff_h])
        return torch.cat([x, skip], dim=1)


class Generator(nn.Module):
    """
    U-Net generator for IF phase → mel spectrogram translation.

    Input:  [B, 1, 257, 204]  (IF phase spectrogram, normalized to [-1, 1])
    Output: [B, 1, 128, 204]  (mel-like spectrogram matching mel distribution)

    The frequency axis changes from 257 → 128, handled by an adaptive
    input projection + careful encoder/decoder sizing.
    """
    def __init__(self, in_channels=1, out_channels=1, features=64):
        super().__init__()
        # Input projection: [1, 257, 204] → [1, 128, 204] via learned freq projection
        # This aligns the frequency axis before the U-Net processes it
        self.freq_proj = nn.Sequential(
            nn.Conv2d(in_channels, features, kernel_size=(3, 1), stride=1, padding=(1, 0)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AdaptiveAvgPool2d((128, 204)),  # Resize freq axis: 257 → 128
            nn.Conv2d(features, in_channels, kernel_size=1),
        )

        # U-Net encoder (all operating on 128×204)
        # Each downsamples by 2× in both dims
        self.down1 = UNetDown(in_channels, 64, normalize=False)  # 64×102
        self.down2 = UNetDown(64, 128)                            # 32×51
        self.down3 = UNetDown(128, 256)                           # 16×25 (25 due to odd 51)
        self.down4 = UNetDown(256, 512)                           # 8×12 (12 due to odd 25)
        self.down5 = UNetDown(512, 512)                           # 4×6
        self.down6 = UNetDown(512, 512)                           # 2×3
        self.down7 = UNetDown(512, 512, normalize=False)          # 1×1 (bottleneck)

        # U-Net decoder (with skip connections, channel = in + skip)
        self.up1 = UNetUp(512, 512, dropout=0.5)     # 2×3,   cat → 1024
        self.up2 = UNetUp(1024, 512, dropout=0.5)    # 4×6,   cat → 1024
        self.up3 = UNetUp(1024, 512, dropout=0.5)    # 8×12,  cat → 1024
        self.up4 = UNetUp(1024, 256)                  # 16×25, cat → 512
        self.up5 = UNetUp(512, 128)                   # 32×51, cat → 256
        self.up6 = UNetUp(256, 64)                    # 64×102,cat → 128

        # Final: upsample to 128×204 and project to output channels
        self.final = nn.Sequential(
            nn.ConvTranspose2d(128, out_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),  # Output in [-1, 1], will be rescaled to mel distribution
        )

    def forward(self, x):
        """
        Args:
            x: [B, 1, 257, 204] normalized IF phase spectrogram
        Returns:
            [B, 1, 128, 204] generated mel-like spectrogram (raw Tanh output in [-1,1])
        """
        # Project frequency axis: 257 → 128
        x = self.freq_proj(x)  # [B, 1, 128, 204]

        # Encoder
        d1 = self.down1(x)    # [B, 64, 64, 102]
        d2 = self.down2(d1)   # [B, 128, 32, 51]
        d3 = self.down3(d2)   # [B, 256, 16, 25]
        d4 = self.down4(d3)   # [B, 512, 8, 12]
        d5 = self.down5(d4)   # [B, 512, 4, 6]
        d6 = self.down6(d5)   # [B, 512, 2, 3]
        d7 = self.down7(d6)   # [B, 512, 1, 1]

        # Decoder with skip connections
        u1 = self.up1(d7, d6)  # [B, 1024, 2, 3]
        u2 = self.up2(u1, d5)  # [B, 1024, 4, 6]
        u3 = self.up3(u2, d4)  # [B, 1024, 8, 12]
        u4 = self.up4(u3, d3)  # [B, 512, 16, 25]
        u5 = self.up5(u4, d2)  # [B, 256, 32, 51]
        u6 = self.up6(u5, d1)  # [B, 128, 64, 102]

        out = self.final(u6)   # [B, 1, 128, 204]

        # Crop/pad to exact target size
        out = out[:, :, :128, :204]
        return out


# =============================================================================
# Discriminator: PatchGAN
# =============================================================================

class Discriminator(nn.Module):
    """
    PatchGAN discriminator (70×70 receptive field).

    Takes concatenation of (IF input projected to 128 freq bins, mel target/fake)
    as conditional GAN input.

    Input:  [B, 2, 128, 204] (condition + real/fake concatenated on channel dim)
    Output: [B, 1, H', W'] patch-wise real/fake prediction
    """
    def __init__(self, in_channels=2):
        super().__init__()
        def disc_block(in_ch, out_ch, stride=2, normalize=True):
            layers = [nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=stride, padding=1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_ch))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return nn.Sequential(*layers)

        self.model = nn.Sequential(
            disc_block(in_channels, 64, normalize=False),  # 64×102
            disc_block(64, 128),                            # 32×51
            disc_block(128, 256),                           # 16×25
            disc_block(256, 512, stride=1),                 # 16×25
            nn.Conv2d(512, 1, kernel_size=4, padding=1),    # 16×25 → patch predictions
        )

    def forward(self, condition, target):
        """
        Args:
            condition: [B, 1, 128, 204] projected IF phase (generator input after freq_proj)
            target:    [B, 1, 128, 204] real or fake mel spectrogram
        Returns:
            [B, 1, H', W'] patch predictions
        """
        # Instance noise: makes it harder for D to memorize exact pixel patterns
        # Decays during training (called with decreasing noise_std)
        if self.training:
            condition = condition + 0.05 * torch.randn_like(condition)
            target = target + 0.05 * torch.randn_like(target)
        x = torch.cat([condition, target], dim=1)  # [B, 2, 128, 204]
        return self.model(x)


# =============================================================================
# Distribution Matching Loss
# =============================================================================

class DistributionMatchingLoss(nn.Module):
    """
    Penalizes divergence between the generated output's statistics and the
    real mel spectrogram's statistics.

    Matches:
        1. Per-frequency-bin mean (captures the spectral energy envelope)
        2. Per-frequency-bin std  (captures the dynamic range per band)
        3. Global histogram shape via moment matching (skewness, kurtosis)

    This ensures the generator output doesn't just minimize pixel error but
    actually follows the characteristic log-energy distribution of mel spectrograms.
    """
    def __init__(self):
        super().__init__()

    def forward(self, fake, real):
        """
        Args:
            fake: [B, 1, 128, 204] generated mel-like spectrogram
            real: [B, 1, 128, 204] target mel spectrogram
        """
        # Per-frequency-bin statistics (average over batch and time)
        fake_mean = fake.mean(dim=(0, 3))  # [1, 128]
        real_mean = real.mean(dim=(0, 3))  # [1, 128]
        fake_std = fake.std(dim=(0, 3)) + 1e-6
        real_std = real.std(dim=(0, 3)) + 1e-6

        # Mean matching: spectral envelope should match
        mean_loss = F.l1_loss(fake_mean, real_mean)

        # Std matching: dynamic range per frequency band should match
        std_loss = F.l1_loss(fake_std, real_std)

        # Global moment matching (skewness and kurtosis)
        # Ensures the overall value distribution shape matches mel's log-energy distribution
        fake_flat = fake.reshape(-1)
        real_flat = real.reshape(-1)

        # Normalize to zero mean, unit variance for moment comparison
        fake_norm = (fake_flat - fake_flat.mean()) / (fake_flat.std() + 1e-6)
        real_norm = (real_flat - real_flat.mean()) / (real_flat.std() + 1e-6)

        # Skewness (3rd moment): mel spectrograms have negative skew (more low-energy frames)
        fake_skew = fake_norm.pow(3).mean()
        real_skew = real_norm.pow(3).mean()
        skew_loss = F.l1_loss(fake_skew, real_skew)

        # Kurtosis (4th moment): captures tail heaviness
        fake_kurt = fake_norm.pow(4).mean()
        real_kurt = real_norm.pow(4).mean()
        kurt_loss = F.l1_loss(fake_kurt, real_kurt)

        return mean_loss + std_loss + 0.5 * (skew_loss + kurt_loss)


# =============================================================================
# Mel Normalizer: maps between raw mel values and [-1, 1] for Tanh output
# =============================================================================

class MelNormalizer:
    """
    Normalizes mel spectrograms to [-1, 1] range (for Tanh generator output)
    and denormalizes back to the original mel distribution.

    Statistics are computed from the training set.
    """
    def __init__(self, dataset):
        self.mel_mean = dataset.mel_mean   # [128]
        self.mel_std = dataset.mel_std     # [128]

        # Compute global min/max from mean±3std for robust normalization
        self.mel_min = (self.mel_mean - 3 * self.mel_std).min()
        self.mel_max = (self.mel_mean + 3 * self.mel_std).max()

    def normalize(self, mel):
        """Raw mel values → [-1, 1]"""
        return 2.0 * (mel - self.mel_min) / (self.mel_max - self.mel_min + 1e-6) - 1.0

    def denormalize(self, mel_norm):
        """[-1, 1] → raw mel values"""
        return (mel_norm + 1.0) / 2.0 * (self.mel_max - self.mel_min + 1e-6) + self.mel_min

    def to(self, device):
        self.mel_mean = self.mel_mean.to(device)
        self.mel_std = self.mel_std.to(device)
        if isinstance(self.mel_min, torch.Tensor):
            self.mel_min = self.mel_min.to(device)
            self.mel_max = self.mel_max.to(device)
        return self


# =============================================================================
# Training
# =============================================================================

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # ---------- Data ----------
    train_dataset = PhaseToMelDataset(args.data_dir, split='train')
    val_dataset = PhaseToMelDataset(args.data_dir, split='val')
    print(f"Train: {len(train_dataset)} pairs, Val: {len(val_dataset)} pairs")

    normalizer = MelNormalizer(train_dataset).to(device)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    # ---------- Models ----------
    generator = Generator(in_channels=1, out_channels=1, features=64).to(device)
    discriminator = Discriminator(in_channels=2).to(device)

    # ---------- Losses ----------
    criterion_adv = nn.MSELoss()        # LSGAN
    criterion_l1 = nn.L1Loss()
    criterion_dist = DistributionMatchingLoss()

    # ---------- Optimizers ----------
    # D learns slower to prevent it from overpowering G
    opt_G = torch.optim.Adam(generator.parameters(), lr=args.lr_g, betas=(0.5, 0.999))
    opt_D = torch.optim.Adam(discriminator.parameters(), lr=args.lr_d * 0.25, betas=(0.5, 0.999))

    sched_G = CosineAnnealingLR(opt_G, T_max=args.epochs, eta_min=args.lr_g * 0.01)
    sched_D = CosineAnnealingLR(opt_D, T_max=args.epochs, eta_min=args.lr_d * 0.01)

    # ---------- Resume ----------
    start_epoch = 0
    best_val_loss = float('inf')
    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        generator.load_state_dict(ckpt['generator'])
        discriminator.load_state_dict(ckpt['discriminator'])
        opt_G.load_state_dict(ckpt['opt_G'])
        opt_D.load_state_dict(ckpt['opt_D'])
        start_epoch = ckpt.get('epoch', 0) + 1
        best_val_loss = ckpt.get('best_val_loss', float('inf'))
        print(f"Resumed from epoch {start_epoch}, best_val_loss={best_val_loss:.4f}")

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # ---------- Training loop ----------
    epochs_no_improve = 0
    for epoch in range(start_epoch, args.epochs):
        generator.train()
        discriminator.train()

        epoch_g_loss = 0.0
        epoch_d_loss = 0.0

        for batch_idx, (phase_if, mel_real) in enumerate(train_loader):
            phase_if = phase_if.to(device)           # [B, 1, 257, 204] in [-1, 1]
            mel_real = mel_real.to(device)            # [B, 1, 128, 204] raw mel values

            # Normalize mel to [-1, 1] for the generator's Tanh output
            mel_real_norm = normalizer.normalize(mel_real)

            # Get generator's projected input for discriminator conditioning
            with torch.no_grad():
                cond = generator.freq_proj(phase_if)  # [B, 1, 128, 204]

            # ---- Train Discriminator ----
            mel_fake_norm = generator(phase_if).detach()

            pred_real = discriminator(cond, mel_real_norm)
            pred_fake = discriminator(cond, mel_fake_norm)

            # Label smoothing: real=0.9, fake=0.1 (prevents D from being overconfident)
            real_label = torch.ones_like(pred_real) * 0.9
            fake_label = torch.ones_like(pred_fake) * 0.1

            loss_D = 0.5 * (criterion_adv(pred_real, real_label) +
                            criterion_adv(pred_fake, fake_label))

            # Only update D if it's not already too strong
            # When D_loss < 0.01, D is overconfident — skip update to let G catch up
            if loss_D.item() > 0.05:
                opt_D.zero_grad()
                loss_D.backward()
                opt_D.step()

            # ---- Train Generator ----
            mel_fake_norm = generator(phase_if)
            cond = generator.freq_proj(phase_if)  # recompute with grad

            pred_fake = discriminator(cond, mel_fake_norm)

            # Adversarial loss: fool discriminator
            loss_G_adv = criterion_adv(pred_fake, torch.ones_like(pred_fake))

            # L1 reconstruction loss
            loss_G_l1 = criterion_l1(mel_fake_norm, mel_real_norm)

            # Distribution matching loss (on denormalized values for real statistics)
            mel_fake_denorm = normalizer.denormalize(mel_fake_norm)
            loss_G_dist = criterion_dist(mel_fake_denorm, mel_real)

            # Total generator loss
            loss_G = (loss_G_adv +
                      args.lambda_l1 * loss_G_l1 +
                      args.lambda_dist * loss_G_dist)

            opt_G.zero_grad()
            loss_G.backward()
            opt_G.step()

            epoch_g_loss += loss_G.item()
            epoch_d_loss += loss_D.item()

            if batch_idx % 50 == 0:
                print(f"  [{epoch}/{args.epochs}] batch {batch_idx}/{len(train_loader)}  "
                      f"D={loss_D.item():.4f}  G={loss_G.item():.4f}  "
                      f"(adv={loss_G_adv.item():.3f} l1={loss_G_l1.item():.3f} "
                      f"dist={loss_G_dist.item():.3f})")

        sched_G.step()
        sched_D.step()

        avg_g = epoch_g_loss / len(train_loader)
        avg_d = epoch_d_loss / len(train_loader)

        # ---------- Validation ----------
        val_loss = validate(generator, val_loader, normalizer, criterion_l1,
                            criterion_dist, device, args)

        print(f"Epoch {epoch}/{args.epochs}  "
              f"Train G={avg_g:.4f} D={avg_d:.4f}  Val L1={val_loss:.4f}")

        # ---------- Checkpointing ----------
        ckpt_data = {
            'epoch': epoch,
            'generator': generator.state_dict(),
            'discriminator': discriminator.state_dict(),
            'opt_G': opt_G.state_dict(),
            'opt_D': opt_D.state_dict(),
            'best_val_loss': best_val_loss,
            'normalizer': {
                'mel_min': normalizer.mel_min,
                'mel_max': normalizer.mel_max,
                'mel_mean': normalizer.mel_mean,
                'mel_std': normalizer.mel_std,
            },
        }

        torch.save(ckpt_data, os.path.join(args.checkpoint_dir, 'pix2pix_latest.pt'))

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(ckpt_data, os.path.join(args.checkpoint_dir, 'pix2pix_best.pt'))
            print(f"  -> New best val loss: {best_val_loss:.4f}")
        else:
            epochs_no_improve += 1

        if (epoch + 1) % 20 == 0:
            torch.save(ckpt_data,
                       os.path.join(args.checkpoint_dir, f'pix2pix_epoch{epoch}.pt'))

        if epochs_no_improve >= args.patience:
            print(f"\nEarly stopping at epoch {epoch} (no improvement for {args.patience} epochs)")
            break

    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")


def validate(generator, val_loader, normalizer, criterion_l1, criterion_dist,
             device, args):
    """Run validation and return average L1 loss."""
    generator.eval()
    total_l1 = 0.0
    total_dist = 0.0
    count = 0

    with torch.no_grad():
        for phase_if, mel_real in val_loader:
            phase_if = phase_if.to(device)
            mel_real = mel_real.to(device)

            mel_real_norm = normalizer.normalize(mel_real)
            mel_fake_norm = generator(phase_if)

            total_l1 += criterion_l1(mel_fake_norm, mel_real_norm).item()

            mel_fake_denorm = normalizer.denormalize(mel_fake_norm)
            total_dist += criterion_dist(mel_fake_denorm, mel_real).item()
            count += 1

    avg_l1 = total_l1 / max(count, 1)
    avg_dist = total_dist / max(count, 1)
    print(f"  Val: L1={avg_l1:.4f}, Dist={avg_dist:.4f}")
    generator.train()
    return avg_l1


# =============================================================================
# Inference: translate IF → mel-like for downstream use
# =============================================================================

@torch.no_grad()
def translate_phase_to_mel(generator, normalizer, phase_if_tensor, device='cuda'):
    """
    Translate a batch of IF phase spectrograms to mel-like spectrograms.

    Args:
        generator: trained Generator model
        normalizer: MelNormalizer with training set statistics
        phase_if_tensor: [B, 1, 257, 204] raw IF values in [-π, π]
        device: torch device

    Returns:
        mel_like: [B, 1, 128, 204] in the mel spectrogram's value distribution
    """
    generator.eval()
    # Normalize IF to [-1, 1]
    phase_input = (phase_if_tensor / np.pi).to(device)
    mel_fake_norm = generator(phase_input)
    mel_like = normalizer.denormalize(mel_fake_norm)
    return mel_like


def load_trained_generator(checkpoint_path, device='cuda'):
    """
    Load a trained generator and normalizer from checkpoint.

    Returns:
        generator: Generator model in eval mode
        normalizer: MelNormalizer with saved statistics
    """
    ckpt = torch.load(checkpoint_path, map_location=device)

    generator = Generator(in_channels=1, out_channels=1, features=64).to(device)
    generator.load_state_dict(ckpt['generator'])
    generator.eval()

    # Reconstruct normalizer from saved stats
    norm_data = ckpt['normalizer']
    normalizer = MelNormalizer.__new__(MelNormalizer)
    normalizer.mel_mean = norm_data['mel_mean'].to(device)
    normalizer.mel_std = norm_data['mel_std'].to(device)
    normalizer.mel_min = norm_data['mel_min'].to(device) if isinstance(norm_data['mel_min'], torch.Tensor) else norm_data['mel_min']
    normalizer.mel_max = norm_data['mel_max'].to(device) if isinstance(norm_data['mel_max'], torch.Tensor) else norm_data['mel_max']

    return generator, normalizer


# =============================================================================
# Evaluation: print distribution comparison
# =============================================================================

def evaluate(args):
    """Load best model and compare distributions on val set."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    generator, normalizer = load_trained_generator(args.resume, device)

    val_dataset = PhaseToMelDataset(args.data_dir, split='val')
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers)

    all_fake = []
    all_real = []

    with torch.no_grad():
        for phase_if, mel_real in val_loader:
            phase_if = phase_if.to(device)
            mel_fake_norm = generator(phase_if)
            mel_fake = normalizer.denormalize(mel_fake_norm)

            all_fake.append(mel_fake.cpu())
            all_real.append(mel_real)

    all_fake = torch.cat(all_fake)
    all_real = torch.cat(all_real)

    print(f"\n{'='*60}")
    print(f"Distribution Comparison (Val Set, {len(all_fake)} spectrograms)")
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

    # Per-frequency-bin comparison
    real_freq_mean = all_real.mean(dim=(0, 1, 3))  # [128]
    fake_freq_mean = all_fake.mean(dim=(0, 1, 3))  # [128]
    freq_mae = (real_freq_mean - fake_freq_mean).abs().mean()
    print(f"\nPer-frequency-bin MAE: {freq_mae:.4f}")

    # Skewness and kurtosis
    real_flat = all_real.reshape(-1)
    fake_flat = all_fake.reshape(-1)
    real_norm = (real_flat - real_flat.mean()) / (real_flat.std() + 1e-6)
    fake_norm = (fake_flat - fake_flat.mean()) / (fake_flat.std() + 1e-6)

    print(f"Skewness  — Real: {real_norm.pow(3).mean():.4f}, "
          f"Generated: {fake_norm.pow(3).mean():.4f}")
    print(f"Kurtosis  — Real: {real_norm.pow(4).mean():.4f}, "
          f"Generated: {fake_norm.pow(4).mean():.4f}")

    # Save side-by-side visualizations
    visualize(args, generator, normalizer, device)


def visualize(args, generator=None, normalizer=None, device=None):
    """Save side-by-side comparisons: IF phase → Generated mel → Real mel."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    if generator is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        generator, normalizer = load_trained_generator(args.resume, device)

    val_dataset = PhaseToMelDataset(args.data_dir, split='val')

    save_dir = os.path.join(args.checkpoint_dir, 'visualizations')
    os.makedirs(save_dir, exist_ok=True)

    # Pick 10 random samples
    indices = np.random.choice(len(val_dataset), size=min(10, len(val_dataset)), replace=False)

    for i, idx in enumerate(indices):
        phase_if, mel_real = val_dataset[idx]  # [1, 257, 204], [1, 128, 204]

        # Generate
        with torch.no_grad():
            phase_input = phase_if.unsqueeze(0).to(device)  # [1, 1, 257, 204]
            mel_fake_norm = generator(phase_input)
            mel_fake = normalizer.denormalize(mel_fake_norm).cpu().squeeze()  # [128, 204]

        phase_np = phase_if.squeeze().numpy()    # [257, 204]
        mel_real_np = mel_real.squeeze().numpy()  # [128, 204]
        mel_fake_np = mel_fake.numpy()            # [128, 204]

        # Use same color range for real and generated mel
        vmin = min(mel_real_np.min(), mel_fake_np.min())
        vmax = max(mel_real_np.max(), mel_fake_np.max())

        fig, axes = plt.subplots(1, 3, figsize=(18, 4))

        axes[0].imshow(phase_np, aspect='auto', origin='lower', cmap='RdBu_r')
        axes[0].set_title('IF Phase Input [257×204]')
        axes[0].set_xlabel('Time')
        axes[0].set_ylabel('Frequency')

        im1 = axes[1].imshow(mel_fake_np, aspect='auto', origin='lower',
                             cmap='magma', vmin=vmin, vmax=vmax)
        axes[1].set_title('Generated Mel [128×204]')
        axes[1].set_xlabel('Time')
        axes[1].set_ylabel('Mel bin')

        im2 = axes[2].imshow(mel_real_np, aspect='auto', origin='lower',
                             cmap='magma', vmin=vmin, vmax=vmax)
        axes[2].set_title('Real Mel [128×204]')
        axes[2].set_xlabel('Time')
        axes[2].set_ylabel('Mel bin')

        fig.colorbar(im2, ax=axes[2], fraction=0.046)
        fig.colorbar(im1, ax=axes[1], fraction=0.046)

        fname = val_dataset.filenames[idx].replace('.npy', '')
        fig.suptitle(f'Sample: {fname}', fontsize=10)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'comparison_{i}_{fname}.png'), dpi=150)
        plt.close()

    print(f"\nSaved {len(indices)} visualizations to {save_dir}/")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Pix2Pix: IF Phase → Mel Spectrogram')
    parser.add_argument('--data_dir', type=str, default='./spectrograms',
                        help='Directory with magnitude/ and phase_if/ subdirs')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints/pix2pix',
                        help='Directory to save checkpoints')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--eval_only', action='store_true',
                        help='Run evaluation only (requires --resume)')
    parser.add_argument('--visualize', action='store_true',
                        help='Save side-by-side visualizations only (requires --resume)')

    # Training hyperparameters
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr_g', type=float, default=2e-4,
                        help='Generator learning rate')
    parser.add_argument('--lr_d', type=float, default=2e-4,
                        help='Discriminator learning rate')
    parser.add_argument('--lambda_l1', type=float, default=100.0,
                        help='Weight for L1 reconstruction loss')
    parser.add_argument('--lambda_dist', type=float, default=10.0,
                        help='Weight for distribution matching loss')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--patience', type=int, default=10,
                        help='Early stopping: stop if val loss does not improve for N epochs')

    args = parser.parse_args()

    if args.visualize:
        assert args.resume, "--resume required for --visualize"
        visualize(args)
    elif args.eval_only:
        assert args.resume, "--resume required for --eval_only"
        evaluate(args)
    else:
        train(args)


if __name__ == '__main__':
    main()
