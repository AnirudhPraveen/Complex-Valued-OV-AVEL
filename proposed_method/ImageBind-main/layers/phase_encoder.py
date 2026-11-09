"""
Phase Encoder module for extracting embeddings from audio phase spectrograms.

Three modes:
  1. 'encoder'      — Standalone CNN (legacy): [B, 10, C, 257, 204] → [B, 10, 1024]
  2. 'if_encoder'   — IF-only CNN: [B, 10, 1, 257, 204] → [B, 10, 1024]
                      Uses IF (instantaneous frequency = ∂phase/∂time) as single channel.
                      Empirical finding: IF shows clear harmonic banding structure that
                      correlates with event content, while raw GD is dominated by noise
                      in low-energy frequency bins (would need Modified GD / cepstral
                      smoothing to be useful). IF directly encodes frequency modulation
                      and varies visibly between event vs background segments.
  3. 'imagebind'    — Project phase to mel shape [1, 128, 204], feed through
                      ImageBind's frozen audio encoder to get [B, 10, 1024]

Literature grounding:
  - Paliwal & Alsteris (2005): Usefulness of phase in speech processing
  - Murthy & Yegnanarayana (2003): Combining evidence from residual phase
  - Note: Raw GD requires Modified GD (MODGD) with cepstral smoothing to be
    useful (Yegnanarayana & Murthy 2000). Without smoothing, raw GD is noisy.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class IFEncoder(nn.Module):
    """
    CNN that maps an Instantaneous Frequency spectrogram to a 1024-dim embedding.

    Uses only IF (∂phase/∂time) as input — empirically shows clear harmonic
    banding that correlates with event content, unlike raw GD which is noisy.
    IF encodes frequency modulation and varies visibly between event/BG segments.

    Input:  [B*T, 1, 257, 204]  (single-channel IF)
    Output: [B*T, 1024]

    Architecture:
        4 conv layers (64→128→256→256) with BatchNorm + ReLU, adaptive pool, linear → 1024
        ~500K params — small enough to train from scratch on OV-AVEL
    """
    def __init__(self, hidden_channels=64, num_layers=4, output_dim=1024):
        super().__init__()
        layers = []
        ch_in = 1  # single channel: IF only
        for i in range(num_layers):
            ch_out = hidden_channels * (2 ** min(i, 2))  # 64, 128, 256, 256
            layers.extend([
                nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(ch_out),
                nn.ReLU(inplace=True),
            ])
            ch_in = ch_out
        self.conv = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(ch_in, output_dim)

    def forward(self, x):
        """
        Args:
            x: [B*T, 1, 257, 204] instantaneous frequency spectrogram
        Returns:
            [B*T, 1024] phase embedding
        """
        x = self.conv(x)           # [B*T, 256, h', w']
        x = self.pool(x)           # [B*T, 256, 1, 1]
        x = x.flatten(1)           # [B*T, 256]
        x = self.fc(x)             # [B*T, 1024]
        return x


class PhaseEncoder(nn.Module):
    """
    Legacy CNN that maps a phase spectrogram to a 1024-dim embedding
    per temporal segment. Supports multi-channel input (IF+GD).

    Input:  [B*T, 1, 257, 204]  (or [B*T, 2, 257, 204] for IF+GD)
    Output: [B*T, 1024]

    Architecture:
        4 conv layers with BatchNorm + ReLU, adaptive pool, linear → 1024
    """
    def __init__(self, in_channels=1, hidden_channels=32, num_layers=4,
                 output_dim=1024):
        super().__init__()
        layers = []
        ch_in = in_channels
        for i in range(num_layers):
            ch_out = hidden_channels * (2 ** min(i, 2))  # 32, 64, 128, 128
            layers.extend([
                nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(ch_out),
                nn.ReLU(inplace=True),
            ])
            ch_in = ch_out
        self.conv = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(ch_in, output_dim)

    def forward(self, x):
        """
        Args:
            x: [B*T, C, 257, 204] phase spectrogram (C=1 for raw, C=2 for IF+GD)
        Returns:
            [B*T, 1024] phase embedding
        """
        x = self.conv(x)           # [B*T, ch_out, h', w']
        x = self.pool(x)           # [B*T, ch_out, 1, 1]
        x = x.flatten(1)           # [B*T, ch_out]
        x = self.fc(x)             # [B*T, 1024]
        return x


class PhaseToMelProjector(nn.Module):
    """
    Projects phase spectrogram [1, 257, 204] → [1, 128, 204] so it can be
    fed through ImageBind's audio encoder (which expects [1, 128, 204]).

    Uses a 1D convolution along the frequency axis (each time step independently).
    """
    def __init__(self, in_freq=257, out_freq=128):
        super().__init__()
        # Linear projection along frequency axis: 257 → 128
        self.proj = nn.Linear(in_freq, out_freq)

    def forward(self, phase):
        """
        Args:
            phase: [B*T, 1, 257, 204]
        Returns:
            projected: [B*T, 1, 128, 204] — same shape as mel spectrogram
        """
        # [B*T, 1, 257, 204] → permute → [B*T, 1, 204, 257]
        x = phase.permute(0, 1, 3, 2)
        x = self.proj(x)  # [B*T, 1, 204, 128]
        x = x.permute(0, 1, 3, 2)  # [B*T, 1, 128, 204]
        return x


def compute_if(phase):
    """
    Compute Instantaneous Frequency (IF) only from raw phase.

    IF = ∂phase/∂time (temporal derivative along time axis)

    IF encodes frequency modulation — shows clear harmonic banding that
    correlates with event content. Visually discriminative between event
    and background segments.

    Args:
        phase: [B*T, 1, 257, 204] raw phase angles in [-π, π]

    Returns:
        inst_freq: [B*T, 1, 257, 204] instantaneous frequency (single channel)
    """
    inst_freq = torch.zeros_like(phase)
    inst_freq[..., 1:] = phase[..., 1:] - phase[..., :-1]
    # Wrap to [-π, π]
    inst_freq = (inst_freq + torch.pi) % (2 * torch.pi) - torch.pi
    return inst_freq


def compute_gd(phase):
    """
    Compute Group Delay (GD) only from raw phase.

    GD = -∂phase/∂freq (negative frequency derivative)

    Note: Raw GD is very noisy in practice. Use Modified GD (MODGD) with
    cepstral smoothing for better results (Yegnanarayana & Murthy 2000).

    Args:
        phase: [B*T, 1, 257, 204] raw phase angles in [-π, π]

    Returns:
        gd: [B*T, 1, 257, 204] group delay (single channel)
    """
    gd = torch.zeros_like(phase)
    gd[..., 1:, :] = -(phase[..., 1:, :] - phase[..., :-1, :])
    # Wrap to [-π, π]
    gd = (gd + torch.pi) % (2 * torch.pi) - torch.pi
    return gd


def compute_if_gd(phase):
    """
    Compute Instantaneous Frequency (IF) and Group Delay (GD) from raw phase.

    IF = ∂phase/∂time   (temporal derivative along time axis)
    GD = -∂phase/∂freq  (negative frequency derivative)

    Both are computed using principal value unwrapping (mod to [-π, π]).

    Args:
        phase: [B*T, 1, 257, 204] raw phase angles in [-π, π]

    Returns:
        if_gd: [B*T, 2, 257, 204] where channel 0 = IF, channel 1 = GD
    """
    # IF: temporal derivative (along last dim = time frames)
    if_diff = torch.zeros_like(phase)
    if_diff[..., 1:] = phase[..., 1:] - phase[..., :-1]
    # Wrap to [-π, π]
    if_diff = (if_diff + torch.pi) % (2 * torch.pi) - torch.pi

    # GD: negative frequency derivative (along dim=-2 = freq bins)
    gd_diff = torch.zeros_like(phase)
    gd_diff[..., 1:, :] = -(phase[..., 1:, :] - phase[..., :-1, :])
    gd_diff = (gd_diff + torch.pi) % (2 * torch.pi) - torch.pi

    # Stack: [B*T, 2, 257, 204]
    return torch.cat([if_diff, gd_diff], dim=1)
