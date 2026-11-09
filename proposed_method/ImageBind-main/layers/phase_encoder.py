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
