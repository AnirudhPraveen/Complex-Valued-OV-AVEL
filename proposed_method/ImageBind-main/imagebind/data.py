import os
#!/usr/bin/env python3
# Portions Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import math

import torch
import torch.nn as nn
import torchaudio
import soundfile as sf
from PIL import Image
from pytorchvideo import transforms as pv_transforms
from pytorchvideo.data.clip_sampling import ConstantClipsPerVideoSampler
from pytorchvideo.data.encoded_video import EncodedVideo
from torchvision import transforms
from torchvision.transforms._transforms_video import NormalizeVideo

from imagebind.models.multimodal_preprocessors import SimpleTokenizer

import numpy as np
import pdb

DEFAULT_AUDIO_FRAME_SHIFT_MS = 10  # in milliseconds

BPE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "bpe", "bpe_simple_vocab_16e6.txt.gz")

print(np.__version__)


# ---------------------------------------------------------------------------
# Phase extraction from kaldi's FFT — uses kaldi's own internal functions.
# Mel spectrogram is IDENTICAL to kaldi.fbank() (verified, diff = 0.0).
# To disable: set USE_PHASE_EXTRACTION = False (original kaldi path).
# ---------------------------------------------------------------------------
USE_PHASE_EXTRACTION = True

# Phase target length (257 freq bins, padded/truncated to this many frames)
PHASE_TARGET_LENGTH = 204

def _fbank_with_phase(waveform):
    """
    Same output as kaldi.fbank() but also returns phase.
    Uses kaldi's own internal functions — no approximation.
    """
    from torchaudio.compliance.kaldi import (
        _get_waveform_and_window_properties,
        _get_window,
        get_mel_banks,
        _get_epsilon,
    )
    device, dtype = waveform.device, waveform.dtype
    waveform_proc, window_shift, window_size, padded_window_size = \
        _get_waveform_and_window_properties(
            waveform, -1, 16000.0, 10.0, 25.0, True, 0.97)
    strided_input, signal_log_energy = _get_window(
        waveform_proc, padded_window_size, window_size, window_shift,
        "hanning", 0.42, True, True, 1.0, 0.0, True, 0.97)

    complex_fft = torch.fft.rfft(strided_input)   # [n_frames, 257] complex
    magnitude = complex_fft.abs()
    phase = complex_fft.angle()                    # [n_frames, 257]
    power = magnitude.pow(2.0)

    mel_banks, _ = get_mel_banks(128, padded_window_size, 16000.0, 20.0, 0.0, 100.0, -500.0, 1.0)
    mel_banks = mel_banks.to(device=device, dtype=dtype)
    mel_banks = torch.nn.functional.pad(mel_banks, (0, 1), mode="constant", value=0)
    mel_energies = torch.mm(power, mel_banks.T)
    mel_energies = torch.max(mel_energies, _get_epsilon(device, dtype)).log()

    return mel_energies, phase


def _fbank_complex(waveform):
    """
    Extract real and imaginary mel spectrograms from audio FFT.
    Uses |real| and |imag| (absolute value) → mel → log.

    Returns:
        real_mel: [n_frames, 128] — mel spectrogram of cos (real) component
        imag_mel: [n_frames, 128] — mel spectrogram of sin (imaginary) component
    """
    from torchaudio.compliance.kaldi import (
        _get_waveform_and_window_properties,
        _get_window,
        get_mel_banks,
        _get_epsilon,
    )
    device, dtype = waveform.device, waveform.dtype
    waveform_proc, window_shift, window_size, padded_window_size = \
        _get_waveform_and_window_properties(
            waveform, -1, 16000.0, 10.0, 25.0, True, 0.97)
    strided_input, signal_log_energy = _get_window(
        waveform_proc, padded_window_size, window_size, window_shift,
        "hanning", 0.42, True, True, 1.0, 0.0, True, 0.97)

    complex_fft = torch.fft.rfft(strided_input)   # [n_frames, 257] complex
    real_abs = complex_fft.real.abs()               # [n_frames, 257]
    imag_abs = complex_fft.imag.abs()               # [n_frames, 257]

    mel_banks, _ = get_mel_banks(128, padded_window_size, 16000.0, 20.0, 0.0, 100.0, -500.0, 1.0)
    mel_banks = mel_banks.to(device=device, dtype=dtype)
    mel_banks = torch.nn.functional.pad(mel_banks, (0, 1), mode="constant", value=0)

    real_mel = torch.max(torch.mm(real_abs, mel_banks.T), _get_epsilon(device, dtype)).log()
    imag_mel = torch.max(torch.mm(imag_abs, mel_banks.T), _get_epsilon(device, dtype)).log()

    return real_mel, imag_mel


def _pad_or_truncate(x, target_length, dim=1):
    """Pad or truncate tensor along given dimension to target_length."""
    n = x.size(dim)
    p = target_length - n
    if p > 0:
        pad_sizes = [0] * (2 * x.ndim)
        pad_sizes[-(2 * dim + 1)] = p
        x = torch.nn.functional.pad(x, pad_sizes, mode="constant", value=0)
    elif p < 0:
        x = x.narrow(dim, 0, target_length)
    return x


def waveform2melspec(waveform, sample_rate, num_mel_bins, target_length,
                     return_phase=False, return_hilbert=False):
    # Based on https://github.com/YuanGongND/ast/blob/d7d8b4b8e06cdaeb6c843cdb38794c1c7692234c/src/dataloader.py#L102
    waveform -= waveform.mean()
    phase_out = None

    if USE_PHASE_EXTRACTION:
        fbank, phase = _fbank_with_phase(waveform)
        # fbank: [n_frames, 128], phase: [n_frames, 257]
        fbank = fbank.transpose(0, 1)   # [128, n_frames]
        phase = phase.transpose(0, 1)   # [257, n_frames]
        # Pad/truncate phase to same temporal length as mel
        phase = _pad_or_truncate(phase, target_length, dim=1)  # [257, target_length]
        phase_out = phase
    else:
        fbank = torchaudio.compliance.kaldi.fbank(
            waveform,
            htk_compat=True,
            sample_frequency=sample_rate,
            use_energy=False,
            window_type="hanning",
            num_mel_bins=num_mel_bins,
            dither=0.0,
            frame_length=25,
            frame_shift=DEFAULT_AUDIO_FRAME_SHIFT_MS,
        )
        fbank = fbank.transpose(0, 1)

    # Pad to target_length
    n_frames = fbank.size(1)
    p = target_length - n_frames
    if abs(p) / max(n_frames, 1) > 0.2:
        logging.warning(
            "Large gap between audio n_frames(%d) and "
            "target_length (%d). Is the audio_target_length "
            "setting correct?",
            n_frames,
            target_length,
        )
    if p > 0:
        fbank = torch.nn.functional.pad(fbank, (0, p), mode="constant", value=0)
    elif p < 0:
        fbank = fbank[:, 0:target_length]

    fbank = fbank.unsqueeze(0)  # [1, 128, target_length]

    if return_hilbert:
        # Hilbert analytic signal imag of mel: [1, 128, target_length]
        # Same shape as mel — can be fed directly through ImageBind audio encoder
        hilbert_imag = _hilbert_analytic_imag(fbank.squeeze(0))  # [128, target_length]
        hilbert_imag = hilbert_imag.unsqueeze(0)  # [1, 128, target_length]
        return fbank, hilbert_imag

    if return_phase and phase_out is not None:
        phase_out = phase_out.unsqueeze(0)  # [1, 257, target_length]
        return fbank, phase_out
    return fbank


def _hilbert_analytic_imag(mel):
    """
    Compute imaginary part of the Hilbert analytic signal along temporal axis.
    Input:  mel [F, T] real (log-mel spectrogram, before unsqueeze)
    Output: hilbert_imag [F, T] real

    The analytic signal z = mel + i*H(mel) has structured phase: the envelope
    |z| tracks energy modulations and angle(z) tracks instantaneous frequency.
    Unlike raw STFT phase, this phase is smooth and event-correlated.
    """
    N = mel.size(-1)
    X = torch.fft.fft(mel, dim=-1)  # [F, T] complex

    # Hilbert multiplier: h[0]=1, h[1..N/2-1]=2, h[N/2]=1, rest=0
    h = torch.zeros(N, device=mel.device, dtype=mel.dtype)
    if N % 2 == 0:
        h[0] = 1.0
        h[1:N // 2] = 2.0
        h[N // 2] = 1.0
    else:
        h[0] = 1.0
        h[1:(N + 1) // 2] = 2.0

    analytic = torch.fft.ifft(X * h.unsqueeze(0), dim=-1)  # [F, T] complex
    return analytic.imag.float()  # [F, T] real


def waveform2melspec_complex(waveform, sample_rate, num_mel_bins, target_length):
    """
    Extract real and imaginary mel spectrograms from audio.
    Returns (real_mel, imag_mel), each [1, 128, target_length].
    """
    waveform -= waveform.mean()
    real_mel, imag_mel = _fbank_complex(waveform)
    # real_mel, imag_mel: [n_frames, 128]
    real_mel = real_mel.transpose(0, 1)  # [128, n_frames]
    imag_mel = imag_mel.transpose(0, 1)  # [128, n_frames]

    # Pad/truncate to target_length
    n_frames = real_mel.size(1)
    p = target_length - n_frames
    if p > 0:
        real_mel = torch.nn.functional.pad(real_mel, (0, p), mode="constant", value=0)
        imag_mel = torch.nn.functional.pad(imag_mel, (0, p), mode="constant", value=0)
    elif p < 0:
        real_mel = real_mel[:, 0:target_length]
        imag_mel = imag_mel[:, 0:target_length]

    return real_mel.unsqueeze(0), imag_mel.unsqueeze(0)  # [1, 128, target_length] each


# Normalization stats for complex mel spectrograms.
# Run compute_mel_stats.py to get these values for your dataset.
# These defaults use magnitude mel stats as fallback — update after computing!
COMPLEX_REAL_MEL_MEAN = -2.628  # Computed via compute_mel_stats.py on full dataset
COMPLEX_REAL_MEL_STD = 1.916    # Computed via compute_mel_stats.py on full dataset
COMPLEX_IMAG_MEL_MEAN = -2.629  # Computed via compute_mel_stats.py on full dataset
COMPLEX_IMAG_MEL_STD = 1.916    # Computed via compute_mel_stats.py on full dataset


def load_and_transform_audio_data_complex(
    audio_paths,
    device=None,
    num_mel_bins=128,
    target_length=204,
    sample_rate=16000,
    real_mean=COMPLEX_REAL_MEL_MEAN,
    real_std=COMPLEX_REAL_MEL_STD,
    imag_mean=COMPLEX_IMAG_MEL_MEAN,
    imag_std=COMPLEX_IMAG_MEL_STD,
):
    """
    Load audio and return real and imaginary mel spectrograms.
    Instead of magnitude² → mel, uses real² → mel and imag² → mel from FFT.

    Returns:
        real_audio: [B, 10, 1, 128, 204] — mel of cos (real) FFT component
        imag_audio: [B, 10, 1, 128, 204] — mel of sin (imaginary) FFT component
    """
    if audio_paths is None:
        return None, None

    audio_duration = 10
    real_outputs = []
    imag_outputs = []

    for audio_path in audio_paths:
        audio_data, sr = sf.read(audio_path, dtype='float32')
        waveform = torch.from_numpy(audio_data)
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        else:
            waveform = waveform.T
        if sample_rate != sr:
            waveform = torchaudio.functional.resample(
                waveform, orig_freq=sr, new_freq=sample_rate
            )

        tmp = sample_rate * audio_duration
        if waveform.size(1) >= tmp:
            waveform = waveform[:, :tmp]
        elif waveform.size(1) < tmp:
            new_waveform = torch.zeros([2, tmp])
            new_waveform[:, :waveform.size(1)] = waveform
            waveform = new_waveform

        all_real = []
        all_imag = []
        for t in range(audio_duration):
            waveform_clip = waveform[:, t*sample_rate:(t+1)*sample_rate]
            waveform_clip = waveform_clip.repeat(1, 2)  # repeat to 2 seconds
            real_mel, imag_mel = waveform2melspec_complex(
                waveform_clip, sample_rate, num_mel_bins, target_length
            )
            all_real.append(real_mel)
            all_imag.append(imag_mel)

        normalize_real = transforms.Normalize(mean=real_mean, std=real_std)
        normalize_imag = transforms.Normalize(mean=imag_mean, std=imag_std)
        all_real = torch.stack([normalize_real(r) for r in all_real], dim=0)  # [10, 1, 128, 204]
        all_imag = torch.stack([normalize_imag(i) for i in all_imag], dim=0)  # [10, 1, 128, 204]
        real_outputs.append(all_real)
        imag_outputs.append(all_imag)

    real_out = torch.stack(real_outputs, dim=0)  # [B, 10, 1, 128, 204]
    imag_out = torch.stack(imag_outputs, dim=0)  # [B, 10, 1, 128, 204]
    return real_out, imag_out


def get_clip_timepoints(clip_sampler, duration):
    # Read out all clips in this video
    all_clips_timepoints = []
    is_last_clip = False
    end = 0.0
    while not is_last_clip:
        start, end, _, _, is_last_clip = clip_sampler(end, duration, annotation=None)
        all_clips_timepoints.append((start, end))
    return all_clips_timepoints


def load_and_transform_vision_data(image_paths, device=None):
    if image_paths is None:
        return None

    image_outputs = []

    data_transform = transforms.Compose(
        [
            transforms.Resize(
                224, interpolation=transforms.InterpolationMode.BICUBIC
            ),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.48145466, 0.4578275, 0.40821073),
                std=(0.26862954, 0.26130258, 0.27577711),
            ),
        ]
    )

    for image_path in image_paths:
        with open(image_path, "rb") as fopen:
            image = Image.open(fopen).convert("RGB")

        # image = data_transform(image).to(device)
        image = data_transform(image)
        image_outputs.append(image)
    return torch.stack(image_outputs, dim=0)


def _rgb_to_hsv(image, eps=1e-6):
    """Convert RGB tensor [*, 3, H, W] (values 0-1) to HSV. H in [0, 2pi], S/V in [0, 1]."""
    import math as _math
    maxc, _ = image.max(-3)
    maxc_mask = image == maxc.unsqueeze(-3)
    _, max_indices = ((maxc_mask.cumsum(-3) == 1) & maxc_mask).max(-3)
    minc = image.min(-3)[0]
    v = maxc
    deltac = maxc - minc
    s = deltac / (maxc + eps)
    deltac = torch.where(deltac == 0, torch.ones_like(deltac), deltac)
    maxc_tmp = maxc.unsqueeze(-3) - image
    rc = maxc_tmp[..., 0, :, :]
    gc = maxc_tmp[..., 1, :, :]
    bc = maxc_tmp[..., 2, :, :]
    h = torch.stack([bc - gc, 2.0 * deltac + rc - bc, 4.0 * deltac + gc - rc], dim=-3)
    h = torch.gather(h, dim=-3, index=max_indices[..., None, :, :])
    h = h.squeeze(-3)
    h = h / deltac
    h = (h / 6.0) % 1.0
    h = 2 * _math.pi * h
    return torch.stack([h, s, v], dim=-3)


def _hsv_to_complex_imag(hsv):
    """Extract imaginary part from iHSV complex conversion. Input: [*, 3, H, W] HSV tensor.
    Returns: [*, 3, H, W] real-valued tensor (the imaginary channels: V, S*sin(H), S)."""
    h = hsv[..., 0, :, :]
    s = hsv[..., 1, :, :]
    v = hsv[..., 2, :, :]
    imag_1 = v
    imag_2 = s * torch.sin(h)
    imag_3 = s
    return torch.stack([imag_1, imag_2, imag_3], dim=-3)


def _ihsv_to_unit_range(x):
    """Map _hsv_to_rgb output (which can fall outside [0,1]) into [0,1] using
    cfg.IHSV.RESCALE_MODE. Reversible: with mode='clamp' this is the original
    behavior bit-for-bit. With mode='minmax' it preserves all signal at the
    cost of a per-image (per-channel) contrast rescale."""
    try:
        from config import cfg as _cfg
        mode = getattr(_cfg.IHSV, 'RESCALE_MODE', 'clamp')
    except Exception:
        mode = 'clamp'
    if mode == 'minmax':
        # x: [..., 3, H, W] — rescale per (channel, image) over the spatial dims
        lo = x.amin(dim=(-2, -1), keepdim=True)
        hi = x.amax(dim=(-2, -1), keepdim=True)
        return (x - lo) / (hi - lo + 1e-8)
    return x.clamp(0, 1)


def load_and_transform_vision_data_imag(image_paths, device=None):
    """Load images, convert RGB -> HSV -> Complex, return imaginary part normalized for ImageBind."""
    if image_paths is None:
        return None

    image_outputs = []

    spatial_transform = transforms.Compose(
        [
            transforms.Resize(
                224, interpolation=transforms.InterpolationMode.BICUBIC
            ),
            transforms.CenterCrop(224),
            transforms.ToTensor(),  # [3, 224, 224] in [0, 1]
        ]
    )

    for image_path in image_paths:
        with open(image_path, "rb") as fopen:
            image = Image.open(fopen).convert("RGB")

        img_tensor = spatial_transform(image)  # [3, 224, 224], values in [0, 1]
        hsv = _rgb_to_hsv(img_tensor)  # [3, 224, 224], H in [0, 2pi], S/V in [0, 1]
        imag_hsv = _hsv_to_complex_imag(hsv)  # [3, 224, 224]
        imag = _ihsv_to_unit_range(_hsv_to_rgb(imag_hsv))  # cfg.IHSV.RESCALE_MODE: 'clamp' | 'minmax'
        # Normalize with ImageBind vision stats so the frozen encoder sees familiar value ranges
        imag = transforms.functional.normalize(
            imag,
            mean=(0.48145466, 0.4578275, 0.40821073),
            std=(0.26862954, 0.26130258, 0.27577711),
        )
        image_outputs.append(imag)
    return torch.stack(image_outputs, dim=0)


# ---------------------------------------------------------------------------
# Complex representation: HSV → Complex → hsv_to_rgb on real & imag parts
# From FCCNs (Fully Complex-valued Convolutional Networks)
# ---------------------------------------------------------------------------

def _hsv_to_rgb(image):
    """Convert HSV tensor [*, 3, H, W] to RGB. H in [0, 2pi], S/V in [0, 1].
    Adapted from FCCNs/utils.py."""
    h = image[..., 0, :, :] / (2 * math.pi)
    s = image[..., 1, :, :]
    v = image[..., 2, :, :]

    hi = torch.floor(h * 6) % 6
    f = ((h * 6) % 6) - hi
    one = torch.tensor(1.0, device=image.device, dtype=image.dtype)
    p = v * (one - s)
    q = v * (one - f * s)
    t = v * (one - (one - f) * s)

    hi = hi.long()
    indices = torch.stack([hi, hi + 6, hi + 12], dim=-3)
    out = torch.stack((v, q, p, p, t, v, t, v, v, q, p, p, p, p, t, v, v, q), dim=-3)
    out = torch.gather(out, -3, indices)
    return out


def _hsv_to_complex_real(hsv):
    """Extract real part from complex HSV conversion. Input: [*, 3, H, W] HSV tensor.
    Returns: [*, 3, H, W] real-valued tensor (the real channels: S*H, S*cos(H), V).
    From FCCNs ToComplex class."""
    h = hsv[..., 0, :, :]
    s = hsv[..., 1, :, :]
    v = hsv[..., 2, :, :]
    real_1 = s * h          # saturation × hue angle
    real_2 = s * torch.cos(h)  # saturation × cos(hue)
    real_3 = v              # brightness
    return torch.stack([real_1, real_2, real_3], dim=-3)


def load_and_transform_vision_data_complex(image_paths, device=None):
    """Load images, convert RGB → HSV → Complex → hsv_to_rgb on real & imag parts.
    Returns (real_rgb, imag_rgb) both [N, 3, 224, 224] normalized for ImageBind."""
    if image_paths is None:
        return None, None

    real_outputs = []
    imag_outputs = []

    spatial_transform = transforms.Compose(
        [
            transforms.Resize(
                224, interpolation=transforms.InterpolationMode.BICUBIC
            ),
            transforms.CenterCrop(224),
            transforms.ToTensor(),  # [3, 224, 224] in [0, 1]
        ]
    )
    imagebind_normalize = transforms.Normalize(
        mean=(0.48145466, 0.4578275, 0.40821073),
        std=(0.26862954, 0.26130258, 0.27577711),
    )

    for image_path in image_paths:
        with open(image_path, "rb") as fopen:
            image = Image.open(fopen).convert("RGB")

        img_tensor = spatial_transform(image)  # [3, 224, 224], values in [0, 1]
        hsv = _rgb_to_hsv(img_tensor)  # [3, 224, 224], H in [0, 2pi], S/V in [0, 1]

        # Complex real part: [S*H, S*cos(H), V] — treated as "HSV" → convert to RGB
        real_hsv = _hsv_to_complex_real(hsv)   # [3, 224, 224]
        real_rgb = _hsv_to_rgb(real_hsv).clamp(0, 1)  # [3, 224, 224]
        real_rgb = imagebind_normalize(real_rgb)

        # Complex imag part: [V, S*sin(H), S] — treated as "HSV" → convert to RGB
        imag_hsv = _hsv_to_complex_imag(hsv)   # [3, 224, 224]
        imag_rgb = _ihsv_to_unit_range(_hsv_to_rgb(imag_hsv))  # cfg.IHSV.RESCALE_MODE: 'clamp' | 'minmax'
        imag_rgb = imagebind_normalize(imag_rgb)

        real_outputs.append(real_rgb)
        imag_outputs.append(imag_rgb)

    return torch.stack(real_outputs, dim=0), torch.stack(imag_outputs, dim=0)


# ---------------------------------------------------------------------------
# Luma (BT.709) imag stream.
# real = RGB; imag = luma replicated to 3 channels so the frozen ImageBind
# vision encoder sees a familiar shape.
# ---------------------------------------------------------------------------

_BT709_LUMA_WEIGHTS = torch.tensor([0.2126, 0.7152, 0.0722]).view(3, 1, 1)


def load_and_transform_vision_data_luma(image_paths, device=None):
    """Load images, return (real_rgb, luma3_rgb) both [N, 3, 224, 224]
    ImageBind-normalized. Imag stream is BT.709 luma broadcast to 3 channels."""
    if image_paths is None:
        return None, None

    real_outputs = []
    imag_outputs = []

    spatial_transform = transforms.Compose(
        [
            transforms.Resize(
                224, interpolation=transforms.InterpolationMode.BICUBIC
            ),
            transforms.CenterCrop(224),
            transforms.ToTensor(),  # [3, 224, 224] in [0, 1]
        ]
    )
    imagebind_normalize = transforms.Normalize(
        mean=(0.48145466, 0.4578275, 0.40821073),
        std=(0.26862954, 0.26130258, 0.27577711),
    )

    for image_path in image_paths:
        with open(image_path, "rb") as fopen:
            image = Image.open(fopen).convert("RGB")

        rgb = spatial_transform(image)  # [3, 224, 224] in [0, 1]
        luma = (rgb * _BT709_LUMA_WEIGHTS).sum(dim=0, keepdim=True)  # [1, 224, 224]
        luma3 = luma.expand(3, -1, -1).contiguous()                   # [3, 224, 224]

        real_outputs.append(imagebind_normalize(rgb))
        imag_outputs.append(imagebind_normalize(luma3))

    return torch.stack(real_outputs, dim=0), torch.stack(imag_outputs, dim=0)


def _phase_only_ifft_clahe(img_u8_gray, clip=3.0, tile=(8, 8)):
    """Phase-only reconstruction of a grayscale image + CLAHE.

    img_u8_gray: [H, W] uint8 numpy.
    Returns:     [H, W] uint8 numpy, contrast-enhanced phase-IFFT.
    """
    import cv2
    F = np.fft.fft2(img_u8_gray.astype(np.float32))
    recon = np.fft.ifft2(np.exp(1j * np.angle(F))).real
    lo, hi = recon.min(), recon.max()
    if hi - lo < 1e-12:
        raw_u8 = np.zeros_like(recon, dtype=np.uint8)
    else:
        raw_u8 = np.clip((recon - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)
    return cv2.createCLAHE(clipLimit=clip, tileGridSize=tile).apply(raw_u8)


def load_and_transform_vision_data_phase_ifft(image_paths, device=None):
    """Load images. Returns (real_rgb, imag_phase_ifft_3ch), both ImageBind-normalized.

    real_rgb:           standard RGB pipeline (resize 224 → center crop → ToTensor → ImageBind mean/std).
    imag_phase_ifft_3ch: grayscale phase-only IFFT + CLAHE, replicated to 3 channels,
                        then run through the same ImageBind mean/std so it matches the
                        distribution the frozen visual encoder expects.

    Shapes: both [N, 3, 224, 224] float tensors.
    """
    import cv2

    if image_paths is None:
        return None, None

    resize_crop = transforms.Compose([
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
    ])
    to_tensor = transforms.ToTensor()
    imagebind_normalize = transforms.Normalize(
        mean=(0.48145466, 0.4578275, 0.40821073),
        std=(0.26862954, 0.26130258, 0.27577711),
    )

    real_outputs, imag_outputs = [], []

    for image_path in image_paths:
        with open(image_path, "rb") as fopen:
            image = Image.open(fopen).convert("RGB")
        pil_224 = resize_crop(image)  # PIL, 224x224

        # Real stream: standard ImageBind pipeline.
        real_tensor = imagebind_normalize(to_tensor(pil_224))  # [3, 224, 224]

        # Imag stream: grayscale phase-IFFT + CLAHE, broadcast to 3 channels.
        gray_u8 = np.array(pil_224.convert("L"), dtype=np.uint8)  # [224, 224]
        phase_u8 = _phase_only_ifft_clahe(gray_u8)                # [224, 224]
        phase_float = phase_u8.astype(np.float32) / 255.0         # [224, 224] in [0, 1]
        phase_3ch = np.stack([phase_float] * 3, axis=0)           # [3, 224, 224]
        imag_tensor = imagebind_normalize(torch.from_numpy(phase_3ch))

        real_outputs.append(real_tensor)
        imag_outputs.append(imag_tensor)

    return torch.stack(real_outputs, dim=0), torch.stack(imag_outputs, dim=0)


def load_and_transform_depth_data(depth_paths, device=None):
    """Load depth maps (uint16 PNGs, 224x224) for ImageBind's depth encoder.

    ImageBind was trained on SUN RGB-D with depth converted to disparity
    (following Omnivore: Girdhar et al., 2022). We convert Depth-Anything-V2's
    relative depth to relative disparity to match the expected distribution.

    Pipeline: uint16 [0,65535] -> float depth [0,1] -> disparity 1/(d+eps) -> normalize [0,1]

    Args:
        depth_paths: list of paths to depth PNG files (uint16, 224x224)

    Returns:
        tensor: [N, 1, 224, 224] float32, disparity in [0, 1]
    """
    if depth_paths is None:
        return None

    import cv2

    depth_outputs = []
    for depth_path in depth_paths:
        # Read as uint16 single-channel
        depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        if depth is None:
            raise ValueError(f"Could not read depth map: {depth_path}")
        # Convert to float [0, 1] relative depth
        depth_float = depth.astype(np.float32) / 65535.0
        # Ensure 224x224
        if depth_float.shape[0] != 224 or depth_float.shape[1] != 224:
            depth_float = cv2.resize(depth_float, (224, 224), interpolation=cv2.INTER_LINEAR)
        # Convert to disparity (ImageBind was trained on disparity, not depth)
        # Higher disparity = closer objects (matches SUN RGB-D disparity distribution)
        disparity = 1.0 / (depth_float + 1e-3)
        # Normalize disparity to [0, 1] per frame
        d_min, d_max = disparity.min(), disparity.max()
        if d_max - d_min > 1e-6:
            disparity = (disparity - d_min) / (d_max - d_min)
        else:
            disparity = np.zeros_like(disparity)
        # To tensor: [1, 224, 224]
        tensor = torch.from_numpy(disparity).unsqueeze(0)
        depth_outputs.append(tensor)

    return torch.stack(depth_outputs, dim=0)  # [N, 1, 224, 224]


def load_and_transform_text(text, device=None):
    if text is None:
        return None
    tokenizer = SimpleTokenizer(bpe_path=BPE_PATH)
    # tokens = [tokenizer(t).unsqueeze(0).to(device) for t in text]
    tokens = [tokenizer(t).unsqueeze(0) for t in text]
    tokens = torch.cat(tokens, dim=0)
    # pdb.set_trace()
    return tokens

def load_and_transform_audio_data(
    audio_paths,
    device=None,
    num_mel_bins=128,
    target_length=204,
    sample_rate=16000,
    clip_duration=2,
    clips_per_video=1,
    mean=-4.268,
    std=9.138,
    return_phase=False,
    return_hilbert=False,
):
    if audio_paths is None:
        return None

    audio_outputs = []
    phase_outputs = []
    clip_sampler = None
    audio_duration = 10 # 10 seconds
    for audio_path in audio_paths:
        # Use soundfile for audio loading (compatible with PyTorch nightly)
        audio_data, sr = sf.read(audio_path, dtype='float32')
        waveform = torch.from_numpy(audio_data)
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        else:
            waveform = waveform.T
        if sample_rate != sr:
            waveform = torchaudio.functional.resample(
                waveform, orig_freq=sr, new_freq=sample_rate
            )

        tmp = sample_rate * audio_duration
        if waveform.size(1) >= tmp:
            waveform = waveform[:, :tmp]
        elif  waveform.size(1) < tmp:
            new_waveform = torch.zeros([2, tmp])
            new_waveform[:, :waveform.size(1)] = waveform
            waveform = new_waveform

        all_clips = []
        all_phases = []
        for t in range(audio_duration):
            waveform_clip = waveform[:, t*sample_rate:(t+1)*sample_rate]
            waveform_clip = waveform_clip.repeat(1, 2) #! repeat to 2 seconds

            if return_hilbert:
                waveform_melspec, hilbert_imag = waveform2melspec(
                    waveform_clip, sample_rate, num_mel_bins, target_length,
                    return_hilbert=True,
                )
                all_phases.append(hilbert_imag)  # [1, 128, 204] — same shape as mel
            elif return_phase and USE_PHASE_EXTRACTION:
                waveform_melspec, phase_spec = waveform2melspec(
                    waveform_clip, sample_rate, num_mel_bins, target_length,
                    return_phase=True,
                )
                all_phases.append(phase_spec)  # [1, 257, 204]
            else:
                waveform_melspec = waveform2melspec(
                    waveform_clip, sample_rate, num_mel_bins, target_length
                )
            all_clips.append(waveform_melspec)

        normalize = transforms.Normalize(mean=mean, std=std)
        all_clips = [normalize(ac) for ac in all_clips]
        all_clips = torch.stack(all_clips, dim=0)   # [10, 1, 128, 204]
        audio_outputs.append(all_clips)

        if (return_phase or return_hilbert) and all_phases:
            if return_hilbert:
                # Normalize Hilbert imag with same mel stats — it's derived from mel
                # and has similar scale. ImageBind expects normalized input.
                all_phases = [normalize(hp) for hp in all_phases]
            all_phases = torch.stack(all_phases, dim=0)
            # hilbert: [10, 1, 128, 204], phase: [10, 1, 257, 204]
            phase_outputs.append(all_phases)

    audio_out = torch.stack(audio_outputs, dim=0)    # [B, 10, 1, 128, 204]

    if (return_phase or return_hilbert) and phase_outputs:
        phase_out = torch.stack(phase_outputs, dim=0)
        # hilbert: [B, 10, 1, 128, 204], phase: [B, 10, 1, 257, 204]
        return audio_out, phase_out

    return audio_out


def ori_load_and_transform_audio_data(
    audio_paths,
    device,
    num_mel_bins=128,
    target_length=204,
    sample_rate=16000,
    clip_duration=2,
    clips_per_video=3,
    mean=-4.268,
    std=9.138,
):
    if audio_paths is None:
        return None

    audio_outputs = []
    clip_sampler = ConstantClipsPerVideoSampler(
        clip_duration=clip_duration, clips_per_video=clips_per_video
    )

    for audio_path in audio_paths:
        # Use soundfile for audio loading (compatible with PyTorch nightly)
        audio_data, sr = sf.read(audio_path, dtype='float32')
        waveform = torch.from_numpy(audio_data)
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        else:
            waveform = waveform.T
        if sample_rate != sr:
            waveform = torchaudio.functional.resample(
                waveform, orig_freq=sr, new_freq=sample_rate
            )
        all_clips_timepoints = get_clip_timepoints(
            clip_sampler, waveform.size(1) / sample_rate
        )
        all_clips = []
        for clip_timepoints in all_clips_timepoints:
            waveform_clip = waveform[
                :,
                int(clip_timepoints[0] * sample_rate) : int(
                    clip_timepoints[1] * sample_rate
                ),
            ]
            waveform_melspec = waveform2melspec(
                waveform_clip, sample_rate, num_mel_bins, target_length
            )
            all_clips.append(waveform_melspec)

        normalize = transforms.Normalize(mean=mean, std=std)
        all_clips = [normalize(ac).to(device) for ac in all_clips]

        all_clips = torch.stack(all_clips, dim=0)
        audio_outputs.append(all_clips)
    pdb.set_trace()

    return torch.stack(audio_outputs, dim=0)


def crop_boxes(boxes, x_offset, y_offset):
    """
    Perform crop on the bounding boxes given the offsets.
    Args:
        boxes (ndarray or None): bounding boxes to perform crop. The dimension
            is `num boxes` x 4.
        x_offset (int): cropping offset in the x axis.
        y_offset (int): cropping offset in the y axis.
    Returns:
        cropped_boxes (ndarray or None): the cropped boxes with dimension of
            `num boxes` x 4.
    """
    cropped_boxes = boxes.copy()
    cropped_boxes[:, [0, 2]] = boxes[:, [0, 2]] - x_offset
    cropped_boxes[:, [1, 3]] = boxes[:, [1, 3]] - y_offset

    return cropped_boxes


def uniform_crop(images, size, spatial_idx, boxes=None, scale_size=None):
    """
    Perform uniform spatial sampling on the images and corresponding boxes.
    Args:
        images (tensor): images to perform uniform crop. The dimension is
            `num frames` x `channel` x `height` x `width`.
        size (int): size of height and weight to crop the images.
        spatial_idx (int): 0, 1, or 2 for left, center, and right crop if width
            is larger than height. Or 0, 1, or 2 for top, center, and bottom
            crop if height is larger than width.
        boxes (ndarray or None): optional. Corresponding boxes to images.
            Dimension is `num boxes` x 4.
        scale_size (int): optinal. If not None, resize the images to scale_size before
            performing any crop.
    Returns:
        cropped (tensor): images with dimension of
            `num frames` x `channel` x `size` x `size`.
        cropped_boxes (ndarray or None): the cropped boxes with dimension of
            `num boxes` x 4.
    """
    assert spatial_idx in [0, 1, 2]
    ndim = len(images.shape)
    if ndim == 3:
        images = images.unsqueeze(0)
    height = images.shape[2]
    width = images.shape[3]

    if scale_size is not None:
        if width <= height:
            width, height = scale_size, int(height / width * scale_size)
        else:
            width, height = int(width / height * scale_size), scale_size
        images = torch.nn.functional.interpolate(
            images,
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )

    y_offset = int(math.ceil((height - size) / 2))
    x_offset = int(math.ceil((width - size) / 2))

    if height > width:
        if spatial_idx == 0:
            y_offset = 0
        elif spatial_idx == 2:
            y_offset = height - size
    else:
        if spatial_idx == 0:
            x_offset = 0
        elif spatial_idx == 2:
            x_offset = width - size
    cropped = images[:, :, y_offset : y_offset + size, x_offset : x_offset + size]
    cropped_boxes = crop_boxes(boxes, x_offset, y_offset) if boxes is not None else None
    if ndim == 3:
        cropped = cropped.squeeze(0)
    return cropped, cropped_boxes


class SpatialCrop(nn.Module):
    """
    Convert the video into 3 smaller clips spatially. Must be used after the
        temporal crops to get spatial crops, and should be used with
        -2 in the spatial crop at the slowfast augmentation stage (so full
        frames are passed in here). Will return a larger list with the
        3x spatial crops as well.
    """

    def __init__(self, crop_size: int = 224, num_crops: int = 3):
        super().__init__()
        self.crop_size = crop_size
        if num_crops == 3:
            self.crops_to_ext = [0, 1, 2]
            self.flipped_crops_to_ext = []
        elif num_crops == 1:
            self.crops_to_ext = [1]
            self.flipped_crops_to_ext = []
        else:
            raise NotImplementedError("Nothing else supported yet")

    def forward(self, videos):
        """
        Args:
            videos: A list of C, T, H, W videos.
        Returns:
            videos: A list with 3x the number of elements. Each video converted
                to C, T, H', W' by spatial cropping.
        """
        assert isinstance(videos, list), "Must be a list of videos after temporal crops"
        assert all([video.ndim == 4 for video in videos]), "Must be (C,T,H,W)"
        res = []
        for video in videos:
            for spatial_idx in self.crops_to_ext:
                res.append(uniform_crop(video, self.crop_size, spatial_idx)[0])
            if not self.flipped_crops_to_ext:
                continue
            flipped_video = transforms.functional.hflip(video)
            for spatial_idx in self.flipped_crops_to_ext:
                res.append(uniform_crop(flipped_video, self.crop_size, spatial_idx)[0])
        return res


def load_and_transform_video_data(
    video_paths,
    device,
    clip_duration=2,
    clips_per_video=5,
    sample_rate=16000,
):
    if video_paths is None:
        return None

    video_outputs = []
    video_transform = transforms.Compose(
        [
            pv_transforms.ShortSideScale(224),
            NormalizeVideo(
                mean=(0.48145466, 0.4578275, 0.40821073),
                std=(0.26862954, 0.26130258, 0.27577711),
            ),
        ]
    )

    clip_sampler = ConstantClipsPerVideoSampler(
        clip_duration=clip_duration, clips_per_video=clips_per_video
    )
    frame_sampler = pv_transforms.UniformTemporalSubsample(num_samples=clip_duration)

    for video_path in video_paths:
        video = EncodedVideo.from_path(
            video_path,
            decoder="decord",
            decode_audio=False,
            **{"sample_rate": sample_rate},
        )

        all_clips_timepoints = get_clip_timepoints(clip_sampler, video.duration)

        all_video = []
        for clip_timepoints in all_clips_timepoints:
            # Read the clip, get frames
            clip = video.get_clip(clip_timepoints[0], clip_timepoints[1])
            if clip is None:
                raise ValueError("No clip found")
            video_clip = frame_sampler(clip["video"])
            video_clip = video_clip / 255.0  # since this is float, need 0-1

            all_video.append(video_clip)

        all_video = [video_transform(clip) for clip in all_video]
        all_video = SpatialCrop(224, num_crops=3)(all_video)

        all_video = torch.stack(all_video, dim=0)
        video_outputs.append(all_video)

    return torch.stack(video_outputs, dim=0).to(device)
