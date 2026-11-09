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
                     return_phase=False):
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

    if return_phase and phase_out is not None:
        phase_out = phase_out.unsqueeze(0)  # [1, 257, target_length]
        return fbank, phase_out
    return fbank


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

            if return_phase and USE_PHASE_EXTRACTION:
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

        if (return_phase) and all_phases:
            all_phases = torch.stack(all_phases, dim=0)
            # hilbert: [10, 1, 128, 204], phase: [10, 1, 257, 204]
            phase_outputs.append(all_phases)

    audio_out = torch.stack(audio_outputs, dim=0)    # [B, 10, 1, 128, 204]

    if (return_phase) and phase_outputs:
        phase_out = torch.stack(phase_outputs, dim=0)
        # hilbert: [B, 10, 1, 128, 204], phase: [B, 10, 1, 257, 204]
        return audio_out, phase_out

    return audio_out


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


def uniform_crop(images, size, spatial_idx, boxes=None):
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
    cropped_boxes = None
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
