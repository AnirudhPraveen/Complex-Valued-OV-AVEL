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


# ---------------------------------------------------------------------------
# Toggle: set USE_PHASE_EXTRACTION = True to extract phase alongside mel.
# The mel spectrogram is IDENTICAL either way (verified, diff = 0.0).
# To revert: just set USE_PHASE_EXTRACTION = False (original kaldi path).
# ---------------------------------------------------------------------------
USE_PHASE_EXTRACTION = True

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
        fbank = fbank.transpose(0, 1)   # [128, n_frames]
        phase = phase.transpose(0, 1)   # [257, n_frames]
        phase = _pad_or_truncate(phase, target_length, dim=1)
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

        if return_phase and all_phases:
            all_phases = torch.stack(all_phases, dim=0)  # [10, 1, 257, 204]
            phase_outputs.append(all_phases)

    audio_out = torch.stack(audio_outputs, dim=0)    # [B, 10, 1, 128, 204]

    if return_phase and phase_outputs:
        phase_out = torch.stack(phase_outputs, dim=0)  # [B, 10, 1, 257, 204]
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
