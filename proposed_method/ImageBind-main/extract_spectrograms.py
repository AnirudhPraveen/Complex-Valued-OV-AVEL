"""
Extract magnitude mel spectrograms and phase spectrograms from OV-AVEL dataset.

Outputs:
    <output_dir>/magnitude/  — [128, 204] log-mel magnitude spectrograms (.npy)
    <output_dir>/phase/      — [257, 204] raw phase angle spectrograms (.npy)
    <output_dir>/phase_if/   — [257, 204] instantaneous frequency spectrograms (.npy)

Each file is named: <split>_<category>_<vid>_t<segment>.npy
(10 segments per video, 1 second each)

Usage:
    python extract_spectrograms.py --output_dir ./spectrograms
    python extract_spectrograms.py --output_dir ./spectrograms --dataset ave_ov
"""

import os
import argparse
import numpy as np
import torch
import torchaudio
import soundfile as sf
from tqdm import tqdm
import pandas as pd

# Reuse the existing extraction functions
import sys
sys.path.insert(0, os.path.dirname(__file__))
import config as cfg_paths
from imagebind.data import waveform2melspec, _fbank_with_phase, _pad_or_truncate, USE_PHASE_EXTRACTION


def extract_from_audio(audio_path, sample_rate=16000, num_mel_bins=128, target_length=204):
    """
    Extract magnitude mel and phase spectrograms from a 10-second audio file.

    Returns:
        magnitudes: list of 10 numpy arrays, each [128, 204]
        phases: list of 10 numpy arrays, each [257, 204]
        phases_if: list of 10 numpy arrays, each [257, 204] (instantaneous frequency)
    """
    audio_data, sr = sf.read(audio_path, dtype='float32')
    if audio_data.ndim == 1:
        audio_data = np.stack([audio_data, audio_data], axis=0)
    else:
        audio_data = audio_data.T

    waveform = torch.from_numpy(audio_data).float()

    # Resample if needed
    if sr != sample_rate:
        waveform = torchaudio.functional.resample(waveform, sr, sample_rate)

    # Pad to 10 seconds if shorter
    total_samples = sample_rate * 10
    if waveform.size(1) < total_samples:
        padded = torch.zeros(2, total_samples)
        padded[:, :waveform.size(1)] = waveform
        waveform = padded

    magnitudes = []
    phases = []
    phases_if = []

    for t in range(10):
        waveform_clip = waveform[:, t * sample_rate:(t + 1) * sample_rate]
        waveform_clip = waveform_clip.repeat(1, 2)  # repeat to 2 seconds (ImageBind convention)

        # Extract magnitude mel + phase
        waveform_clip = waveform_clip - waveform_clip.mean()
        fbank, phase = _fbank_with_phase(waveform_clip)
        # fbank: [n_frames, 128], phase: [n_frames, 257]

        fbank = fbank.transpose(0, 1)  # [128, n_frames]
        phase = phase.transpose(0, 1)  # [257, n_frames]

        # Pad/truncate to target_length
        n_frames = fbank.size(1)
        p = target_length - n_frames
        if p > 0:
            fbank = torch.nn.functional.pad(fbank, (0, p), mode="constant", value=0)
            phase = torch.nn.functional.pad(phase, (0, p), mode="constant", value=0)
        elif p < 0:
            fbank = fbank[:, :target_length]
            phase = phase[:, :target_length]

        # Compute IF (instantaneous frequency = temporal derivative of phase)
        phase_if = torch.zeros_like(phase)
        phase_if[:, 1:] = phase[:, 1:] - phase[:, :-1]
        phase_if = (phase_if + torch.pi) % (2 * torch.pi) - torch.pi

        magnitudes.append(fbank.numpy())       # [128, 204]
        phases.append(phase.numpy())           # [257, 204]
        phases_if.append(phase_if.numpy())     # [257, 204]

    return magnitudes, phases, phases_if


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, default='./spectrograms')
    parser.add_argument('--dataset', type=str, default='ovavel', choices=['ovavel', 'ave_ov'])
    parser.add_argument('--splits', type=str, nargs='+', default=['train', 'val', 'test'])
    args = parser.parse_args()

    # Dataset paths come from config.py (OVAVEL_DATASET selects the dataset)
    os.environ['OVAVEL_DATASET'] = args.dataset
    import importlib
    importlib.reload(cfg_paths)
    data_base_dir = cfg_paths.cfg.DATA_BASE_DIR
    meta_csv = cfg_paths.cfg.META_CSV_PATH

    # Create output dirs
    for subdir in ['magnitude', 'phase', 'phase_if']:
        os.makedirs(os.path.join(args.output_dir, subdir), exist_ok=True)

    # Read meta CSV
    meta_df = pd.read_csv(meta_csv)
    print(f"Dataset: {args.dataset}, Total videos: {len(meta_df)}")
    print(f"Columns: {list(meta_df.columns)}")
    print(f"Splits requested: {args.splits}")

    total_extracted = 0
    total_failed = 0

    for split in args.splits:
        split_df = meta_df[meta_df.iloc[:, 0] == split]
        print(f"\n=== Split: {split}, Videos: {len(split_df)} ===")

        for idx in tqdm(range(len(split_df)), desc=f"Extracting {split}"):
            row = split_df.iloc[idx]
            category = row.iloc[1]
            vid_name = row.iloc[3]

            audio_path = os.path.join(data_base_dir, split, 'audio', str(category), vid_name + '.wav')

            if not os.path.exists(audio_path):
                total_failed += 1
                continue

            try:
                magnitudes, phases, phases_if = extract_from_audio(audio_path)

                for t in range(10):
                    fname = f"{split}_{category}_{vid_name}_t{t}.npy"
                    np.save(os.path.join(args.output_dir, 'magnitude', fname), magnitudes[t])
                    np.save(os.path.join(args.output_dir, 'phase', fname), phases[t])
                    np.save(os.path.join(args.output_dir, 'phase_if', fname), phases_if[t])

                total_extracted += 1
            except Exception as e:
                print(f"\nFailed: {audio_path}: {e}")
                total_failed += 1

    print(f"\n=== Done ===")
    print(f"Extracted: {total_extracted} videos ({total_extracted * 10} spectrograms)")
    print(f"Failed: {total_failed} videos")
    print(f"\nOutput:")
    print(f"  Magnitude mels: {args.output_dir}/magnitude/  [128, 204] log-mel")
    print(f"  Raw phase:      {args.output_dir}/phase/      [257, 204] angles [-π, π]")
    print(f"  Phase IF:       {args.output_dir}/phase_if/   [257, 204] inst. freq [-π, π]")

    # Print distribution stats from a sample
    mag_files = [f for f in os.listdir(os.path.join(args.output_dir, 'magnitude')) if f.endswith('.npy')]
    if mag_files:
        sample_mags = [np.load(os.path.join(args.output_dir, 'magnitude', f)) for f in mag_files[:100]]
        sample_phases = [np.load(os.path.join(args.output_dir, 'phase_if', f)) for f in mag_files[:100]]
        all_mag = np.concatenate([m.flatten() for m in sample_mags])
        all_phase = np.concatenate([p.flatten() for p in sample_phases])

        print(f"\n=== Distribution Stats (first 100 spectrograms) ===")
        print(f"Magnitude mel:  mean={all_mag.mean():.3f}, std={all_mag.std():.3f}, "
              f"min={all_mag.min():.3f}, max={all_mag.max():.3f}")
        print(f"Phase IF:       mean={all_phase.mean():.3f}, std={all_phase.std():.3f}, "
              f"min={all_phase.min():.3f}, max={all_phase.max():.3f}")
        print(f"\nNote: Magnitude is log-scale, Phase IF is in [-π, π]")
        print(f"A GAN would need to bridge: phase IF distribution → magnitude mel distribution")


if __name__ == '__main__':
    main()
