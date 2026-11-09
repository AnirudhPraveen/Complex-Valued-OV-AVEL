"""Build the AVE-OV dataset folder from the original AVE videos.

AVE-OV re-casts AVE (Tian et al., ECCV 2018) into the open-vocabulary setting
(20 seen + 8 unseen classes). This script cuts the first 10 s of every clip listed
in meta_anno_files/ave_ov/ave_ov_dataset_meta.csv and writes

    <out_dir>/<split>/audio/<category>/<video_id>.wav
    <out_dir>/<split>/video/<category>/<video_id>/frame_01.jpg ... frame_10.jpg

(one frame per second), i.e. the same layout as OV-AVEBench.

Usage:
    python tools/prepare_ave_ov.py --ave_videos /path/to/AVE --out_dir ../../AVE_Dataset

`--ave_videos` is the folder of AVE .mp4 files from the official AVE release
(https://github.com/YapengTian/AVE-ECCV18). Requires `moviepy==1.0.3`.
"""
import argparse
import os

import pandas as pd
from moviepy.editor import VideoFileClip

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
DEFAULT_META = os.path.join(REPO_ROOT, "meta_anno_files", "ave_ov", "ave_ov_dataset_meta.csv")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ave_videos", required=True, help="folder with the AVE .mp4 videos")
    ap.add_argument("--out_dir", default=os.path.join(REPO_ROOT, "AVE_Dataset"))
    ap.add_argument("--meta_csv", default=DEFAULT_META)
    args = ap.parse_args()

    meta = pd.read_csv(args.meta_csv)
    done = failed = 0
    for row in meta.itertuples(index=False):
        split, category, vid = row.split, row.cls_name, row.vid_name
        src = os.path.join(args.ave_videos, f"{vid}.mp4")
        audio_dir = os.path.join(args.out_dir, split, "audio", category)
        frame_dir = os.path.join(args.out_dir, split, "video", category, vid)
        wav_path = os.path.join(audio_dir, f"{vid}.wav")
        if os.path.isfile(wav_path) and os.path.isdir(frame_dir) and len(os.listdir(frame_dir)) == 10:
            done += 1
            continue
        if not os.path.isfile(src):
            print(f"[missing] {src}")
            failed += 1
            continue
        os.makedirs(audio_dir, exist_ok=True)
        os.makedirs(frame_dir, exist_ok=True)
        try:
            with VideoFileClip(src) as video:
                clip = video.subclip(0, min(10, video.duration))
                clip.audio.write_audiofile(wav_path, codec="pcm_s16le", verbose=False, logger=None)
                for i in range(int(clip.duration)):
                    clip.save_frame(os.path.join(frame_dir, f"frame_{i + 1:02d}.jpg"), t=i)
            done += 1
        except Exception as exc:  # keep going; report at the end
            print(f"[error] {vid}: {exc}")
            failed += 1
    print(f"AVE-OV prepared: {done} clips OK, {failed} failed -> {args.out_dir}")


if __name__ == "__main__":
    main()
