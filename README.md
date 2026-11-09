# Open-Vocabulary Audio-Visual Event Localization via Complex-Valued Fusion

Anirudh Praveen, Koteswar Rao Jerripothula, Pratik Joshi, Aveen Dayal, Neela Sawant

Official code for our BMVC 2026 paper *Open-Vocabulary Audio-Visual Event Localization via Complex-Valued Fusion*.

> This repository builds on the official code of **OV-AVEL** ([Zhou et al., CVPR 2025](https://arxiv.org/abs/2411.11278), [jasongief/OV-AVEL](https://github.com/jasongief/OV-AVEL)). The OV-AVEBench dataset, the evaluation protocol and the training-free / fine-tuned baselines come from their work, so please also cite it (see [Citation](#citation)).

## Overview

Open-Vocabulary Audio-Visual Event Localization (OV-AVEL) labels every one-second segment of a video with an event class, including classes never seen during training. Existing pipelines embed the visual frame, the audio mel-spectrogram and every candidate class name with a frozen ImageBind. They then compute a visual–text and an audio–text similarity and collapse the two with a fixed rule.

We instead form **complex-valued similarities** and learn their fusion with a **complex-valued CNN**:

| Modality | Real part | Imaginary (companion) part |
|---|---|---|
| Vision | RGB frame | imaginary part of iHSV |
| Audio | mel-spectrogram | instantaneous-frequency phase, translated to the mel domain by a frozen CycleGAN |

The ImageBind-Huge encoders stay frozen. Only the per-stream temporal self-attention blocks and the ~33k-parameter ComplexCNN are trained. The CycleGAN is trained once and frozen, and a trained generator ships with this repo.

## 1. Installation

```bash
git clone https://github.com/AnirudhPraveen/Complex-Valued-OV-AVEL.git
cd Complex-Valued-OV-AVEL/proposed_method/ImageBind-main

conda create -n ovavel python=3.10 -y
conda activate ovavel
# Install PyTorch for your CUDA version first: https://pytorch.org/get-started/locally/
pip install -r requirements.txt
```

All commands below are run from `proposed_method/ImageBind-main/`.

## 2. Data and checkpoints

### 2.1 OV-AVEBench

Download the **preprocessed** OV-AVEBench release from the OV-AVEL authors and extract it to `ovave_dataset_preprocessed/` at the repository root:

- Preprocessed audio + frames: [download](https://mailhfuteducn-my.sharepoint.com/:u:/g/personal/2018110964_mail_hfut_edu_cn/Efm9NKaGQFBAsOC2ZOMZRvcB26TKXJ84H4VW6g8BR5SukQ?e=OPgMOt)
- (Optional) raw videos: [download](https://mailhfuteducn-my.sharepoint.com/:u:/g/personal/2018110964_mail_hfut_edu_cn/EcVHOp2zOyVHvi1Au-i1zFQBf5wQNi-Yff9Aso_SJ4MV8Q?e=OeRlQh)

The meta/annotation files are already included in `meta_anno_files/`.

### 2.2 AVE-OV

AVE-OV re-casts AVE ([Tian et al., ECCV 2018](https://github.com/YapengTian/AVE-ECCV18)) into the open-vocabulary setting with 20 seen and 8 unseen classes (3,274 train / 401 test clips). Its split and annotations are included in `meta_anno_files/ave_ov/`. Download the AVE videos (the `AVE/` folder of `.mp4` files) from the AVE repository, then build the dataset:

```bash
pip install moviepy==1.0.3
python tools/prepare_ave_ov.py --ave_videos /path/to/AVE
```

This writes `AVE_Dataset/` at the repository root with the same layout as OV-AVEBench.

### 2.3 Checkpoints

- **ImageBind-Huge** (`imagebind_huge.pth`, ~4.5 GB) is downloaded automatically on the first run to `proposed_method/ImageBind-main/.checkpoints/`.
- **CycleGAN generator** (IF phase → mel, 45 MB) is included at `proposed_method/ImageBind-main/checkpoints/cyclegan_if2mel.pt`. To retrain it, see [Section 6](#6-optional-retrain-the-cyclegan).

### 2.4 Expected layout

```
Complex-Valued-OV-AVEL/
├── meta_anno_files/                     # included (OV-AVEBench + ave_ov/)
├── ovave_dataset_preprocessed/          # OV-AVEBench (Section 2.1)
│   └── {train,val,test}/
│       ├── audio/<category>/<video_id>.wav
│       └── video/<category>/<video_id>/<10 frames>.jpg
├── AVE_Dataset/                         # AVE-OV (Section 2.2), same structure, {train,test}
└── proposed_method/ImageBind-main/      # code: run everything from here
    ├── .checkpoints/imagebind_huge.pth  # auto-downloaded
    ├── checkpoints/cyclegan_if2mel.pt   # included
    ├── config.py                        # all model settings
    ├── train_complex_fusion.py          # training / evaluation
    └── run_*.sh                         # experiment scripts
```

If your data lives elsewhere, symlink it or set `OVAVEL_DATA_DIR` (OV-AVEBench), `OVAVEL_AVE_DIR` (AVE-OV), `IMAGEBIND_CKPT` or `CYCLEGAN_CKPT`.

## 3. Train and evaluate

```bash
bash run_complex_fusion.sh
```

This trains our 4-stream complex-fusion model on OV-AVEBench (Adam, lr 5e-5, batch size 32, 5 epochs). It keeps the best epoch by (Acc + Seg-F1 + Eve-F1)/3 on the *total* split, evaluates it on the **seen** (close), **unseen** (open) and **total** test splits, and prints a results table. Outputs go to `ExpResults/<dataset>/<run>/`.

The script is controlled by environment variables:

| Variable | Values |
|---|---|
| `OVAVEL_DATASET` | `ovavel` (OV-AVEBench, default) · `ave_ov` (AVE-OV, 10 epochs) |
| `STREAMS` | `4` (default) · `2` (RGB + mel only) |
| `FUSION` | `complex` (ComplexCNN, default) · `real` (parameter-matched RealCNN) · `none` (OV-AVEL fine-tuned baseline) |
| `PHASE_MODE` | `cyclegan` (default) · `imagebind` (raw IF through a learnable projector) |
| `GPU`, `SEED`, `N_EPOCH`, `LR`, `TRAIN_BS`, `TEST_BS` | training settings |

The model settings live in `config.py`, whose defaults are the paper's main model. The scripts set `STREAMS` / `FUSION` / `PHASE_MODE` through the `CFG_OVERRIDES_JSON` environment variable, so `config.py` never needs editing.

## 4. Reproducing the paper

| Paper result | Command |
|---|---|
| Main results, OV-AVEBench: 4-stream / 2-stream | `bash run_complex_fusion.sh` · `STREAMS=2 bash run_complex_fusion.sh` |
| Main results, AVE-OV: 4-stream / 2-stream | `OVAVEL_DATASET=ave_ov bash run_complex_fusion.sh` · `OVAVEL_DATASET=ave_ov STREAMS=2 bash run_complex_fusion.sh` |
| Fine-tuned baseline (re-implemented) | `FUSION=none bash run_complex_fusion.sh` (add `OVAVEL_DATASET=ave_ov` for AVE-OV) |
| Training-free baseline | `OVAVEL_DATASET=ave_ov bash run_baseline_v0.sh` |
| Ablation: Complex vs Real fusion, 2/4 streams, and CycleGAN phase vs raw IF | `bash run_ablations.sh` (add `OVAVEL_DATASET=ave_ov` for AVE-OV) |
| Cross-dataset transfer | `bash run_cross_dataset.sh <checkpoint> <target dataset>` (see below) |

The OV-AVEBench numbers for the prior methods and the baselines in our main table are taken from the OV-AVEL paper.

**Cross-dataset transfer.** Train on one dataset, then evaluate on the unseen categories of the other. Set `STREAMS` / `FUSION` to match the checkpoint:

```bash
# OV-AVEBench -> AVE-OV
bash run_cross_dataset.sh ExpResults/ovavel/4stream_complex_seed123/train/task_FullySupervised_best_model.pth.tar ave_ov
# AVE-OV -> OV-AVEBench
bash run_cross_dataset.sh ExpResults/ave_ov/4stream_complex_seed123/train/task_FullySupervised_best_model.pth.tar ovavel
```

**Evaluating a checkpoint only:**

```bash
python train_complex_fusion.py --evaluate --model eval --seed 123 \
  --resume path/to/task_FullySupervised_best_model.pth.tar \
  --test_data_type open --test_strategy_type v1 --test_batch_size 32 \
  --snapshot_pref ExpResults/eval_open
```

For a quick smoke test, add `--debug` to any `train_complex_fusion.py` command. It uses only 32 videos per split.

## 5. Notes

- As in the original OV-AVEL code, the best epoch is selected on the test split passed as `--test_data_type` (the `total` split during training). The validation split is not used.
- Runs are deterministic for a fixed seed on the same hardware and software (cuDNN deterministic mode).

## 6. (Optional) Retrain the CycleGAN

```bash
bash run_train_cyclegan.sh
CYCLEGAN_CKPT=./checkpoints/cyclegan/cyclegan_best.pt bash run_complex_fusion.sh
```

`run_train_cyclegan.sh` first extracts magnitude / phase / instantaneous-frequency spectrograms from the OV-AVEBench train and val audio. This needs well over 100 GB of disk; set `SPEC_DIR` to put them elsewhere. It then trains the unpaired IF ↔ mel CycleGAN with the paper's settings: 200 epochs, batch size 16, lr<sub>G</sub> 2e-4, lr<sub>D</sub> 1e-4, λ<sub>cyc</sub> = 10, λ<sub>idt</sub> = λ<sub>dist</sub> = 2.

## Citation

If you use this code, please cite our paper **and** the OV-AVEL paper it builds on:

```bibtex
@inproceedings{praveen2026complex,
  title     = {Open-Vocabulary Audio-Visual Event Localization via Complex-Valued Fusion},
  author    = {Praveen, Anirudh and Jerripothula, Koteswar Rao and Joshi, Pratik and Dayal, Aveen and Sawant, Neela},
  booktitle = {British Machine Vision Conference (BMVC)},
  year      = {2026}
}

@inproceedings{zhou2025towards,
  title     = {Towards Open-Vocabulary Audio-Visual Event Localization},
  author    = {Zhou, Jinxing and Guo, Dan and Guo, Ruohao and Mao, Yuxin and Hu, Jingjing and Zhong, Yiran and Chang, Xiaojun and Wang, Meng},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2025}
}
```

## Acknowledgements

This code builds on [OV-AVEL](https://github.com/jasongief/OV-AVEL), [ImageBind](https://github.com/facebookresearch/ImageBind) and the [AVE](https://github.com/YapengTian/AVE-ECCV18) dataset. We thank the authors for releasing their code, models and data.
