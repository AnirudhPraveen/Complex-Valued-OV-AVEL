# Open-Vocabulary Audio-Visual Event Localization via Complex-Valued Fusion

Anirudh Praveen, Koteswar Rao Jerripothula, Pratik Joshi, Aveen Dayal, Neela Sawant

Official code for our BMVC 2026 paper *Open-Vocabulary Audio-Visual Event Localization via Complex-Valued Fusion*.

> This repository builds on the official code of **OV-AVEL** ([Zhou et al., CVPR 2025](https://arxiv.org/abs/2411.11278), [jasongief/OV-AVEL](https://github.com/jasongief/OV-AVEL)). The OV-AVEBench dataset, evaluation protocol and the training-free / fine-tuning baselines are theirs; please cite their work as well (see [Citation](#citation)).

## Overview

Open-Vocabulary Audio-Visual Event Localization (OV-AVEL) labels every one-second segment of a video with an event class, including classes never seen during training. Existing pipelines embed the visual frame, the audio mel-spectrogram and every candidate class name with a frozen ImageBind, compute a visual–text and an audio–text similarity, and collapse the two with a fixed rule (e.g. a geometric mean).

We instead build **complex-valued similarities** and learn their fusion with a **complex-valued CNN**:

| Modality | Real part | Imaginary (companion) part |
|---|---|---|
| Vision | RGB frame | imaginary part of iHSV |
| Audio | mel-spectrogram | instantaneous-frequency phase, translated to a mel-like image by a CycleGAN |

All encoders (ImageBind-Huge) stay frozen. Only the per-stream temporal-attention blocks and the ~33k-parameter ComplexCNN are trained. The CycleGAN is trained once and frozen; a trained generator ships with this repo.

## 1. Installation

```bash
git clone <this-repo-url> OV-AVEL
cd OV-AVEL/proposed_method/ImageBind-main

conda create -n ovavel python=3.10 -y
conda activate ovavel
# Install PyTorch for your CUDA version first: https://pytorch.org/get-started/locally/
pip install -r requirements.txt
```

## 2. Data and checkpoints

### 2.1 OV-AVEBench dataset

Download the **preprocessed** OV-AVEBench data (audio `.wav` + 10 frames per video) released by the OV-AVEL authors:

- Preprocessed data: [download link](https://mailhfuteducn-my.sharepoint.com/:u:/g/personal/2018110964_mail_hfut_edu_cn/Efm9NKaGQFBAsOC2ZOMZRvcB26TKXJ84H4VW6g8BR5SukQ?e=OPgMOt)
- (Optional) raw videos: [download link](https://mailhfuteducn-my.sharepoint.com/:u:/g/personal/2018110964_mail_hfut_edu_cn/EcVHOp2zOyVHvi1Au-i1zFQBf5wQNi-Yff9Aso_SJ4MV8Q?e=OeRlQh)

Extract it to `ovave_dataset_preprocessed/` at the **repository root**. The meta/annotation files are already included in `meta_anno_files/`.

### 2.2 ImageBind-Huge weights

Nothing to do: `imagebind_huge.pth` (~4.5 GB) is downloaded automatically on the first run to `proposed_method/ImageBind-main/.checkpoints/`. To download it manually:

```bash
mkdir -p proposed_method/ImageBind-main/.checkpoints
wget https://dl.fbaipublicfiles.com/imagebind/imagebind_huge.pth -P proposed_method/ImageBind-main/.checkpoints/
```

### 2.3 CycleGAN (phase → mel) generator

Included at `proposed_method/ImageBind-main/checkpoints/cyclegan_if2mel.pt` (45 MB). To retrain it yourself, see [Section 5](#5-optional-retrain-the-cyclegan).

### 2.4 Expected layout

```
OV-AVEL/
├── meta_anno_files/                     # included
│   ├── ovave_dataset_meta.csv
│   ├── released_ovavel_dataset_anno.json
│   └── ...
├── ovave_dataset_preprocessed/          # downloaded (Section 2.1)
│   ├── train/
│   │   ├── audio/<category>/<video_id>.wav
│   │   └── video/<category>/<video_id>/00000001.jpg ... 00000010.jpg
│   ├── val/    (same structure)
│   └── test/   (same structure)
└── proposed_method/ImageBind-main/      # code — run everything from here
    ├── .checkpoints/imagebind_huge.pth  # auto-downloaded (Section 2.2)
    ├── checkpoints/cyclegan_if2mel.pt   # included
    ├── configs/complex_fusion_4stream.json
    ├── train_complex_fusion.py
    └── run_complex_fusion.sh
```

If the data lives elsewhere, either symlink it (`ln -s /path/to/ovave_dataset_preprocessed .`) or set environment variables:

| Variable | Default |
|---|---|
| `OVAVEL_DATA_DIR` | `<repo>/ovave_dataset_preprocessed` |
| `OVAVEL_META_DIR` | `<repo>/meta_anno_files` |
| `IMAGEBIND_CKPT` | `<repo>/proposed_method/ImageBind-main/.checkpoints/imagebind_huge.pth` |
| `CYCLEGAN_CKPT` | `<repo>/proposed_method/ImageBind-main/checkpoints/cyclegan_if2mel.pt` |

## 3. Train and evaluate our model

```bash
cd proposed_method/ImageBind-main
bash run_complex_fusion.sh
```

This trains the 4-stream complex-fusion model (Adam, lr 5e-5, batch size 32, 5 epochs, best epoch selected on the validation *total* split), then evaluates the best checkpoint on the **close**, **open** and **total** test splits and prints a table of accuracy, segment-level F1 and event-level F1.

Useful overrides:

```bash
GPU=1 SEED=456 bash run_complex_fusion.sh      # other GPU / seed
N_EPOCH=10 TRAIN_BS=16 bash run_complex_fusion.sh
FUSION=real bash run_complex_fusion.sh         # ablation: real-valued CNN on the same 4 streams
```

Checkpoints and logs are written to `ExpResults/complex_fusion/`. The model configuration used in the paper is in `configs/complex_fusion_4stream.json` and is applied on top of `config.py` through the `CFG_OVERRIDES_JSON` environment variable.

To evaluate an existing checkpoint only:

```bash
CFG_OVERRIDES_JSON="$(cat configs/complex_fusion_4stream.json)" \
python train_complex_fusion.py --evaluate --model complex_fusion --seed 123 \
  --resume path/to/task_FullySupervised_best_model.pth.tar \
  --snapshot_pref ./ExpResults/complex_fusion/eval_open \
  --test_data_type open --val_data_type total --test_strategy_type v1 --test_batch_size 32
```

For a quick smoke test, add `--debug` to any `python train_complex_fusion.py ...` command: it uses only 32 videos per split.

## 4. OV-AVEL baselines

The original OV-AVEL baselines are kept for comparison:

```bash
cd proposed_method/ImageBind-main
bash run_baseline_v0.sh               # training-free baseline
bash run_baseline_v1_train_fully.sh   # fine-tuning baseline
```

## 5. (Optional) Retrain the CycleGAN

```bash
cd proposed_method/ImageBind-main
bash run_train_cyclegan.sh
CYCLEGAN_CKPT=./checkpoints/cyclegan/cyclegan_best.pt bash run_complex_fusion.sh
```

`run_train_cyclegan.sh` first extracts magnitude / phase / instantaneous-frequency spectrograms from the training and validation audio (needs well over 100 GB of free disk; set `SPEC_DIR` to put them elsewhere), then trains the unpaired IF-phase ↔ mel CycleGAN with the paper settings (200 epochs, batch size 16, lr<sub>G</sub> 2e-4, lr<sub>D</sub> 1e-4, λ<sub>cyc</sub> = 10, λ<sub>idt</sub> = λ<sub>dist</sub> = 2).

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

This code is built on [OV-AVEL](https://github.com/jasongief/OV-AVEL) and [ImageBind](https://github.com/facebookresearch/ImageBind). We thank the authors for releasing their code, models and the OV-AVEBench dataset.
