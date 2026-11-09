# Code

See the [repository README](../../README.md) for installation, data preparation and how to reproduce every result.

| File | Purpose |
|---|---|
| `run_complex_fusion.sh` | train + evaluate one model (`STREAMS`, `FUSION`, `PHASE_MODE`, `OVAVEL_DATASET`) |
| `run_ablations.sh` | Complex vs Real fusion (2/4 streams) and CycleGAN-phase vs raw-IF ablations |
| `run_cross_dataset.sh` | cross-dataset transfer evaluation |
| `run_baseline_v0.sh` | OV-AVEL training-free baseline |
| `run_train_cyclegan.sh` | (optional) retrain the IF-phase → mel CycleGAN |
| `train_complex_fusion.py` | training / evaluation entry point |
| `config.py` | model and dataset settings (defaults = paper's 4-stream ComplexCNN) |
| `tools/prepare_ave_ov.py` | build AVE-OV from the original AVE videos |
