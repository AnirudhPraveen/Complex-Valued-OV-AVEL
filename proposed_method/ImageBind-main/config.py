import os as _os
import json as _json
from easydict import EasyDict as edict

"""
Configuration for complex-valued audio-visual fusion on OV-AVEL.

The defaults reproduce the paper's main model (4-stream + ComplexCNN on
OV-AVEBench). Every value can be overridden per run, without editing this file,
through the CFG_OVERRIDES_JSON environment variable (dotted keys), e.g.

    CFG_OVERRIDES_JSON='{"IHSV.ENABLED": false, "PHASE.ENABLED": false}'   # 2-stream
    CFG_OVERRIDES_JSON='{"FUSION_CNN_TYPE": "real"}'                       # RealCNN ablation
    CFG_OVERRIDES_JSON='{"PHASE.MODE": "imagebind"}'                       # raw-IF ablation
    CFG_OVERRIDES_JSON='{"FUSION_CNN_ENABLED": false, "IHSV.ENABLED": false,
                         "PHASE.ENABLED": false}'                          # fine-tuned baseline

The dataset is selected with the OVAVEL_DATASET environment variable:
    'ovavel' (OV-AVEBench, default) or 'ave_ov' (AVE-OV).
"""
cfg = edict()

# -----------------------------------------------------------------------------
# Dataset and paths (resolved relative to this repository; override with env vars)
#   OVAVEL_DATA_DIR  OV-AVEBench preprocessed data   (default <repo>/ovave_dataset_preprocessed)
#   OVAVEL_META_DIR  OV-AVEBench meta/annotations    (default <repo>/meta_anno_files)
#   OVAVEL_AVE_DIR   AVE-OV data root                (default <repo>/AVE_Dataset)
#   CYCLEGAN_CKPT    IF-phase -> mel CycleGAN generator
# -----------------------------------------------------------------------------
CODE_DIR = _os.path.dirname(_os.path.abspath(__file__))
REPO_ROOT = _os.path.abspath(_os.path.join(CODE_DIR, "..", ".."))
_OVAVEL_DATA = _os.environ.get("OVAVEL_DATA_DIR", _os.path.join(REPO_ROOT, "ovave_dataset_preprocessed"))
_OVAVEL_META = _os.environ.get("OVAVEL_META_DIR", _os.path.join(REPO_ROOT, "meta_anno_files"))
_AVE_DIR = _os.environ.get("OVAVEL_AVE_DIR", _os.path.join(REPO_ROOT, "AVE_Dataset"))
_AVE_OV_META = _os.path.join(REPO_ROOT, "meta_anno_files", "ave_ov")

cfg.DATASET = _os.environ.get("OVAVEL_DATASET", "ovavel")
if cfg.DATASET == "ave_ov":
    # AVE re-cast to open vocabulary: 20 seen + 8 unseen classes + "other"
    cfg.TRAIN_BG_CLASS_ID = 20  # "other" index in the 21-entry train category list
    cfg.TOTAL_BG_CLASS_ID = 28  # "other" index in the 29-entry total category list
    cfg.DATA_BASE_DIR = _AVE_DIR
    cfg.META_CSV_PATH = _os.path.join(_AVE_OV_META, "ave_ov_dataset_meta.csv")
    cfg.ANNO_JSON_PATH = _os.path.join(_AVE_OV_META, "ave_ov_annotations.json")
    cfg.TOTAL_CLOSE_OPEN_CATEGORY_CSV_PATH = _os.path.join(_AVE_OV_META, "ave_ov_total_close_open_categories.csv")
    cfg.TRAIN_CLOSE_CATEGORY_CSV_PATH = _os.path.join(_AVE_OV_META, "ave_ov_train_close_categories.csv")
elif cfg.DATASET == "ovavel":
    # OV-AVEBench: 46 seen + 21 unseen classes + "other"
    cfg.TRAIN_BG_CLASS_ID = 46
    cfg.TOTAL_BG_CLASS_ID = 67
    cfg.DATA_BASE_DIR = _OVAVEL_DATA
    cfg.META_CSV_PATH = _os.path.join(_OVAVEL_META, "ovave_dataset_meta.csv")
    cfg.ANNO_JSON_PATH = _os.path.join(_OVAVEL_META, "released_ovavel_dataset_anno.json")
    cfg.TOTAL_CLOSE_OPEN_CATEGORY_CSV_PATH = _os.path.join(_OVAVEL_META, "ovave_total_close_open_categories.csv")
    cfg.TRAIN_CLOSE_CATEGORY_CSV_PATH = _os.path.join(_OVAVEL_META, "ovave_train_close_categories.csv")
else:
    raise ValueError(f"Unknown OVAVEL_DATASET={cfg.DATASET!r} (expected 'ovavel' or 'ave_ov')")

# -----------------------------------------------------------------------------
# Per-stream temporal self-attention (one block per stream, trainable)
# -----------------------------------------------------------------------------
cfg.TA = edict()
cfg.TA.TA_ATTN_FLAG = True
cfg.TA.SA_LAYER_NUM = 1
cfg.TA.XA_LAYER_NUM = 1   # allocated at init (keeps the original OV-AVEL RNG stream); unused in forward
cfg.TA.HIDDEN_DIM = 256
cfg.TA.FF_DIM = 512
cfg.TA.HEAD_NUM = 1
cfg.TA.DROPOUT = 0.1
cfg.TA.USE_ADJ_IN_ATTN = False
cfg.TA.GAMMA = 0.6
cfg.TA.BIAS = 0.2
cfg.TA.USE_MASK_IN_ATTN = False
cfg.TA.WIN_SIZE = 5
cfg.TA.NORM_FLAG = None

# -----------------------------------------------------------------------------
# Fusion
#   FUSION_CNN_ENABLED=False -> OV-AVEL fine-tuned baseline: sqrt(relu(s_at * s_vt))
#   FUSION_CNN_TYPE='complex' -> ComplexCNN (ours); 'real' -> parameter-matched RealCNN
# -----------------------------------------------------------------------------
cfg.FUSION_CNN_ENABLED = True
cfg.FUSION_CNN_TYPE = 'complex'

# -----------------------------------------------------------------------------
# Visual imaginary stream: iHSV (HSV -> complex -> imaginary part -> RGB)
# -----------------------------------------------------------------------------
cfg.IHSV = edict()
cfg.IHSV.ENABLED = True
cfg.IHSV.RESCALE_MODE = 'minmax'   # maps the iHSV imaginary image into [0, 1]

# -----------------------------------------------------------------------------
# Audio imaginary stream: instantaneous-frequency phase
#   MODE='cyclegan'  -> IF -> frozen CycleGAN G_IF2Mel -> frozen ImageBind audio encoder (ours)
#   MODE='imagebind' -> IF -> learnable projector -> frozen ImageBind audio encoder (raw-IF ablation)
# -----------------------------------------------------------------------------
cfg.PHASE = edict()
cfg.PHASE.ENABLED = True
cfg.PHASE.MODE = 'cyclegan'
cfg.PHASE.CYCLEGAN_CKPT = _os.environ.get(
    'CYCLEGAN_CKPT', _os.path.join(CODE_DIR, 'checkpoints', 'cyclegan_if2mel.pt'))


# -----------------------------------------------------------------------------
# Per-run overrides via CFG_OVERRIDES_JSON (dotted keys -> values)
# -----------------------------------------------------------------------------
def _apply_env_overrides(_cfg):
    raw = _os.environ.get("CFG_OVERRIDES_JSON", "").strip()
    if not raw:
        return
    try:
        overrides = _json.loads(raw)
    except Exception as exc:
        raise ValueError(f"CFG_OVERRIDES_JSON is not valid JSON: {exc}")
    for dotted_key, value in overrides.items():
        parts = dotted_key.split(".")
        target = _cfg
        for p in parts[:-1]:
            if p not in target:
                raise KeyError(f"CFG_OVERRIDES_JSON: unknown config key {dotted_key!r}")
            target = target[p]
        if parts[-1] not in target:
            raise KeyError(f"CFG_OVERRIDES_JSON: unknown config key {dotted_key!r}")
        target[parts[-1]] = value
    print(f"[config.py] applied CFG_OVERRIDES_JSON: {overrides}")


_apply_env_overrides(cfg)
