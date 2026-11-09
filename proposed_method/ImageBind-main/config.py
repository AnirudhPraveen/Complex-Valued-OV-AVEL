import os
from easydict import EasyDict as edict
import yaml
import pdb

"""
default config
"""
cfg = edict()
cfg.TRAIN_BG_CLASS_ID = 46 # train
cfg.TOTAL_BG_CLASS_ID = 67 # total
# Paths are resolved relative to the repository root so the code runs from any clone.
# Override with environment variables if your data lives elsewhere:
#   OVAVEL_DATA_DIR  -> preprocessed dataset (default: <repo>/ovave_dataset_preprocessed)
#   OVAVEL_META_DIR  -> meta/annotation files  (default: <repo>/meta_anno_files)
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_META_DIR = os.environ.get("OVAVEL_META_DIR", os.path.join(_REPO_ROOT, "meta_anno_files"))
cfg.DATA_BASE_DIR = os.environ.get("OVAVEL_DATA_DIR", os.path.join(_REPO_ROOT, "ovave_dataset_preprocessed"))
cfg.META_CSV_PATH = os.path.join(_META_DIR, "ovave_dataset_meta.csv")
# cfg.META_CSV_PATH = os.path.join(_META_DIR, "ovave_trainratio0.75_meta.csv")
cfg.ANNO_JSON_PATH = os.path.join(_META_DIR, "released_ovavel_dataset_anno.json")
cfg.TOTAL_CLOSE_OPEN_CATEGORY_CSV_PATH = os.path.join(_META_DIR, "ovave_total_close_open_categories.csv")
cfg.TRAIN_CLOSE_CATEGORY_CSV_PATH = os.path.join(_META_DIR, "ovave_train_close_categories.csv")

#### spatial attention
cfg.SA = edict()
cfg.SA.SA_ATTN_FLAG = False #!
cfg.SA.SA_ATTN_TYPE = "bothLast" # select from [bothFirst, bothLast, evenFirst, evenLast, fixedBlkids]
cfg.SA.SA_BOTH_LAYER_K = 1
cfg.SA.SA_EVEN_LAYER_K = 1
cfg.SA.SA_FIXED_LAYER_A = [0, 3]
cfg.SA.SA_FIXED_LAYER_V = [8, 16]

#### temporal attention
cfg.TA = edict()
cfg.TA.TA_ATTN_FLAG = True #!
cfg.TA.SA_LAYER_NUM = 1
cfg.TA.XA_LAYER_NUM = 1
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

#### text projection tuning
cfg.TEXT_TUNE_FLAG = False