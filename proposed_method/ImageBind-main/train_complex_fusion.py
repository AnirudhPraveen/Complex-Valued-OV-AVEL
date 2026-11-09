import os
import time
import random
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR


import numpy as np
from configs.opts import parser
from utils import AverageMeter, Prepare_logger, get_and_save_args


# ============================================================================
# Post-training analyses on the best epoch: boundary F1 (±1 frame), tSNE plot,
# top1-top2 sharpness histogram, and a raw predictions dump (preds_<split>.npz).
# ============================================================================
def _extract_boundaries(event_seq):
    event_seq = np.asarray(event_seq).astype(np.int32)
    diff = np.diff(event_seq, prepend=event_seq[0])
    onsets = np.where(diff == 1)[0]
    offsets = np.where(diff == -1)[0]
    return onsets, offsets


def compute_boundary_f1(pred_event_seqs, gt_event_seqs, tolerance=1):
    def _match(pred_idx, gt_idx, tol):
        matched = 0
        used_pred = set()
        for g in gt_idx:
            best, best_d = -1, tol + 1
            for i, p in enumerate(pred_idx):
                if i in used_pred:
                    continue
                d = abs(p - g)
                if d <= tol and d < best_d:
                    best, best_d = i, d
            if best != -1:
                matched += 1
                used_pred.add(best)
        return matched

    on_tp = off_tp = 0
    on_fp = off_fp = 0
    on_fn = off_fn = 0

    for pred_seq, gt_seq in zip(pred_event_seqs, gt_event_seqs):
        pred_on, pred_off = _extract_boundaries(pred_seq)
        gt_on, gt_off = _extract_boundaries(gt_seq)
        m_on = _match(list(pred_on), list(gt_on), tolerance)
        m_off = _match(list(pred_off), list(gt_off), tolerance)
        on_tp += m_on
        off_tp += m_off
        on_fp += len(pred_on) - m_on
        off_fp += len(pred_off) - m_off
        on_fn += len(gt_on) - m_on
        off_fn += len(gt_off) - m_off

    def _prf(tp, fp, fn):
        p = tp / (tp + fp + 1e-8)
        r = tp / (tp + fn + 1e-8)
        f = 2 * p * r / (p + r + 1e-8)
        return p, r, f

    on_p, on_r, on_f = _prf(on_tp, on_fp, on_fn)
    off_p, off_r, off_f = _prf(off_tp, off_fp, off_fn)
    c_p, c_r, c_f = _prf(on_tp + off_tp, on_fp + off_fp, on_fn + off_fn)

    return {
        'onset_precision': on_p, 'onset_recall': on_r, 'onset_f1': on_f,
        'offset_precision': off_p, 'offset_recall': off_r, 'offset_f1': off_f,
        'boundary_precision': c_p, 'boundary_recall': c_r, 'boundary_f1': c_f,
        'tolerance_frames': tolerance,
    }


def compute_sharpness(pred_logits_all):
    pred_logits_all = np.asarray(pred_logits_all)
    sorted_desc = -np.sort(-pred_logits_all, axis=-1)
    gap = sorted_desc[:, 0] - sorted_desc[:, 1]
    return {'sharpness_mean': float(gap.mean()),
            'sharpness_median': float(np.median(gap)),
            'sharpness_values': gap}


def run_tsne_plot(pred_logits_all, pred_classes_all, gt_event_all, gt_category_all,
                  bg_class_id, save_path, title, max_points=3000, seed=123):
    try:
        from sklearn.manifold import TSNE
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[tsne] skipped ({e})")
        return None

    X = np.asarray(pred_logits_all)
    ev = np.asarray(gt_event_all).astype(np.int32)
    pc = np.asarray(pred_classes_all).astype(np.int32)

    N = X.shape[0]
    if N > max_points:
        rng = np.random.RandomState(seed)
        idx = rng.choice(N, max_points, replace=False)
        X, ev, pc = X[idx], ev[idx], pc[idx]

    emb = TSNE(n_components=2, random_state=seed,
               perplexity=min(30, max(5, len(X) // 50)),
               init='pca', learning_rate='auto').fit_transform(X)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for lab, color, name in [(0, 'tab:gray', 'background'), (1, 'tab:red', 'event')]:
        mask = ev == lab
        axes[0].scatter(emb[mask, 0], emb[mask, 1], s=6, alpha=0.5, c=color, label=name)
    axes[0].set_title(f"{title} — event vs background")
    axes[0].legend(markerscale=2)
    axes[0].set_xticks([]); axes[0].set_yticks([])

    uniq, cnts = np.unique(pc[pc != bg_class_id], return_counts=True)
    top_ids = uniq[np.argsort(-cnts)[:8]] if len(uniq) else []
    cmap = plt.get_cmap('tab10')
    for i, cid in enumerate(top_ids):
        mask = pc == cid
        axes[1].scatter(emb[mask, 0], emb[mask, 1], s=6, alpha=0.6,
                        c=[cmap(i)], label=f"cls {cid}")
    bg_mask = pc == bg_class_id
    axes[1].scatter(emb[bg_mask, 0], emb[bg_mask, 1], s=4, alpha=0.3, c='lightgray', label='BG')
    axes[1].set_title(f"{title} — top categories (argmax)")
    axes[1].legend(fontsize=7, markerscale=2, loc='best')
    axes[1].set_xticks([]); axes[1].set_yticks([])

    plt.tight_layout()
    plt.savefig(save_path, dpi=110)
    plt.close(fig)
    return save_path


def run_sharpness_plot(sharp_values, save_path, title):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return None
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(sharp_values, bins=50, color='steelblue', alpha=0.8)
    ax.set_xlabel("top1 − top2 (per-frame confidence gap)")
    ax.set_ylabel("# frames")
    ax.set_title(f"{title}\nmean={np.mean(sharp_values):.4f}  median={np.median(sharp_values):.4f}")
    plt.tight_layout()
    plt.savefig(save_path, dpi=110)
    plt.close(fig)
    return save_path


# ============================================================================
# Complex-Valued CNN for Audio-Visual Fusion
# ============================================================================

class ComplexConv2d(nn.Module):
    """
    Complex-valued 2D convolution layer.

    For complex input z = x + iy and complex weight W = A + iB:
    W * z = (A*x - B*y) + i(A*y + B*x)
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super(ComplexConv2d, self).__init__()
        self.conv_real = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        self.conv_imag = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)

    def forward(self, real, imag):
        # W * z = (A*x - B*y) + i(A*y + B*x)
        # where A = conv_real weights, B = conv_imag weights
        # x = real input, y = imag input
        real_out = self.conv_real(real) - self.conv_imag(imag)
        imag_out = self.conv_real(imag) + self.conv_imag(real)
        return real_out, imag_out


class ComplexBatchNorm2d(nn.Module):
    """
    Batch normalization for complex values.
    Applies separate batch norm to real and imaginary parts.
    """
    def __init__(self, num_features):
        super(ComplexBatchNorm2d, self).__init__()
        self.bn_real = nn.BatchNorm2d(num_features)
        self.bn_imag = nn.BatchNorm2d(num_features)

    def forward(self, real, imag):
        return self.bn_real(real), self.bn_imag(imag)


class ComplexReLU(nn.Module):
    """
    CReLU: ReLU applied separately to the real and imaginary parts.
    """
    def forward(self, real, imag):
        return F.relu(real), F.relu(imag)


class ComplexCNN(nn.Module):
    """
    5-layer Complex-Valued CNN for multi-modal fusion.

    4-stream (in_channels=2):
        - Channel 0: z_v = simm_vt + i·simm_imag_vt   (visual: RGB + i·iHSV)
        - Channel 1: z_a = simm_at + i·simm_pt        (audio:  mel + i·CycleGAN-phase)

    2-stream (in_channels=1):
        - Channel 0: z = simm_vt + i·simm_at

    Output:
        - magnitude of complex output [bs, 10, C+1]

    Note: Uses (temporal_kernel, 1) kernel to capture temporal patterns while
    remaining category-agnostic (important for open-vocabulary setting where
    train has 47 categories but test has 68 categories).
    """
    def __init__(self, hidden_channels=32, temporal_kernel=5, in_channels=1):
        super(ComplexCNN, self).__init__()

        t_pad = temporal_kernel // 2  # padding to maintain temporal dimension (10 -> 10)

        def make_act(num_features):
            return ComplexReLU()

        # Layer 1: in_channels -> hidden_channels (temporal convolution only)
        self.conv1 = ComplexConv2d(in_channels, hidden_channels, kernel_size=(temporal_kernel, 1), padding=(t_pad, 0))
        self.bn1 = ComplexBatchNorm2d(hidden_channels)
        self.act1 = make_act(hidden_channels)

        # Layer 2: hidden_channels -> hidden_channels (temporal convolution only)
        self.conv2 = ComplexConv2d(hidden_channels, hidden_channels, kernel_size=(temporal_kernel, 1), padding=(t_pad, 0))
        self.bn2 = ComplexBatchNorm2d(hidden_channels)
        self.act2 = make_act(hidden_channels)

        # Layer 3: hidden_channels -> hidden_channels (temporal convolution only)
        self.conv3 = ComplexConv2d(hidden_channels, hidden_channels, kernel_size=(temporal_kernel, 1), padding=(t_pad, 0))
        self.bn3 = ComplexBatchNorm2d(hidden_channels)
        self.act3 = make_act(hidden_channels)

        # Layer 4: hidden_channels -> hidden_channels (temporal convolution only)
        self.conv4 = ComplexConv2d(hidden_channels, hidden_channels, kernel_size=(temporal_kernel, 1), padding=(t_pad, 0))
        self.bn4 = ComplexBatchNorm2d(hidden_channels)
        self.act4 = make_act(hidden_channels)

        # Layer 5: hidden_channels -> 1 (1x1 conv to reduce channels back to 1)
        self.conv5 = ComplexConv2d(hidden_channels, 1, kernel_size=(1, 1), padding=0)
        # No activation after final layer - we'll take magnitude

    def forward(self, simm_vt, simm_at, simm_imag_vt=None, simm_pt=None,
                return_phase=False):
        """
        Complex channel packing:
          4-stream: ch0 = simm_vt + i·simm_imag_vt   (visual: RGB + i·iHSV)
                    ch1 = simm_at + i·simm_pt        (audio:  mel + i·CycleGAN-phase)
                    (a missing companion stream is replaced by zeros)
          2-stream: ch0 = simm_vt + i·simm_at
        """
        has_extra = (simm_imag_vt is not None) or (simm_pt is not None)

        if not has_extra:
            real = simm_vt.unsqueeze(1)
            imag = simm_at.unsqueeze(1)
        else:
            zero_like = torch.zeros_like(simm_vt)
            iv = simm_imag_vt if simm_imag_vt is not None else zero_like
            pt = simm_pt      if simm_pt      is not None else zero_like
            real = torch.stack([simm_vt, simm_at], dim=1)
            imag = torch.stack([iv, pt], dim=1)

        # Layer 1
        real, imag = self.conv1(real, imag)
        real, imag = self.bn1(real, imag)
        real, imag = self.act1(real, imag)

        # Layer 2
        real, imag = self.conv2(real, imag)
        real, imag = self.bn2(real, imag)
        real, imag = self.act2(real, imag)

        # Layer 3
        real, imag = self.conv3(real, imag)
        real, imag = self.bn3(real, imag)
        real, imag = self.act3(real, imag)

        # Layer 4
        real, imag = self.conv4(real, imag)
        real, imag = self.bn4(real, imag)
        real, imag = self.act4(real, imag)

        # Layer 5
        real, imag = self.conv5(real, imag)

        # Compute magnitude: |z| = sqrt(real^2 + imag^2)
        magnitude = torch.sqrt(real ** 2 + imag ** 2 + 1e-8)  # [bs, 1, 10, C+1]

        # Remove channel dimension: [bs, 10, C+1]
        pred_logits = magnitude.squeeze(1)

        if return_phase:
            phase = torch.atan2(imag, real).squeeze(1)  # [bs, 10, C+1] in [-pi, pi]
            return pred_logits, phase
        return pred_logits


# ============================================================================
# Real-Valued CNN for Audio-Visual Fusion (Alternative to ComplexCNN)
# ============================================================================
class RealCNN(nn.Module):
    """
    5-layer Real-Valued CNN for audio-visual fusion.

    4-channel input (stacked along channel dim):
        ch0: simm_vt       — RGB visual · text
        ch1: simm_imag_vt  — iHSV visual · text
        ch2: simm_at       — mel-audio · text
        ch3: simm_pt       — phase-audio · text
    Missing streams are replaced with zeros.

    Output:
        - refined prediction logits [bs, 10, C+1]
    """
    def __init__(self, hidden_channels=32, temporal_kernel=5, in_channels=4,
                 nonneg_activation='abs'):
        super(RealCNN, self).__init__()

        # Output non-negativity activation. Options: 'abs' (default, mirrors
        # ComplexCNN's |z| = sqrt(real^2 + imag^2) on a 1D output), 'softplus'
        # (smooth, but not a magnitude), 'none' (raw signed logits).
        assert nonneg_activation in ('softplus', 'abs', 'none')
        self.nonneg_activation = nonneg_activation

        t_pad = temporal_kernel // 2  # padding to maintain temporal dimension (10 -> 10)

        # Layer 1: in_channels -> hidden_channels
        self.conv1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=(temporal_kernel, 1), padding=(t_pad, 0))
        self.bn1 = nn.BatchNorm2d(hidden_channels)

        # Layer 2: hidden_channels -> hidden_channels
        self.conv2 = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=(temporal_kernel, 1), padding=(t_pad, 0))
        self.bn2 = nn.BatchNorm2d(hidden_channels)

        # Layer 3: hidden_channels -> hidden_channels
        self.conv3 = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=(temporal_kernel, 1), padding=(t_pad, 0))
        self.bn3 = nn.BatchNorm2d(hidden_channels)

        # Layer 4: hidden_channels -> hidden_channels
        self.conv4 = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=(temporal_kernel, 1), padding=(t_pad, 0))
        self.bn4 = nn.BatchNorm2d(hidden_channels)

        # Layer 5: hidden_channels -> 1
        self.conv5 = nn.Conv2d(hidden_channels, 1, kernel_size=(1, 1), padding=0)

    def forward(self, simm_vt, simm_imag_vt, simm_at, simm_pt=None):
        """
        Args:
            simm_vt:      RGB visual-text similarity   [bs, 10, C+1]
            simm_imag_vt: iHSV visual-text similarity  [bs, 10, C+1] (None → omitted)
            simm_at:      Mel audio-text similarity    [bs, 10, C+1]
            simm_pt:      Phase audio-text similarity  [bs, 10, C+1] (None → omitted)

        Returns:
            pred_logits: Refined output [bs, 10, C+1]
        """
        channels = [simm_vt, simm_at]
        if simm_imag_vt is not None:
            channels.append(simm_imag_vt)
        if simm_pt is not None:
            channels.append(simm_pt)
        x = torch.stack(channels, dim=1)  # [bs, in_channels, 10, C+1]

        # Layer 1
        x = F.relu(self.bn1(self.conv1(x)))

        # Layer 2
        x = F.relu(self.bn2(self.conv2(x)))

        # Layer 3
        x = F.relu(self.bn3(self.conv3(x)))

        # Layer 4
        x = F.relu(self.bn4(self.conv4(x)))

        # Layer 5
        x = self.conv5(x)

        # Enforce non-negativity (to mirror ComplexCNN magnitude output).
        # Switchable: abs (default, |x| = sqrt(x^2) — 1D analog of |z|),
        # softplus (smooth alternative), none (signed logits).
        if self.nonneg_activation == 'abs':
            x = torch.abs(x)
        elif self.nonneg_activation == 'softplus':
            x = F.softplus(x)
        # 'none' → leave x as-is

        # Remove channel dimension: [bs, 10, C+1]
        return x.squeeze(1)


# ============================================================================
# Helper function for min-max normalization
# ============================================================================
def min_max_normalize(x, dim=-1):
    """
    Min-max normalize tensor along specified dimension.
    Args:
        x: input tensor
        dim: dimension to normalize along (default: -1, i.e., category dimension)
    Returns:
        normalized tensor with values in [0, 1]
    """
    x_min = x.min(dim, keepdim=True)[0]
    x_max = x.max(dim, keepdim=True)[0]
    return (x - x_min) / (x_max - x_min + 1e-8)
from utils.eval_metrics import segment_level, event_level


from imagebind_finetune.models import imagebind_model_fully
from imagebind_finetune.models.imagebind_model_fully import ModalityType

from dataloader_fully import OVAVE_Dataset
from config import cfg
from diagnose_segment_acc import Diagnoser
from layers.phase_encoder import PhaseToMelProjector, compute_if
from cyclegan_phase2mel import load_trained_cyclegan, translate_phase_to_mel
from layers.temporal_av_attn_layer import TransformerEncoder
from copy import deepcopy as dcp






# configs
# dataset_configs = get_and_save_args(parser)
# parser.set_defaults(**dataset_configs)
# args = parser.parse_args()
args = get_and_save_args(parser)
# print parameters
# print('----------------args-----------------')
# for k in list(vars(args).keys()):
#     print('%s: %s' % (k, vars(args)[k]))
# print('----------------args-----------------')



 # =================================  seed config ============================
SEED = args.seed
random.seed(SEED)
np.random.seed(seed=SEED)
torch.manual_seed(seed=SEED)
torch.cuda.manual_seed_all(seed=SEED)
torch.backends.cudnn.deterministic = True   # match original; trade speed for reproducibility
torch.backends.cudnn.benchmark = False      # match original
# =============================================================================


# select GPUs
# os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
# Honor CUDA_VISIBLE_DEVICES from the runner; default to GPU 0 if unset.
os.environ.setdefault('CUDA_VISIBLE_DEVICES', "0")

'''Create snapshot_pred dir for copying code and saving models '''
if not os.path.exists(args.snapshot_pref):
    os.makedirs(args.snapshot_pref, exist_ok=True)

# Previously: `args.snapshot_pref = os.path.dirname(args.resume)` whenever
# --resume pointed to a real file. That clobbered any user-supplied
# --snapshot_pref and dumped all eval logs into the train-checkpoint dir, so
# orchestrators that run many evals into per-config subdirs (e.g. the LA
# sweep in run_full_pipeline.sh) saw their <ts>-Eval.log files written to
# the wrong place. Removed; --snapshot_pref is now respected as passed.
# if os.path.isfile(args.resume):
#     args.snapshot_pref = os.path.dirname(args.resume)

logger = Prepare_logger(args, eval=args.evaluate)

ext = "_noSattn"
audio_layer_ids = []
visual_layer_ids = []
sattn_type = 'none'
# print(f"==> Spatial Attention Mode: {sattn_type}")
# print(f"==> Spatial Attention audio_layer_ids: {audio_layer_ids}")
# print(f"==> Spatial Attention visual_layer_ids: {visual_layer_ids}")
logger.info(f"==> Spatial Attention Mode: {sattn_type}")
logger.info(f"==> Spatial Attention audio_layer_ids: {audio_layer_ids}")
logger.info(f"==> Spatial Attention visual_layer_ids: {visual_layer_ids}")
logger.info(f"==> Audio-Visual Similarity Modulation: DISABLED")


if not args.evaluate:
    logger.info(f'\nCreating folder: {args.snapshot_pref}')
    logger.info('\nRuntime args\n\n{}\n'.format(json.dumps(vars(args), indent=4)))
else:
    logger.info(f'\nLog file will be save in {args.snapshot_pref}/Eval_{args.test_data_type}.log.')
    logger.info('\nRuntime args\n\n{}\n'.format(json.dumps(vars(args), indent=4)))


def main():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    train_dataloader = DataLoader(
        OVAVE_Dataset(split='train', debug=args.debug),
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True
    )
    # NOTE: as in the original OV-AVEL code, the best epoch is selected on the
    # test split given by --test_data_type (the 'val' split is not used).
    test_dataloader = DataLoader(
        OVAVE_Dataset(split='test', test_data_type=args.test_data_type, debug=args.debug),
        batch_size=args.test_batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True
    )
    logger.info(f"==> Logit Adjustment: DISABLED")

    ''' Model selection '''
    mainModel = imagebind_model_fully.imagebind_huge(
        pretrained=True,
        # spatial attention,
        spatial_av_attn_layer_ids=(audio_layer_ids, visual_layer_ids),
        sattn_flag=sattn_type,
        # temporal attention,
        tattn_flag=cfg.TA.TA_ATTN_FLAG,
        sa_layer_num=cfg.TA.SA_LAYER_NUM,
        xa_layer_num=cfg.TA.XA_LAYER_NUM,
        hid_dim=cfg.TA.HIDDEN_DIM,
        d_ff=cfg.TA.FF_DIM,
        head_num=cfg.TA.HEAD_NUM,
        dropout=cfg.TA.DROPOUT,
        use_adj_in_attn=cfg.TA.USE_ADJ_IN_ATTN,
        gamma=cfg.TA.GAMMA,
        bias=cfg.TA.BIAS,
        use_mask_in_attn=cfg.TA.USE_MASK_IN_ATTN,
        win_size=cfg.TA.WIN_SIZE,
        norm_flag=cfg.TA.NORM_FLAG,
    )
    # mainModel.eval()
    mainModel.to(device)

    # Log temporal attention status
    logger.info(f"==> REAL TEMPORAL ATTENTION: ENABLED (TemporalAttentionModule)")
    logger.info(f"    Standard real-valued self-attention on temporal dimension")
    logger.info(f"    sa_layer_num={cfg.TA.SA_LAYER_NUM}, head_num={cfg.TA.HEAD_NUM}")

    #! =====================================================================
    #! FUSION CNN (can be disabled via config)
    #! =====================================================================
    fusion_cnn_type = cfg.FUSION_CNN_TYPE   # 'complex' (ours) or 'real' (ablation)
    USE_REAL_CNN = (fusion_cnn_type == 'real')

    if cfg.FUSION_CNN_ENABLED:
        if fusion_cnn_type == 'real':
            ''' Parameter-matched stacked-channel RealCNN (ablation): up to 4 real channels
                [simm_vt, simm_at, (+ simm_imag_vt if iHSV), (+ simm_pt if phase)]. '''
            real_in_channels = 2  # simm_vt, simm_at always present
            ch_desc = ["simm_vt (RGB visual)", "simm_at (mel audio)"]
            if cfg.IHSV.ENABLED:
                real_in_channels += 1
                ch_desc.append("simm_imag_vt (iHSV visual)")
            if cfg.PHASE.ENABLED:
                real_in_channels += 1
                ch_desc.append("simm_pt (phase audio)")
            real_hidden = 45
            fusionCNN = RealCNN(hidden_channels=real_hidden, in_channels=real_in_channels)
            fusionCNN.to(device)
            logger.info(f"==> Real CNN for AV Fusion: ENABLED (5 layers, in_channels={real_in_channels}, hidden_channels={real_hidden}, temporal_kernel=5)")
            for i, d in enumerate(ch_desc):
                logger.info(f"    ch{i}: {d}")
        else:
            ''' ComplexCNN: 2 complex channels when any companion stream (iHSV or phase) is
                enabled (a missing companion is zero-filled), else 1 channel (2-stream). '''
            complex_in_ch = 2 if (cfg.IHSV.ENABLED or cfg.PHASE.ENABLED) else 1
            cplx_tk = 5
            cplx_hidden = 32
            fusionCNN = ComplexCNN(hidden_channels=cplx_hidden, in_channels=complex_in_ch,
                                   temporal_kernel=cplx_tk)
            fusionCNN.to(device)
            logger.info(f"==> Complex CNN for AV Fusion: ENABLED (5 layers, in_channels={complex_in_ch}, hidden_channels={cplx_hidden}, temporal_kernel={cplx_tk}, activation='crelu')")
            if complex_in_ch == 2:
                logger.info(f"    ch0 = simm_vt + i·simm_imag_vt   (visual real + visual imag)")
                logger.info(f"    ch1 = simm_at + i·simm_pt        (audio real  + audio imag)")
            else:
                logger.info(f"    z   = simm_vt + i·simm_at        (real=RGB visual, imag=audio mel)")
    else:
        fusionCNN = None
        logger.info(f"==> Fusion CNN: DISABLED (using direct similarity fusion)")

    phase_projector = None
    phase_temporal_attn = None
    cyclegan_generator = None
    cyclegan_if_range = None
    cyclegan_mel_range = None
    if cfg.PHASE.ENABLED:
        if cfg.PHASE.MODE == 'imagebind':
            # Learnable projection: phase [1, 257, 204] → mel shape [1, 128, 204]
            phase_projector = PhaseToMelProjector(in_freq=257, out_freq=128).to(device)
            # Phase gets its own temporal self-attention (same architecture as audio's)
            phase_sa_block = TransformerEncoder(
                d_model=1024,
                hid_dim=cfg.TA.HIDDEN_DIM,
                d_ff=cfg.TA.FF_DIM,
                n_heads=cfg.TA.HEAD_NUM,
                dropout=cfg.TA.DROPOUT,
                use_adj_in_attn=cfg.TA.USE_ADJ_IN_ATTN,
                gamma=cfg.TA.GAMMA,
                bias=cfg.TA.BIAS,
                use_mask_in_attn=cfg.TA.USE_MASK_IN_ATTN,
                win_size=cfg.TA.WIN_SIZE,
                norm_flag=cfg.TA.NORM_FLAG,
            )
            phase_temporal_attn = nn.Sequential(
                *[dcp(phase_sa_block) for _ in range(cfg.TA.SA_LAYER_NUM)]
            ).to(device)
            logger.info(f"==> Phase via ImageBind: ENABLED (mode='imagebind', input_type='if')")
            logger.info(f"    Pipeline: phase [1,257,204] → proj [1,128,204] → ImageBind audio encoder → [1024]")
            logger.info(f"    Phase temporal self-attention: {cfg.TA.SA_LAYER_NUM} layers, hid={cfg.TA.HIDDEN_DIM}, heads={cfg.TA.HEAD_NUM}")
            logger.info(f"    Output: simm_pt [bs, 10, C+1] (phase-text similarity)")

        elif cfg.PHASE.MODE == 'cyclegan':
            # Vanilla (unconditional) CycleGAN: G_IF2Mel takes IF → mel-like spectrogram
            # Generator is FROZEN — only temporal attention is trainable
            cyclegan_generator, cyclegan_if_range, cyclegan_mel_range = load_trained_cyclegan(
                cfg.PHASE.CYCLEGAN_CKPT, device=device)
            cyclegan_generator.eval()
            for p in cyclegan_generator.parameters():
                p.requires_grad = False
            # Phase gets its own temporal self-attention (same as other modes)
            phase_sa_block = TransformerEncoder(
                d_model=1024,
                hid_dim=cfg.TA.HIDDEN_DIM,
                d_ff=cfg.TA.FF_DIM,
                n_heads=cfg.TA.HEAD_NUM,
                dropout=cfg.TA.DROPOUT,
                use_adj_in_attn=cfg.TA.USE_ADJ_IN_ATTN,
                gamma=cfg.TA.GAMMA,
                bias=cfg.TA.BIAS,
                use_mask_in_attn=cfg.TA.USE_MASK_IN_ATTN,
                win_size=cfg.TA.WIN_SIZE,
                norm_flag=cfg.TA.NORM_FLAG,
            )
            phase_temporal_attn = nn.Sequential(
                *[dcp(phase_sa_block) for _ in range(cfg.TA.SA_LAYER_NUM)]
            ).to(device)
            logger.info(f"==> Phase via vanilla CycleGAN: ENABLED (mode='cyclegan')")
            logger.info(f"    Checkpoint: {cfg.PHASE.CYCLEGAN_CKPT}")
            logger.info(f"    Pipeline: phase [1,257,204] → compute_if → CycleGAN G_IF2Mel (frozen, unconditional) → mel-like [1,128,204] → ImageBind audio encoder (frozen) → [1024]")
            logger.info(f"    Phase temporal self-attention: {cfg.TA.SA_LAYER_NUM} layers, hid={cfg.TA.HIDDEN_DIM}, heads={cfg.TA.HEAD_NUM}")

    else:
        logger.info(f"==> Phase: DISABLED")
    logger.info(f"==> Depth Temporal Self-Attention: DISABLED (DEPTH not enabled)")

    #! =====================================================================
    #! iHSV TEMPORAL SELF-ATTENTION (optional, when IHSV is enabled)
    #! =====================================================================
    ihsv_temporal_attn = None
    if cfg.IHSV.ENABLED:
        ihsv_sa_block = TransformerEncoder(
            d_model=1024,
            hid_dim=cfg.TA.HIDDEN_DIM,
            d_ff=cfg.TA.FF_DIM,
            n_heads=cfg.TA.HEAD_NUM,
            dropout=cfg.TA.DROPOUT,
            use_adj_in_attn=cfg.TA.USE_ADJ_IN_ATTN,
            gamma=cfg.TA.GAMMA,
            bias=cfg.TA.BIAS,
            use_mask_in_attn=cfg.TA.USE_MASK_IN_ATTN,
            win_size=cfg.TA.WIN_SIZE,
            norm_flag=cfg.TA.NORM_FLAG,
        )
        ihsv_temporal_attn = nn.Sequential(
            *[dcp(ihsv_sa_block) for _ in range(cfg.TA.SA_LAYER_NUM)]
        ).to(device)
        logger.info(f"==> iHSV Temporal Self-Attention: ENABLED")
        logger.info(f"    {cfg.TA.SA_LAYER_NUM} layers, hid={cfg.TA.HIDDEN_DIM}, heads={cfg.TA.HEAD_NUM}")
    else:
        logger.info(f"==> iHSV Temporal Self-Attention: DISABLED (IHSV not enabled)")
    logger.info(f"==> Phase-Variance FG/BG Gate (Idea 4): DISABLED")

    #! =====================================================================
    #! Idea 1 — AV phase coherence loss flag (no module; just an auxiliary loss)
    #! =====================================================================
    logger.info(f"==> AV Phase Coherence Loss (Idea 1): DISABLED")

    '''optimizer setting'''
    # mainModel = nn.DataParallel(mainModel).cuda()
    # learned_parameters = []
    for name, params in mainModel.named_parameters():
        if 'spatial_av_layers' in name:
            params.requires_grad = True
        elif 'temporal_av_layer' in name:
            params.requires_grad = True
        # 'temporal_imag_vision_layer' was removed (1 SA per stream now lives
        # in ihsv_temporal_attn). This branch is harmless dead code — it never
        # matches because the parameter no longer exists on the model.
        # elif 'temporal_imag_vision_layer' in name:
        #     params.requires_grad = True
        elif 'task_res' in name:
            params.requires_grad = True
            # learned_parameters.append(params)
        else:
            params.requires_grad = False
    for name, params in mainModel.named_parameters():
        if params.requires_grad:
            print(name)

    # Print fusion CNN trainable parameters (if enabled)
    if fusionCNN is not None:
        print("Fusion CNN trainable parameters:")
        for name, params in fusionCNN.named_parameters():
            print(f"  {name}")
        # Combine parameters from mainModel and fusionCNN
        trainable_params = list(filter(lambda p: p.requires_grad, mainModel.parameters())) + list(fusionCNN.parameters())
    else:
        print("Fusion CNN: DISABLED")
        trainable_params = list(filter(lambda p: p.requires_grad, mainModel.parameters()))
    if phase_projector is not None:
        print("Phase Projector trainable parameters:")
        for name, params in phase_projector.named_parameters():
            print(f"  phase_projector.{name}")
        trainable_params = trainable_params + list(phase_projector.parameters())
    if phase_temporal_attn is not None:
        print("Phase Temporal Self-Attention trainable parameters:")
        for name, params in phase_temporal_attn.named_parameters():
            print(f"  phase_temporal_attn.{name}")
        trainable_params = trainable_params + list(phase_temporal_attn.parameters())
    if ihsv_temporal_attn is not None:
        print("iHSV Temporal Self-Attention trainable parameters:")
        for name, params in ihsv_temporal_attn.named_parameters():
            print(f"  ihsv_temporal_attn.{name}")
        trainable_params = trainable_params + list(ihsv_temporal_attn.parameters())

    optimizer = torch.optim.Adam(trainable_params, lr=args.lr)
    scheduler = MultiStepLR(optimizer, milestones=[10, 20, 30], gamma=0.5)

    """loss"""
    criterion = nn.BCEWithLogitsLoss().cuda()
    criterion_event = nn.CrossEntropyLoss().cuda()
    logger.info(f"==> Classification loss: CrossEntropy")

    '''Resume from a checkpoint'''
    if os.path.isfile(args.resume):
        logger.info(f"\nLoading Checkpoint: {args.resume}\n")
        ckpt = torch.load(args.resume)
        if isinstance(ckpt, dict) and 'mainModel' in ckpt:
            # Backward-compat: older checkpoints predate ModReLU's learnable bias
            # and gated-adjacency loc_adj; load non-strict and let new params keep init.
            missing, unexpected = mainModel.load_state_dict(ckpt['mainModel'], strict=False)
            if missing:
                logger.info(f"  mainModel missing keys (using init): {len(missing)} → e.g. {missing[:3]}")
            if unexpected:
                logger.info(f"  mainModel unexpected keys (ignored):  {len(unexpected)} → e.g. {unexpected[:3]}")
            if fusionCNN is not None and 'fusionCNN' in ckpt:
                missing, unexpected = fusionCNN.load_state_dict(ckpt['fusionCNN'], strict=False)
                if missing:
                    logger.info(f"  fusionCNN missing keys (using init): {missing}")
                if unexpected:
                    logger.info(f"  fusionCNN unexpected keys (ignored): {unexpected}")
                logger.info("  Loaded fusionCNN from checkpoint")
            if phase_temporal_attn is not None and 'phase_temporal_attn' in ckpt:
                phase_temporal_attn.load_state_dict(ckpt['phase_temporal_attn'])
                logger.info("  Loaded phase_temporal_attn from checkpoint")
            if ihsv_temporal_attn is not None and 'ihsv_temporal_attn' in ckpt:
                ihsv_temporal_attn.load_state_dict(ckpt['ihsv_temporal_attn'])
                logger.info("  Loaded ihsv_temporal_attn from checkpoint")
            if phase_projector is not None and 'phase_projector' in ckpt:
                phase_projector.load_state_dict(ckpt['phase_projector'])
                logger.info("  Loaded phase_projector from checkpoint")
        else:
            mainModel.load_state_dict(ckpt)
    elif args.resume != "" and (not os.path.isfile(args.resume)):
        print(f"resume argument: {args.resume}")
        raise FileNotFoundError


    '''Only Evaluate'''
    if args.evaluate:
        logger.info(f"\nStart testing..")
        _score, dump = validate_epoch(mainModel, fusionCNN, test_dataloader, criterion, criterion_event, epoch=0, eval_only=True, use_real_cnn=USE_REAL_CNN,
                       phase_projector=phase_projector, phase_temporal_attn=phase_temporal_attn,
                       cyclegan_generator=cyclegan_generator, cyclegan_if_range=cyclegan_if_range, cyclegan_mel_range=cyclegan_mel_range,
                       ihsv_temporal_attn=ihsv_temporal_attn,
                       collect=True)
        if dump is not None:
            _split = args.test_data_type
            np.savez_compressed(
                os.path.join(args.snapshot_pref, f'preds_{_split}.npz'),
                pred_logits=dump['pred_logits'],
                pred_argmax=dump['pred_argmax'],
                gt_event=dump['gt_event'],
                gt_category=dump['gt_category'],
            )
            logger.info(f"[eval] wrote preds_{_split}.npz to {args.snapshot_pref}")
        return


    best_accuracy, best_accuracy_epoch = 0, 0
    best_dump = None
    '''Training and Evaluation'''
    for epoch in range(args.n_epoch):
        loss = train_epoch(mainModel, fusionCNN, train_dataloader, criterion, criterion_event, optimizer, epoch, use_real_cnn=USE_REAL_CNN,
                           phase_projector=phase_projector, phase_temporal_attn=phase_temporal_attn,
                           cyclegan_generator=cyclegan_generator, cyclegan_if_range=cyclegan_if_range, cyclegan_mel_range=cyclegan_mel_range,
                           ihsv_temporal_attn=ihsv_temporal_attn)

        if ((epoch) % args.eval_freq == 0) or (epoch == args.n_epoch - 1):
            acc, dump = validate_epoch(mainModel, fusionCNN, test_dataloader, criterion, criterion_event, epoch, use_real_cnn=USE_REAL_CNN,
                                 phase_projector=phase_projector, phase_temporal_attn=phase_temporal_attn,
                                 cyclegan_generator=cyclegan_generator, cyclegan_if_range=cyclegan_if_range, cyclegan_mel_range=cyclegan_mel_range,
                                 ihsv_temporal_attn=ihsv_temporal_attn,
                           collect=True)
            if acc > best_accuracy:
                best_accuracy = acc
                best_dump = dump
                logger.info(f'best average result of three metrics at epoch-{epoch}: {best_accuracy:.4f}')
                best_accuracy_epoch = epoch
                checkpoint_dict = {'mainModel': mainModel.state_dict()}
                if fusionCNN is not None:
                    checkpoint_dict['fusionCNN'] = fusionCNN.state_dict()
                if phase_projector is not None:
                    checkpoint_dict['phase_projector'] = phase_projector.state_dict()
                if phase_temporal_attn is not None:
                    checkpoint_dict['phase_temporal_attn'] = phase_temporal_attn.state_dict()
                if ihsv_temporal_attn is not None:
                    checkpoint_dict['ihsv_temporal_attn'] = ihsv_temporal_attn.state_dict()
                save_checkpoint(
                    checkpoint_dict,
                    top1=best_accuracy,
                    task='FullySupervised',
                    epoch=epoch + 1,
                    seed=SEED
                )
        scheduler.step()

    # =====================================================================
    # Post-training analyses on the best-epoch dump (matches ablation_study.py)
    #   - boundary F1 (±1 frame, onset+offset)
    #   - top1-top2 sharpness
    #   - tSNE plot + sharpness histogram
    #   - raw preds dump (.npz)
    # Outputs are written under args.snapshot_pref, which the shell driver
    # sets to .../<row>/sel_<split>/ — so artifacts are per-row, per-split.
    # =====================================================================
    if best_dump is not None:
        run_dir = args.snapshot_pref
        split = args.test_data_type
        title = f"{os.path.basename(os.path.dirname(run_dir.rstrip('/')) or run_dir)} ({split})"

        pred_event = (best_dump['pred_argmax'] != cfg.TOTAL_BG_CLASS_ID).astype(np.int32)
        boundary = compute_boundary_f1(list(pred_event), list(best_dump['gt_event']), tolerance=1)
        logger.info(
            f"[best epoch {best_accuracy_epoch}/{split}] Boundary (±1): "
            f"onset-F1={boundary['onset_f1']:.4f}  offset-F1={boundary['offset_f1']:.4f}  "
            f"combined-F1={boundary['boundary_f1']:.4f}"
        )

        logits_flat = best_dump['pred_logits'].reshape(-1, best_dump['pred_logits'].shape[-1])
        sharp = compute_sharpness(logits_flat)
        logger.info(
            f"[best epoch {best_accuracy_epoch}/{split}] Sharpness (top1-top2): "
            f"mean={sharp['sharpness_mean']:.4f}  median={sharp['sharpness_median']:.4f}"
        )

        tsne_path = os.path.join(run_dir, f'tsne_{split}.png')
        sharp_path = os.path.join(run_dir, f'sharpness_{split}.png')
        ev_flat = best_dump['gt_event'].reshape(-1)
        pc_flat = best_dump['pred_argmax'].reshape(-1)
        run_tsne_plot(logits_flat, pc_flat, ev_flat,
                      best_dump['gt_category'], cfg.TOTAL_BG_CLASS_ID,
                      tsne_path, title)
        run_sharpness_plot(sharp['sharpness_values'], sharp_path, title)

        np.savez_compressed(
            os.path.join(run_dir, f'preds_{split}.npz'),
            pred_logits=best_dump['pred_logits'],
            pred_argmax=best_dump['pred_argmax'],
            gt_event=best_dump['gt_event'],
            gt_category=best_dump['gt_category'],
        )

        with open(os.path.join(run_dir, f'analyses_{split}.json'), 'w') as f:
            json.dump({
                'best_epoch': int(best_accuracy_epoch),
                'best_avg': float(best_accuracy),
                'split': split,
                'boundary': {k: float(v) if isinstance(v, (int, float, np.floating)) else v
                              for k, v in boundary.items()},
                'sharpness_mean': sharp['sharpness_mean'],
                'sharpness_median': sharp['sharpness_median'],
                'tsne_plot': tsne_path,
                'sharpness_plot': sharp_path,
            }, f, indent=2)
    else:
        logger.info("[post-training analyses] skipped — no dump collected "
                    "(validation may not have hit the v1 branch).")


def train_epoch(model, fusion_cnn, train_dataloader, criterion, criterion_event, optimizer, epoch, use_real_cnn=True,
                phase_projector=None, phase_temporal_attn=None,
                cyclegan_generator=None, cyclegan_if_range=None, cyclegan_mel_range=None,
                ihsv_temporal_attn=None):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    train_acc = AverageMeter()
    train_seg_fscore = AverageMeter()
    train_eve_fscore = AverageMeter()
    end_time = time.time()

    model.train()
    if fusion_cnn is not None:
        fusion_cnn.train()
    if phase_projector is not None:
        phase_projector.train()
    if phase_temporal_attn is not None:
        phase_temporal_attn.train()
    if ihsv_temporal_attn is not None:
        ihsv_temporal_attn.train()
    # Note: here we set the model to a double type precision,
    # since the extracted features are in a double type.
    # This will also lead to the size of the model double increases.
    # model.double()
    optimizer.zero_grad()

    for n_iter, batch_data in enumerate(train_dataloader):

        data_time.update(time.time() - end_time)
        '''Feed input to model'''
        if cfg.IHSV.ENABLED and cfg.PHASE.ENABLED:
            audio, visual, visual_imag, text, full_label, avc_label, category_label, vid_name, phase_raw = batch_data
            visual_imag_inputs = visual_imag.cuda()
            phase_raw = phase_raw.squeeze(1).cuda()
        elif cfg.IHSV.ENABLED:
            audio, visual, visual_imag, text, full_label, avc_label, category_label, vid_name = batch_data
            visual_imag_inputs = visual_imag.cuda()
            phase_raw = None
        elif cfg.PHASE.ENABLED:
            audio, visual, text, full_label, avc_label, category_label, vid_name, phase_raw = batch_data
            visual_imag_inputs = None
            phase_raw = phase_raw.squeeze(1).cuda()  # [bs, 10, 1, 257, 204]
        else:
            audio, visual, text, full_label, avc_label, category_label, vid_name = batch_data
            visual_imag_inputs = None
            phase_raw = None
        audio_inputs = audio.squeeze(1).cuda()  # [bs, 10, 1, 128, 204]
        visual_inputs = visual.cuda() # [bs, 10, 3, 224, 224]
        text_inputs = text[0].cuda() # [46+1, 77]
        avc_labels = avc_label.cuda() # [bs, 10] 0 or 1
        full_labels = full_label.cuda() # [bs, 10] category id for CE loss
        category_labels = category_label.squeeze(1).cuda() # [bs]


        """forward"""
        bs = visual_inputs.size(0)
        inputs = {
            ModalityType.TEXT: text_inputs.cuda(), # [46+1, 77]
            ModalityType.VISION: visual_inputs.cuda(), # [bs, 10, 3, 224, 224]
            ModalityType.AUDIO: audio_inputs.cuda(), # [bs, 10, 1, 128, 204]
        }
        if visual_imag_inputs is not None:
            inputs[ModalityType.VISION_IMAG] = visual_imag_inputs  # [bs, 10, 3, 224, 224]
        # with torch.no_grad():
        embeddings = model(inputs) #todo
        audio_feas = embeddings['audio'] # [bs, 10, 1024], without normalization
        visual_feas = embeddings['vision'] # [bs, 10, 1024], without normalization
        text_feas = embeddings['text'] # [46+1, 1024], without normalization
        text_feas = text_feas.unsqueeze(0).repeat(bs, 1, 1) # [bs, 46+1, 1024]
        # pred_logits = embeddings['pred'] # [bs, 10, 46+1], without sigmoid

        #! iHSV: imaginary visual x text similarity
        simm_imag_vt = None
        if ModalityType.VISION_IMAG in embeddings:
            imag_visual_feas = embeddings[ModalityType.VISION_IMAG]  # [bs, 10, 1024]
            # iHSV temporal self-attention (trainable)
            if ihsv_temporal_attn is not None:
                for blk in ihsv_temporal_attn:
                    imag_visual_feas = blk(imag_visual_feas, imag_visual_feas, imag_visual_feas)
            simm_imag_vt = compute_cross_modal_similarity(imag_visual_feas, text_feas)  # [bs, 10, C+1]

        #! =====================================================================
        #! AUDIO IMAGINARY / PHASE FEATURE COMPUTATION
        #! =====================================================================
        simm_pt = None


        if phase_raw is not None and cfg.PHASE.MODE == 'cyclegan':
            #! Vanilla CycleGAN pipeline: phase IF → G_IF2Mel (frozen, unconditional) → mel-like → frozen audio encoder → temporal attn
            B_ph, T_ph = phase_raw.shape[:2]
            phase_flat = phase_raw.reshape(B_ph * T_ph, *phase_raw.shape[2:])  # [bs*10, 1, 257, 204]

            # Step 1: Compute IF (instantaneous frequency)
            phase_if = compute_if(phase_flat)  # [bs*10, 1, 257, 204]

            # Step 2: Translate IF → mel-like via frozen vanilla CycleGAN generator
            with torch.no_grad():
                phase_mel = translate_phase_to_mel(
                    cyclegan_generator, cyclegan_if_range, cyclegan_mel_range, phase_if
                )  # [bs*10, 1, 128, 204]

            # Step 3: Feed through ImageBind's audio encoder (frozen)
            phase_input = {ModalityType.AUDIO: phase_mel.reshape(B_ph, T_ph, 1, 128, 204)}
            with torch.no_grad():
                phase_embeddings = model(phase_input)
            phase_feas = phase_embeddings['audio']  # [bs, 10, 1024]

            # Step 4: Phase temporal self-attention (trainable)
            if phase_temporal_attn is not None:
                for blk in phase_temporal_attn:
                    phase_feas = blk(phase_feas, phase_feas, phase_feas)

            # Step 5: Phase-text similarity
            simm_pt = compute_cross_modal_similarity(phase_feas, text_feas)  # [bs, 10, C+1]

        elif phase_raw is not None and phase_projector is not None:
            #! Original phase pipeline: phase → projector → frozen audio encoder → temporal attn
            B_ph, T_ph = phase_raw.shape[:2]
            phase_flat = phase_raw.reshape(B_ph * T_ph, *phase_raw.shape[2:])  # [bs*10, 1, 257, 204]
            phase_for_proj = compute_if(phase_flat)

            # Step 1: Project phase to mel shape
            phase_mel = phase_projector(phase_for_proj)  # [bs*10, 1, 128, 204]

            # Step 2: Feed through ImageBind's audio encoder (frozen)
            phase_input = {ModalityType.AUDIO: phase_mel.reshape(B_ph, T_ph, 1, 128, 204)}
            with torch.no_grad():
                phase_embeddings = model(phase_input)
            phase_feas = phase_embeddings['audio']  # [bs, 10, 1024]

            # Step 3: Phase temporal self-attention (trainable)
            if phase_temporal_attn is not None:
                for blk in phase_temporal_attn:
                    phase_feas = blk(phase_feas, phase_feas, phase_feas)

            # Step 4: Phase-text similarity
            simm_pt = compute_cross_modal_similarity(phase_feas, text_feas)  # [bs, 10, C+1]

        #! Real-part similarities: visual-text and audio-text
        simm_at = compute_cross_modal_similarity(audio_feas, text_feas) # [bs, 10, 46+1]
        simm_vt = compute_cross_modal_similarity(visual_feas, text_feas) # [bs, 10, 46+1]

        #! =====================================================================
        #! FUSION: RealCNN, ComplexCNN, or simple average (when disabled)
        #! =====================================================================
        # Normalize all available similarity matrices
        simm_vt_norm = min_max_normalize(simm_vt, dim=-1)  # [bs, 10, C+1] -> [0, 1]
        simm_at_norm = min_max_normalize(simm_at, dim=-1)  # [bs, 10, C+1] -> [0, 1]
        simm_imag_vt_norm = min_max_normalize(simm_imag_vt, dim=-1) if simm_imag_vt is not None else None
        simm_pt_norm = min_max_normalize(simm_pt, dim=-1) if simm_pt is not None else None
        simm_vis_imag_norm = simm_imag_vt_norm

        if fusion_cnn is None:
            #! Original-paper baseline (matches fine_tune_baseline_v1_train_fully.py (original OV-AVEL):281-289):
            #!   pred = sqrt(relu(simm_at * simm_vt))   ← multiply RAW similarities
            #!   then per-segment min-max normalize.
            #! Note: when iHSV/PHASE are off (current default), the *_norm variants
            #! aren't used; we go straight from the raw similarities computed above.
            pred_logits = torch.sqrt(F.relu(simm_at * simm_vt))  # [bs, 10, C+1]
            pred_logits = min_max_normalize(pred_logits, dim=-1)
        elif use_real_cnn:
            #! RealCNN: up to 4-channel input (simm_vt, simm_vis_imag, simm_at, simm_pt)
            pred_logits = fusion_cnn(simm_vt_norm, simm_vis_imag_norm, simm_at_norm, simm_pt_norm)  # [bs, 10, C+1]
            pred_logits = min_max_normalize(pred_logits, dim=-1)  # output min-max norm
        else:
            #! ComplexCNN: 2 complex channels (visual / audio domains)
            pred_logits = fusion_cnn(simm_vt_norm, simm_at_norm,
                                     simm_vis_imag_norm, simm_pt_norm)  # [bs, 10, C+1]
            pred_logits = min_max_normalize(pred_logits, dim=-1)  # output min-max norm

        if torch.sum(pred_logits.isnan()) > 0:
            continue

        loss = criterion_event(pred_logits.permute(0, 2, 1), full_labels)


        loss.backward()


        '''Compute Accuracy'''
        if args.test_strategy_type == 'v01':
            # simm_at = compute_cross_modal_similarity(audio_feas, text_feas) # [bs, 10, 46+1]
            # simm_vt = compute_cross_modal_similarity(visual_feas, text_feas) # [bs, 10, 46+1]
            #! DISABLED: Multiply simm_at and simm_vt by AV similarity for consistency
            # simm_at_mod = simm_at * av_weights  # [bs, 10, 46+1]
            # simm_vt_mod = simm_vt * av_weights  # [bs, 10, 46+1]
            is_event_scores, event_scores = postprocess_simm(simm_at, simm_vt, bg_flag=cfg.TRAIN_BG_CLASS_ID) # [bs, 10], [bs, 46+1]
            acc = compute_accuracy_supervised_v0(is_event_scores, event_scores, avc_labels, category_labels, bg_flag=cfg.TRAIN_BG_CLASS_ID)
            train_acc.update(acc.item(), bs * 10)
            # pdb.set_trace()
            seg_f, eve_f = compute_seg_eve_fscores_v0(is_event_scores, event_scores, avc_labels, category_labels, bg_flag=cfg.TRAIN_BG_CLASS_ID)
            train_seg_fscore.update(seg_f, n=1)
            train_eve_fscore.update(eve_f, n=1)
        elif args.test_strategy_type == 'v0':
            simm_at_v0 = simm_at[:, :, :-1] # [bs, 10, 46]
            simm_vt_v0 = simm_vt[:, :, :-1] # [bs, 10, 46]
            #! DISABLED: Multiply by AV similarity
            # simm_at_v0 = simm_at_v0 * av_weights  # [bs, 10, 46]
            # simm_vt_v0 = simm_vt_v0 * av_weights  # [bs, 10, 46]
            is_event_scores, event_scores = postprocess_simm(simm_at_v0, simm_vt_v0, bg_flag=cfg.TRAIN_BG_CLASS_ID) # [bs, 10], [bs, 46+1]
            acc = compute_accuracy_supervised_v0(is_event_scores, event_scores, avc_labels, category_labels, bg_flag=cfg.TRAIN_BG_CLASS_ID)
            train_acc.update(acc.item(), bs * 10)
            # pdb.set_trace()
            seg_f, eve_f = compute_seg_eve_fscores_v0(is_event_scores, event_scores, avc_labels, category_labels, bg_flag=cfg.TRAIN_BG_CLASS_ID)
            train_seg_fscore.update(seg_f, n=1)
            train_eve_fscore.update(eve_f, n=1)
        elif args.test_strategy_type == 'v1':
            acc = compute_accuracy_supervised_v1(pred_logits, full_labels, bg_flag=cfg.TRAIN_BG_CLASS_ID)
            train_acc.update(acc.item(), bs*10)
            seg_f, eve_f = compute_seg_eve_fscores_v1(pred_logits, avc_labels, category_labels, bg_flag=cfg.TRAIN_BG_CLASS_ID)
            train_seg_fscore.update(seg_f, n=1)
            train_eve_fscore.update(eve_f, n=1)

        '''Clip Gradient'''
        if args.clip_gradient is not None:
            # total_norm = clip_grad_norm_(model.parameters(), args.clip_gradient)
            total_norm = clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()), args.clip_gradient)
            # if total_norm > args.clip_gradient:
            #     logger.info(f'Clipping gradient: {total_norm} with coef {args.clip_gradient/total_norm}.')

        '''Update parameters'''
        optimizer.step()
        optimizer.zero_grad()


        losses.update(loss.item(), bs*10)
        batch_time.update(time.time() - end_time)
        end_time = time.time()


        '''Print logs in Terminal'''
        if n_iter % args.print_iter_freq == 0:
            logger.info(
                f'Train Epoch: [{epoch}][{n_iter}/{len(train_dataloader)}]\t'
                # f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                # f'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                f'Loss {losses.val:.4f} ({losses.avg:.4f})\t'
                f'Prec@1 {train_acc.val:.3f} ({train_acc.avg:.3f})\t'
                f'Seg@F1 {train_seg_fscore.val:.3f} ({train_seg_fscore.avg:.3f})\t'
                f'Eve@F1 {train_eve_fscore.val:.3f} ({train_eve_fscore.avg:.3f})'
            )

    return losses.avg



def compute_cross_modal_similarity(tensor_a, tensor_t):
    B, T, D = tensor_a.shape
    _, C, _ = tensor_t.shape
    tensor_a_expanded = tensor_a.unsqueeze(2).expand(B, T, C, D)
    tensor_t_expanded = tensor_t.unsqueeze(1).expand(B, T, C, D)
    cos_sim = F.cosine_similarity(tensor_a_expanded, tensor_t_expanded, dim=-1)

    # norm_a = F.normalize(tensor_a, dim=-1)
    # norm_t = F.normalize(tensor_t, dim=-1)
    # simm_at = torch.bmm(norm_a, norm_t.permute(0, 2, 1))
    # pdb.set_trace()
    return cos_sim

def postprocess_simm(simm_at, simm_vt, bg_flag=cfg.TOTAL_BG_CLASS_ID):
    # simm_at, simm_vt: [B, T, C=67 / 67+1]
    max_prob_at_idx = simm_at.max(dim=-1)[1] # [B, T]
    max_prob_vt_idx = simm_vt.max(dim=-1)[1] # [B, T]
    is_event_flag = (max_prob_at_idx == max_prob_vt_idx).float() # [B, T]
    # is_event_flag = (simm_at.max(dim=-1)[1] == simm_vt.max(dim=-1)[1]).float() # [B, T]
    B = is_event_flag.shape[0]
    # C = simm_at.shape[-1]
    event_flag = torch.zeros([B, bg_flag+1]).cuda()
    # print(is_event_flag)
    for i in range(B):
        if torch.all(is_event_flag[i] == 0):
            # event_flag[i][cfg.TOTAL_BG_CLASS_ID] = 1 # background video, all segments are backgrounds
            event_flag[i][bg_flag] = 1 # background video, all segments are backgrounds
        else:
            nonzero_pos = torch.nonzero(is_event_flag[i], as_tuple=False).squeeze()
            if len(nonzero_pos.size()) == 0: # only one postive position
                category_id = max_prob_at_idx[i][nonzero_pos.item()]
            else:
                category_id = max_prob_at_idx[i][nonzero_pos[0].item()]
            event_flag[i][category_id] = 1
    # pdb.set_trace()
    return is_event_flag, event_flag


@torch.no_grad()
def validate_epoch(model, fusion_cnn, val_dataloader, criterion, criterion_event, epoch, eval_only=False, use_real_cnn=True,
                   phase_projector=None, phase_temporal_attn=None,
                   cyclegan_generator=None, cyclegan_if_range=None, cyclegan_mel_range=None,
                   ihsv_temporal_attn=None,
                   collect=False):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    accuracy = AverageMeter()
    seg_fscore = AverageMeter()
    eve_fscore = AverageMeter()
    end_time = time.time()

    # Per-batch collection for post-training analyses (boundary F1, tSNE, sharpness)
    coll_logits, coll_argmax, coll_gt_event, coll_gt_category, coll_full_label = [], [], [], [], []

    # Diagnostic: per-segment confusion / bg-fg / boundary-interior breakdown
    diag = Diagnoser(num_classes=cfg.TOTAL_BG_CLASS_ID + 1, bg_flag=cfg.TOTAL_BG_CLASS_ID)

    model.eval()
    if fusion_cnn is not None:
        fusion_cnn.eval()
    if phase_projector is not None:
        phase_projector.eval()
    if phase_temporal_attn is not None:
        phase_temporal_attn.eval()

    for n_iter, batch_data in enumerate(val_dataloader):
        data_time.update(time.time() - end_time)

        '''Feed input to model'''
        if cfg.IHSV.ENABLED and cfg.PHASE.ENABLED:
            audio, visual, visual_imag, text, full_label, avc_label, category_label, vid_name, phase_raw = batch_data
            visual_imag_inputs = visual_imag.cuda()
            phase_raw = phase_raw.squeeze(1).cuda()
        elif cfg.IHSV.ENABLED:
            audio, visual, visual_imag, text, full_label, avc_label, category_label, vid_name = batch_data
            visual_imag_inputs = visual_imag.cuda()
            phase_raw = None
        elif cfg.PHASE.ENABLED:
            audio, visual, text, full_label, avc_label, category_label, vid_name, phase_raw = batch_data
            visual_imag_inputs = None
            phase_raw = phase_raw.squeeze(1).cuda()
        else:
            audio, visual, text, full_label, avc_label, category_label, vid_name = batch_data
            visual_imag_inputs = None
            phase_raw = None
        audio_inputs = audio.squeeze(1).cuda()  # [bs, 10, 1, 128, 204]
        visual_inputs = visual.cuda() # [bs, 10, 3, 224, 224]
        text_inputs = text[0].cuda() # [46/67, 77]
        avc_labels = avc_label.cuda() # [bs, 10], 0 or 1
        category_labels = category_label.squeeze(1).cuda() # [bs]
        full_labels = full_label.cuda() # [bs, 10], class ids


        bs = visual_inputs.size(0)
        inputs = {
            ModalityType.TEXT: text_inputs.cuda(), # [67+1, 77]
            ModalityType.VISION: visual_inputs.cuda(), # [bs, 10, 3, 224, 224]
            ModalityType.AUDIO: audio_inputs.cuda(), # [bs, 10, 1, 128, 204]
        }
        if visual_imag_inputs is not None:
            inputs[ModalityType.VISION_IMAG] = visual_imag_inputs
        with torch.no_grad():
            embeddings = model(inputs)
        audio_feas = embeddings['audio'] # [bs, 10, 1024]
        visual_feas = embeddings['vision'] # [bs, 10, 1024]
        text_feas = embeddings['text'] # [67+1, 1024]
        text_feas = text_feas.unsqueeze(0).repeat(bs, 1, 1) # [bs, 67+1, 1024]
        # pred_logits = embeddings['pred'] # [bs, 10, 67+1]

        #! =====================================================================
        #! AUDIO IMAGINARY / PHASE FEATURE COMPUTATION — same as train_epoch
        #! =====================================================================
        simm_pt = None


        if phase_raw is not None and cfg.PHASE.MODE == 'cyclegan':
            #! Vanilla CycleGAN pipeline (eval): unconditional IF → mel translation
            B_ph, T_ph = phase_raw.shape[:2]
            phase_flat = phase_raw.reshape(B_ph * T_ph, *phase_raw.shape[2:])
            phase_if = compute_if(phase_flat)
            with torch.no_grad():
                phase_mel = translate_phase_to_mel(
                    cyclegan_generator, cyclegan_if_range, cyclegan_mel_range, phase_if
                )
                phase_input = {ModalityType.AUDIO: phase_mel.reshape(B_ph, T_ph, 1, 128, 204)}
                phase_embeddings = model(phase_input)
                phase_feas = phase_embeddings['audio']
                if phase_temporal_attn is not None:
                    for blk in phase_temporal_attn:
                        phase_feas = blk(phase_feas, phase_feas, phase_feas)
                simm_pt = compute_cross_modal_similarity(phase_feas, text_feas)

        elif phase_raw is not None and phase_projector is not None:
            B_ph, T_ph = phase_raw.shape[:2]
            phase_flat = phase_raw.reshape(B_ph * T_ph, *phase_raw.shape[2:])
            phase_for_proj = compute_if(phase_flat)
            with torch.no_grad():
                phase_mel = phase_projector(phase_for_proj)
                phase_input = {ModalityType.AUDIO: phase_mel.reshape(B_ph, T_ph, 1, 128, 204)}
                phase_embeddings = model(phase_input)
                phase_feas = phase_embeddings['audio']
                if phase_temporal_attn is not None:
                    for blk in phase_temporal_attn:
                        phase_feas = blk(phase_feas, phase_feas, phase_feas)
                simm_pt = compute_cross_modal_similarity(phase_feas, text_feas)

        #! iHSV / complex imag visual x text similarity
        simm_imag_vt = None
        if ModalityType.VISION_IMAG in embeddings:
            imag_visual_feas = embeddings[ModalityType.VISION_IMAG]  # [bs, 10, 1024]
            # iHSV temporal self-attention (trainable)
            if ihsv_temporal_attn is not None:
                for blk in ihsv_temporal_attn:
                    imag_visual_feas = blk(imag_visual_feas, imag_visual_feas, imag_visual_feas)
            simm_imag_vt = compute_cross_modal_similarity(imag_visual_feas, text_feas)  # [bs, 10, C+1]

        #! Real-part similarities: visual-text and audio-text
        simm_at = compute_cross_modal_similarity(audio_feas, text_feas) # [bs, 10, 67+1]
        simm_vt = compute_cross_modal_similarity(visual_feas, text_feas) # [bs, 10, 67+1]

        #! =====================================================================
        #! FUSION: RealCNN, ComplexCNN, or simple average (when disabled)
        #! =====================================================================
        # Normalize all available similarity matrices
        simm_vt_norm = min_max_normalize(simm_vt, dim=-1)
        simm_at_norm = min_max_normalize(simm_at, dim=-1)
        simm_imag_vt_norm = min_max_normalize(simm_imag_vt, dim=-1) if simm_imag_vt is not None else None
        simm_pt_norm = min_max_normalize(simm_pt, dim=-1) if simm_pt is not None else None
        simm_vis_imag_norm = simm_imag_vt_norm

        if fusion_cnn is None:
            #! Original-paper baseline: sqrt(relu(simm_at * simm_vt)) on RAW similarities,
            #! then per-segment min-max. Matches fine_tune_baseline_v1_train_fully.py (original OV-AVEL):281-289.
            pred_logits = torch.sqrt(F.relu(simm_at * simm_vt))  # [bs, 10, C+1]
            pred_logits = min_max_normalize(pred_logits, dim=-1)
        elif use_real_cnn:
            #! RealCNN: up to 4-channel input (simm_vt, simm_vis_imag, simm_at, simm_pt)
            pred_logits = fusion_cnn(simm_vt_norm, simm_vis_imag_norm, simm_at_norm, simm_pt_norm)  # [bs, 10, C+1]
            pred_logits = min_max_normalize(pred_logits, dim=-1)  # output min-max norm
        else:
            #! ComplexCNN: 2 complex channels (visual / audio domains)
            pred_logits = fusion_cnn(simm_vt_norm, simm_at_norm,
                                     simm_vis_imag_norm, simm_pt_norm)  # [bs, 10, C+1]
            pred_logits = min_max_normalize(pred_logits, dim=-1)  # output min-max norm

        #! =====================================================================
        # baseline v0: training free: computing audio-text visual-text similarity
        if args.test_strategy_type != 'v1':
            if args.test_strategy_type == 'v0':
                text_feas_v0 = text_feas[:, :-1, :] # [bs, 67, 1024]
                # pdb.set_trace()
                simm_at_v0 = compute_cross_modal_similarity(audio_feas, text_feas_v0) # [bs, 10, 67]
                simm_vt_v0 = compute_cross_modal_similarity(visual_feas, text_feas_v0) # [bs, 10, 67]
                #! DISABLED: Multiply by AV similarity
                # simm_at_v0 = simm_at_v0 * av_weights  # [bs, 10, 67]
                # simm_vt_v0 = simm_vt_v0 * av_weights  # [bs, 10, 67]
                # pdb.set_trace()
                is_event_scores, event_scores = postprocess_simm(simm_at_v0, simm_vt_v0, bg_flag=cfg.TOTAL_BG_CLASS_ID) # [bs, 10], [bs, 67+1]
                # is_event_scores: avc predction;
                # event_scores: event category prediction
            # baseline v01: training free: using `other' category, computing audio-text visual-text similarity"""
            elif args.test_strategy_type == 'v01':
                # pdb.set_trace()
                # simm_at = compute_cross_modal_similarity(audio_feas, text_feas) # [bs, 10, 67+1]
                # simm_vt = compute_cross_modal_similarity(visual_feas, text_feas) # [bs, 10, 67+1]
                #! DISABLED: Multiply by AV similarity
                # simm_at_mod = simm_at * av_weights  # [bs, 10, 67+1]
                # simm_vt_mod = simm_vt * av_weights  # [bs, 10, 67+1]
                is_event_scores, event_scores = postprocess_simm(simm_at, simm_vt, bg_flag=cfg.TOTAL_BG_CLASS_ID) # [bs, 10], [bs, 67+1]
            loss_is_event = criterion_event(is_event_scores, avc_labels)
            loss_event_class = criterion_event(event_scores, category_labels)
            loss = loss_is_event + loss_event_class

            # print(vid_name)
            acc = compute_accuracy_supervised_v0(is_event_scores, event_scores, avc_labels, category_labels)
            accuracy.update(acc.item(), bs * 10)
            # pdb.set_trace()
            seg_f, eve_f = compute_seg_eve_fscores_v0(is_event_scores, event_scores, avc_labels, category_labels, bg_flag=cfg.TOTAL_BG_CLASS_ID)
            seg_fscore.update(seg_f, n=1)
            eve_fscore.update(eve_f, n=1)
        else:
            # breakpoint()
            loss = criterion_event(pred_logits.permute(0, 2, 1), full_labels)

            pred_logits_eval = pred_logits

            acc = compute_accuracy_supervised_v1(pred_logits_eval, full_labels, bg_flag=cfg.TOTAL_BG_CLASS_ID)
            accuracy.update(acc.item(), bs * 10)
            diag.update(pred_logits_eval.argmax(-1), full_labels)
            seg_f, eve_f = compute_seg_eve_fscores_v1(pred_logits_eval, avc_labels, category_labels, bg_flag=cfg.TOTAL_BG_CLASS_ID)
            seg_fscore.update(seg_f, n=1)
            eve_fscore.update(eve_f, n=1)

            if collect:
                argmax = pred_logits_eval.argmax(dim=-1)  # [B, T]
                coll_logits.append(pred_logits.detach().cpu().numpy())  # raw logits preserved for downstream analysis
                coll_argmax.append(argmax.detach().cpu().numpy())
                coll_gt_event.append(avc_labels.detach().cpu().numpy())
                coll_gt_category.append(category_labels.detach().cpu().numpy())
                coll_full_label.append(full_labels.detach().cpu().numpy())

        batch_time.update(time.time() - end_time)
        end_time = time.time()
        losses.update(loss.item(), bs * 10)

        '''Print logs in Terminal'''
        # if n_iter % args.print_iter_freq == 0:
        if n_iter % args.print_iter_freq == 0:
            logger.info(
                f'Test Epoch [{epoch}][{n_iter}/{len(val_dataloader)}]\t'
                # f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                # f'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                f'Loss {losses.val:.4f} ({losses.avg:.4f})\t'
                f'Prec@1 {accuracy.val:.3f} ({accuracy.avg:.3f})\t'
                f'Seg@F1 {seg_fscore.val:.3f} ({seg_fscore.avg:.3f})\t'
                f'Eve@F1 {eve_fscore.val:.3f} ({eve_fscore.avg:.3f})'
            )

    logger.info(
        f"\tEvaluation results (acc): {accuracy.avg:.4f}\t (Segment-level F1score): {seg_fscore.avg:.4f}\t (Event-level F1score): {eve_fscore.avg:.4f}"
    )

    logger.info(f"--- Diagnostic dump (epoch {epoch}) ---")
    diag.report(logger=logger)

    score = (accuracy.avg + seg_fscore.avg + eve_fscore.avg) / 3
    if collect:
        dump = None
        if len(coll_logits) > 0:
            dump = {
                'pred_logits': np.concatenate(coll_logits, axis=0),       # [N,T,C+1]
                'pred_argmax': np.concatenate(coll_argmax, axis=0),       # [N,T]
                'gt_event':    np.concatenate(coll_gt_event, axis=0),     # [N,T]
                'gt_category': np.concatenate(coll_gt_category, axis=0),  # [N]
                'full_label':  np.concatenate(coll_full_label, axis=0),   # [N,T]
            }
        return score, dump
    return score




def compute_accuracy_supervised_v0(is_event_scores_pred, event_scores_pred, avc_labels, category_labels, bg_flag=cfg.TOTAL_BG_CLASS_ID):
    # is_event_scores_pred, avc_labels: [bs, 10]
    # event_scores_pred: [bs, C+1]
    # category_labels: [bs]
    def obtain_label_mat(is_event_scores, event_scores, bg_flag):
        scores_mask = (is_event_scores == 0)
        _, event_class = event_scores.max(-1) # foreground classification, [B]
        pred = is_event_scores.long() # [B, 10]
        pred *= event_class[:, None]
        # add mask
        pred[scores_mask] = bg_flag # 67 denotes bg
        return pred

    pred = obtain_label_mat(is_event_scores_pred, event_scores_pred, bg_flag)
    is_event_scores_gt = avc_labels
    event_scores_gt = torch.zeros_like(event_scores_pred) # [bs, C+1]
    for i in range(event_scores_gt.size(0)):
        event_scores_gt[i, category_labels[i]] = 1

    targets = obtain_label_mat(is_event_scores_gt, event_scores_gt, bg_flag)
    correct = pred.eq(targets)
    correct_num = correct.sum().double()
    # acc = correct_num * (100. / correct.numel())
    acc = correct_num / correct.numel()
    # pdb.set_trace()
    return acc

# def compute_accuracy_supervised(is_event_scores_pred, event_scores_pred, avc_labels, category_labels, bg_flag=cfg.TOTAL_BG_CLASS_ID):
def compute_accuracy_supervised_v1(pred_logits, full_labels, bg_flag):
    # pred_logits: [bs, 10, 46/67 + 1]
    # full_labels: [bs, 10], class ids
    _, pred = pred_logits.max(-1)
    targets = full_labels
    correct = pred.eq(targets)
    correct_num = correct.sum().double()
    # acc = correct_num * (100. / correct.numel())
    acc = correct_num / correct.numel()
    # pdb.set_trace()
    return acc


def compute_seg_eve_fscores_v0(is_event_scores_pred, event_scores_pred, avc_labels, category_labels, bg_flag=cfg.TOTAL_BG_CLASS_ID):
    # is_event_scores_pred, avc_labels: [bs, 10]
    # event_scores_pred: [bs, C+1]
    # category_labels: [bs]
    def obtain_pred_mat(is_event_scores, event_scores, bg_flag):
        B, T = is_event_scores.shape
        K = event_scores.size(-1) # C+1
        assert K == bg_flag + 1
        # pdb.set_trace()
        pred_mat = torch.zeros(B, K, T).cuda()
        for i in range(B):
            class_id = torch.nonzero(event_scores[i]).squeeze().item()
            if class_id != bg_flag:
                pred_mat[i][class_id] = is_event_scores[i]
                pred_mat[i][-1] = 1 - is_event_scores[i] #  background
            else:
                pred_mat[i][-1] = 1 - is_event_scores[i] #  background
            # pdb.set_trace()
        return pred_mat.cpu().data.numpy()

    def obtain_gt_mat(avc_labels, category_labels, bg_flag):
        # avc_labels: [bs, 10], category_labels: [bs]
        B, T = avc_labels.shape
        K = bg_flag + 1
        gt_mat = torch.zeros(B, K, T).cuda()
        for i in range(B):
            class_id = category_labels[i].item()
            if class_id != bg_flag:
                gt_mat[i][class_id] = avc_labels[i]
                gt_mat[i][-1] = 1 - avc_labels[i]
            else:
                gt_mat[i][-1] = 1 - avc_labels[i]
            # pdb.set_trace()
        return gt_mat.cpu().data.numpy()

    pred = obtain_pred_mat(is_event_scores_pred, event_scores_pred, bg_flag)
    targets = obtain_gt_mat(avc_labels, category_labels, bg_flag)
    # pdb.set_trace()

    B = avc_labels.shape[0]
    seg_fscore = np.zeros(B)
    eve_fscore = np.zeros(B)
    for i in range(B):
        seg_f = segment_level(pred[i], targets[i])
        seg_fscore[i] = seg_f
        eve_f = event_level(pred[i], targets[i])
        eve_fscore[i] = eve_f
    avg_batch_seg_fscore = np.mean(seg_fscore)
    avg_batch_eve_fscore = np.mean(eve_fscore)
    # pdb.set_trace()
    return avg_batch_seg_fscore, avg_batch_eve_fscore



def compute_seg_eve_fscores_v1(pred_logits, avc_labels, category_labels, bg_flag=cfg.TOTAL_BG_CLASS_ID):
    # pred_logits: [bs, 10, 67+1]
    # avc_labels: [bs, 10], 0 or 1
    # category_labels: [bs]
    def obtain_pred_mat(pred_logits, bg_flag):
        pred_logits = pred_logits.permute(0, 2, 1) # [bs, 67+1, 10]
        assert pred_logits.shape[1] == bg_flag + 1
        pred_mat = torch.zeros_like(pred_logits).cuda() # [bs, 67+1, 10]
        pred_cls_ids = pred_logits.max(dim=1)[1] # [bs, 10]
        B = pred_logits.shape[0]
        for i in range(B):
            class_id = pred_cls_ids[i] # [10]
            for j in range(len(class_id)):
                pred_mat[i][class_id[j]][j] = 1
        return pred_mat.cpu().data.numpy()

    def obtain_gt_mat(avc_labels, category_labels, bg_flag):
        # avc_labels: [bs, 10], category_labels: [bs]
        B, T = avc_labels.shape
        K = bg_flag + 1
        gt_mat = torch.zeros(B, K, T).cuda()
        for i in range(B):
            class_id = category_labels[i].item()
            if class_id != bg_flag:
                gt_mat[i][class_id] = avc_labels[i]
                gt_mat[i][-1] = 1 - avc_labels[i]
            else:
                gt_mat[i][-1] = 1 - avc_labels[i]
            # pdb.set_trace()
        return gt_mat.cpu().data.numpy()

    pred = obtain_pred_mat(pred_logits, bg_flag) # [bs, 46/67 + 1, 10]
    targets = obtain_gt_mat(avc_labels, category_labels, bg_flag) # [bs, 46/67 + 1, 10]

    B = avc_labels.shape[0]
    seg_fscore = np.zeros(B)
    eve_fscore = np.zeros(B)
    for i in range(B):
        seg_f = segment_level(pred[i], targets[i])
        seg_fscore[i] = seg_f
        eve_f = event_level(pred[i], targets[i])
        eve_fscore[i] = eve_f
    avg_batch_seg_fscore = np.mean(seg_fscore)
    avg_batch_eve_fscore = np.mean(eve_fscore)
    # pdb.set_trace()
    return avg_batch_seg_fscore, avg_batch_eve_fscore



def save_checkpoint(state_dict, top1, task, epoch, seed):
    # model_name = f'{args.snapshot_pref}/model_seed_{seed}_epoch_{epoch}_top1_{top1:.3f}_task_{task}_best_model.pth.tar'
    model_name = f'{args.snapshot_pref}/task_{task}_best_model.pth.tar'
    torch.save(state_dict, model_name)
    print("best model is saved to ", model_name)


if __name__ == '__main__':
    cur = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
    print(f'current time: {cur}')
    start = time.time()

    main()

    end = time.time()
    print(f'duration time {(end - start) / 60} mins.')
    cur = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
    print(f'current time: {cur}')
