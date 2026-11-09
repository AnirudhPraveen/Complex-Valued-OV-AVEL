import os
import time
import random
import json
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import StepLR, MultiStepLR


import numpy as np
import pandas as pd
from configs.opts import parser
from utils import AverageMeter, Prepare_logger, get_and_save_args


# ============================================================================
# Analyses: boundary F1, tSNE, sharpness
# (Copied verbatim from ablation_study.py so the 8-way produces the same
#  artifacts as the 6-way — boundary-F1 (±1 frame, onset+offset), tSNE plot,
#  top1-top2 sharpness histogram, and raw preds dump — per split.)
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
# Multi-class Focal Loss (drop-in replacement for nn.CrossEntropyLoss)
#   FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
#   gamma=0, alpha=1  -> standard CE
# Accepts logits shaped (B, C, T) and targets (B, T) — matches the existing
# call sites: criterion_event(pred_logits.permute(0,2,1), full_labels).
# ============================================================================
# class MulticlassFocalLoss(nn.Module):
#     def __init__(self, gamma=2.0, alpha=None, ignore_index=-100, reduction='mean'):
#         super().__init__()
#         self.gamma = gamma
#         self.ignore_index = ignore_index
#         self.reduction = reduction
#         if alpha is None:
#             self.alpha = None
#         else:
#             self.register_buffer('alpha', torch.as_tensor(alpha, dtype=torch.float32),
#                                  persistent=False)
#
#     def forward(self, logits, target):
#         # logits: (B, C, ...), target: (B, ...) with class indices in [0, C)
#         logp = F.log_softmax(logits, dim=1)
#         valid = target != self.ignore_index
#         target_clamped = target.clamp_min(0)
#         logp_t = logp.gather(1, target_clamped.unsqueeze(1)).squeeze(1)
#         p_t = logp_t.exp()
#         loss = -((1.0 - p_t) ** self.gamma) * logp_t
#         if self.alpha is not None:
#             a = self.alpha.to(logits.device)[target_clamped]
#             loss = a * loss
#         loss = loss * valid.float()
#         if self.reduction == 'sum':
#             return loss.sum()
#         if self.reduction == 'none':
#             return loss
#         denom = valid.float().sum().clamp_min(1.0)
#         return loss.sum() / denom


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
    ReLU activation for complex values.
    Applies ReLU separately to real and imaginary parts.
    WARNING: Destroys phase information. Use ComplexModReLU instead.
    """
    def forward(self, real, imag):
        return F.relu(real), F.relu(imag)


class ComplexModReLU(nn.Module):
    """
    modReLU activation for complex values (Arjovsky et al., 2016).

    modReLU(z) = ReLU(|z| + b) * z / |z|

    Thresholds magnitude while preserving phase — the correct nonlinearity
    for complex-valued networks. The learnable bias b creates a dead zone
    (when b < 0) that filters out low-magnitude (uncertain) signals.

    Reference: Arjovsky, Shah & Bengio, "Unitary Evolution Recurrent Neural
    Networks", ICML 2016. Also used in Trabelsi et al., "Deep Complex
    Networks", ICLR 2018.
    """
    def __init__(self, num_features, bias_init=-0.1):
        super().__init__()
        # Learnable bias per channel — initialized negative to create dead zone
        self.bias = nn.Parameter(torch.full((1, num_features, 1, 1), bias_init))

    def forward(self, real, imag):
        # |z| = sqrt(real^2 + imag^2 + eps)
        magnitude = torch.sqrt(real ** 2 + imag ** 2 + 1e-8)
        # ReLU(|z| + b) — thresholds magnitude
        scaled_mag = F.relu(magnitude + self.bias)
        # z / |z| — unit complex number (phase)
        phase_real = real / magnitude
        phase_imag = imag / magnitude
        # modReLU(z) = ReLU(|z| + b) * z/|z|
        return scaled_mag * phase_real, scaled_mag * phase_imag


class ComplexCNN(nn.Module):
    """
    5-layer Complex-Valued CNN for multi-modal fusion.

    Dual-complex mode (in_channels=2, when iHSV + phase enabled):
        - Channel 0: z₁ = simm_vt + i·simm_at       (content: what is seen/heard)
        - Channel 1: z₂ = simm_imag_vt + i·simm_pt   (structure: color + phase)

    Single-complex mode (in_channels=1, original):
        - Channel 0: z = simm_vt + i·simm_at

    Output:
        - magnitude of complex output [bs, 10, C+1]

    Note: Uses (temporal_kernel, 1) kernel to capture temporal patterns while
    remaining category-agnostic (important for open-vocabulary setting where
    train has 47 categories but test has 68 categories).
    """
    def __init__(self, hidden_channels=32, temporal_kernel=5, in_channels=1,
                 activation='crelu', modrelu_bias_init=-0.1):
        """
        activation:
            'crelu'    = ComplexReLU (default; F.relu on real and imag separately;
                         destroys phase — restricts hidden phases to [0, π/2]).
            'modrelu'  = ComplexModReLU (Arjovsky 2016; ReLU(|z|+b)·z/|z|; phase-
                         preserving; lets phase structure build across all 5 layers).
        modrelu_bias_init: initial bias `b` for each ModReLU layer (only used
            when activation='modrelu'). Negative creates a dead zone for low-
            magnitude / uncertain features.
        """
        super(ComplexCNN, self).__init__()
        assert activation in ('crelu', 'modrelu'), \
            f"activation must be 'crelu' or 'modrelu', got {activation!r}"
        self.activation_name = activation

        t_pad = temporal_kernel // 2  # padding to maintain temporal dimension (10 -> 10)

        def make_act(num_features):
            if activation == 'modrelu':
                return ComplexModReLU(num_features, bias_init=modrelu_bias_init)
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
        Channel pairing is selected by cfg.FUSION_CHANNEL_MODE:
          'complex_pair'      (default): ch0 = simm_vt + i·simm_imag_vt   (per-domain visual)
                                         ch1 = simm_at + i·simm_pt        (per-domain audio)
          'content_structure': ch0 = simm_at + i·simm_vt                  (content)
                               ch1 = simm_imag_vt + i·simm_pt             (structure)
          'modality':          ch0 = simm_at + i·simm_pt                  (audio + phase)
                               ch1 = simm_vt + i·simm_imag_vt             (visual + iHSV)
        Single-channel fallback (no iHSV/phase): ch0 = simm_vt + i·simm_at.
        """
        from config import cfg as _cfg
        mode = getattr(_cfg, 'FUSION_CHANNEL_MODE', 'complex_pair')
        # RUN-1 swap ablation: swap real<->imag of a channel by swapping the
        # SEMANTIC inputs BEFORE mode packing (mode-independent, identical in
        # train & eval). 'none'|'visual'|'audio'|'both'. See config.SWAP_REIM.
        swap = getattr(_cfg, 'SWAP_REIM', 'none')

        has_extra = (simm_imag_vt is not None) or (simm_pt is not None)

        if not has_extra:
            # 2-stream single channel z = S_vt + i·S_at. Any non-'none' swap
            # flips it to z = S_at + i·S_vt (there is only one channel here).
            if swap != 'none':
                real = simm_at.unsqueeze(1)
                imag = simm_vt.unsqueeze(1)
            else:
                real = simm_vt.unsqueeze(1)
                imag = simm_at.unsqueeze(1)
        else:
            zero_like = torch.zeros_like(simm_vt)
            iv = simm_imag_vt if simm_imag_vt is not None else zero_like
            pt = simm_pt      if simm_pt      is not None else zero_like

            # Swap re<->im per channel on the semantic inputs (visual = simm_vt
            # real / iv imag ; audio = simm_at real / pt imag) before packing.
            if swap in ('visual', 'both'):
                simm_vt, iv = iv, simm_vt
            if swap in ('audio', 'both'):
                simm_at, pt = pt, simm_at

            if mode == 'complex_pair':
                # ch0: visual real + visual imag ; ch1: audio real + audio imag
                real_ch0, imag_ch0 = simm_vt, iv
                real_ch1, imag_ch1 = simm_at, pt
            elif mode == 'modality':
                # ch0: audio + phase ; ch1: visual + iHSV
                real_ch0, imag_ch0 = simm_at, pt
                real_ch1, imag_ch1 = simm_vt, iv
            else:  # 'content_structure'
                real_ch0, imag_ch0 = simm_at, simm_vt
                real_ch1, imag_ch1 = iv, pt

            real = torch.stack([real_ch0, real_ch1], dim=1)
            imag = torch.stack([imag_ch0, imag_ch1], dim=1)

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
# FCCN auxiliary phase loss (3 modes, selected by cfg.FCCN_LOSS.MODE).
# Operates on the complex output of ComplexCNN; combines with CE-on-magnitude.
# ============================================================================
def apply_fccn_aux_loss(loss_mag, fusion_phase, full_labels, cfg_obj):
    """Combine the CE-on-magnitude base loss with the FCCN auxiliary phase term.

    fusion_phase: [B, T, C+1]  (atan2(imag, real), ∈ [-π, π])
    full_labels:  [B, T]       (long, integer class index)

    Returns: scalar loss tensor according to cfg_obj.FCCN_LOSS.MODE.
        'fccn'      → (CE - λ·mean θ[y]) / 2          (original FCCN)
        'cosine'    → CE + λ·mean(1 - cos θ[y])       (circular-aware)
        'coherence' → CE + λ·(1 - mean_b |R̄_b|)      (temporal coherence
                       within event segments per clip; |R̄| = circular
                       resultant of θ[y] across FG time-steps)
    Falls back to loss_mag unchanged when ENABLED=False or fusion_phase is None.
    """
    fl = getattr(cfg_obj, 'FCCN_LOSS', None)
    if fl is None or not fl.ENABLED or fusion_phase is None:
        return loss_mag

    mode = getattr(fl, 'MODE', 'fccn')
    lam = float(fl.LAMBDA)
    bg = cfg_obj.TRAIN_BG_CLASS_ID

    # Pick θ at the true class for every (b, t): [B, T].
    y_idx = full_labels.long().unsqueeze(-1)
    theta_y = fusion_phase.gather(-1, y_idx).squeeze(-1)
    fg_mask = (full_labels != bg).to(fusion_phase.dtype)            # [B, T]

    if mode == 'fccn':
        if fl.MASK_BG:
            denom = fg_mask.sum().clamp_min(1.0)
            phase_term = (theta_y * fg_mask).sum() / denom
        else:
            phase_term = theta_y.mean()
        return (loss_mag - lam * phase_term) / 2.0

    if mode == 'cosine':
        cos_loss = 1.0 - torch.cos(theta_y)                         # [B, T] ∈ [0, 2]
        if fl.MASK_BG:
            denom = fg_mask.sum().clamp_min(1.0)
            term = (cos_loss * fg_mask).sum() / denom
        else:
            term = cos_loss.mean()
        return loss_mag + lam * term

    if mode == 'coherence':
        # Per-clip circular resultant of θ[y] over FG segments.
        cos_v = torch.cos(theta_y)
        sin_v = torch.sin(theta_y)
        denom = fg_mask.sum(dim=-1).clamp_min(1.0)                  # [B]
        mc = (cos_v * fg_mask).sum(dim=-1) / denom                  # [B]
        ms = (sin_v * fg_mask).sum(dim=-1) / denom                  # [B]
        Rbar = torch.sqrt(mc ** 2 + ms ** 2 + 1e-8)                 # [B] ∈ [0, 1]
        # Only count clips with ≥2 FG segments (coherence undefined for 1).
        valid = (fg_mask.sum(dim=-1) >= 2.0).to(fusion_phase.dtype)
        if valid.sum() < 1.0:
            return loss_mag
        term = ((1.0 - Rbar) * valid).sum() / valid.sum().clamp_min(1.0)
        return loss_mag + lam * term

    raise ValueError(f"Unknown cfg.FCCN_LOSS.MODE: {mode!r} "
                     "(expected 'fccn' | 'cosine' | 'coherence')")


# ============================================================================
# Modality-balance loss (NEW — exploits the §3 finding that events live near
# arg z = π/4 in the raw similarity plane). Applied to the ComplexCNN fused
# phase output; circular-aware (1 - cos(Δ)) to handle wrap-around.
# ----------------------------------------------------------------------------
# fusion_phase: [B, T, C+1]   atan2(im, re) of fused complex output
# full_labels:  [B, T]        long GT class index (event id or bg_id)
# Returns the auxiliary loss (scalar), so caller adds  loss += λ_balance * aux.
# ============================================================================
def compute_modality_balance_loss(fusion_phase, full_labels, cfg_obj,
                                  learnable_event=None, learnable_bg=None,
                                  per_class_targets=None):
    """1 - cos(θ_y - target).  Event target = π/4 (or learnable), bg target = 0.

    'fixed':       target_evt = cfg.EVENT_TARGET (constant, π/4 by default)
                    target_bg  = cfg.BG_TARGET   (0 by default)
    'learnable':   target_evt = learnable_event (1 scalar), bg likewise.
    'per_class':   target_evt = per_class_targets[y] (C-length vector).
    """
    mbl = cfg_obj.MODALITY_BALANCE_LOSS
    if not mbl.ENABLED or fusion_phase is None:
        return torch.zeros((), device=fusion_phase.device if fusion_phase is not None
                           else torch.device('cpu'))
    bg = cfg_obj.TRAIN_BG_CLASS_ID
    y_idx = full_labels.long().unsqueeze(-1)                         # [B, T, 1]
    theta_y = fusion_phase.gather(-1, y_idx).squeeze(-1)             # [B, T]
    fg_mask = (full_labels != bg).to(fusion_phase.dtype)             # [B, T]
    bg_mask = 1.0 - fg_mask

    mode = getattr(mbl, 'MODE', 'fixed')
    if mode == 'fixed':
        evt_target = torch.tensor(float(mbl.EVENT_TARGET), device=theta_y.device)
        bg_target  = torch.tensor(float(mbl.BG_TARGET),    device=theta_y.device)
    elif mode == 'learnable':
        evt_target = learnable_event
        bg_target  = learnable_bg
    elif mode == 'per_class':
        # per_class_targets is [C]; index by full_labels (event classes only)
        target_all = per_class_targets[full_labels.long().clamp_max(per_class_targets.shape[0] - 1)]
        evt_target = target_all
        bg_target  = torch.tensor(float(mbl.BG_TARGET), device=theta_y.device)
    else:
        raise ValueError(f"Unknown MODALITY_BALANCE_LOSS.MODE: {mode!r}")

    # 1 - cos(Δ) on event segments
    fg_term = (1.0 - torch.cos(theta_y - evt_target)) * fg_mask      # [B, T]
    fg_denom = fg_mask.sum().clamp_min(1.0)
    aux_evt = fg_term.sum() / fg_denom                                # scalar in [0, 2]

    aux_bg = torch.zeros_like(aux_evt)
    if float(mbl.LAMBDA_BG) > 0:
        bg_term = (1.0 - torch.cos(theta_y - bg_target)) * bg_mask
        bg_denom = bg_mask.sum().clamp_min(1.0)
        aux_bg = bg_term.sum() / bg_denom

    return float(mbl.LAMBDA_EVENT) * aux_evt + float(mbl.LAMBDA_BG) * aux_bg


# ============================================================================
# Phase-Spread Loss (NEW).
# §3-O3 says class identity lives in a per-class angular spread. Collapse-style
# phase losses (FCCN cosine/coherence/MBL) all push θ to a single point and
# destroy that spread. This loss does the opposite: it (i) tightens phase
# within a class (alignment / vMF concentration) and (ii) pushes batch
# class-mean phases apart (hyperspherical uniformity / Spread-Out / CIDER).
# Background is treated as one more class — bg/event separation is inter-class
# spread between bg and the event classes.
#
# References:
#   ArcFace [Deng+ CVPR'19], Wang & Isola "Alignment & Uniformity" [ICML'20],
#   CIDER [Ming+ ICLR'23], Spread-Out [Zhang+ CVPR'17], vMF MLE [Banerjee+ '05].
#
# Open-vocab safe: no per-class learnable parameters. Class means are computed
# from the batch — unseen classes simply aren't in training batches.
# ============================================================================
def compute_phase_spread_loss(fusion_phase, full_labels, cfg_obj):
    """
    L = λ_intra · (1 − mean_c R̄_c)   ← per-class concentration (vMF)
      + λ_spread · mean_{c≠c'} max(0, cos(φ_c−φ_c′) − cos(margin))
                                       ↑ inter-class spread (hyperspherical)

    Args:
        fusion_phase: [B, T, K]   atan2(im, re) of ComplexCNN output
        full_labels:  [B, T]      long GT class index (0..K-1; bg = TRAIN_BG_CLASS_ID)
        cfg_obj:      cfg with cfg.PHASE_SPREAD_LOSS.{ENABLED, LAMBDA_INTRA,
                      LAMBDA_SPREAD, MARGIN, BG_EVENT_WEIGHT}.
    Returns: scalar loss tensor (zero if disabled or insufficient classes).
    """
    psl = getattr(cfg_obj, 'PHASE_SPREAD_LOSS', None)
    if psl is None or not psl.ENABLED or fusion_phase is None:
        return torch.zeros((), device=fusion_phase.device if fusion_phase is not None
                           else torch.device('cpu'))

    B, T, K = fusion_phase.shape
    bg_id   = cfg_obj.TRAIN_BG_CLASS_ID

    # θ at the GT class for every (b, t): [B*T]
    y       = full_labels.long().reshape(-1)                         # [N]
    theta   = fusion_phase.gather(-1, full_labels.long().unsqueeze(-1)).squeeze(-1)
    theta_y = theta.reshape(-1)                                      # [N]
    cos_y, sin_y = torch.cos(theta_y), torch.sin(theta_y)

    # Aggregate per-class sums and counts in one shot (no Python loops).
    sum_cos   = torch.zeros(K, device=theta_y.device, dtype=theta_y.dtype)
    sum_sin   = torch.zeros(K, device=theta_y.device, dtype=theta_y.dtype)
    count     = torch.zeros(K, device=theta_y.device, dtype=theta_y.dtype)
    sum_cos.index_add_(0, y, cos_y)
    sum_sin.index_add_(0, y, sin_y)
    count.index_add_(0,  y, torch.ones_like(cos_y))

    valid_mask = count >= 1.0                                         # [K]
    if valid_mask.sum() < 2:
        return torch.zeros((), device=theta_y.device)

    # Per-class circular mean (mc, ms) and mean resultant length R̄_c.
    denom = count[valid_mask].clamp_min(1.0)
    mc    = sum_cos[valid_mask] / denom
    ms    = sum_sin[valid_mask] / denom
    Rbar  = torch.sqrt(mc ** 2 + ms ** 2 + 1e-8)                      # [V]
    phi   = torch.atan2(ms, mc)                                        # [V]
    V     = phi.shape[0]

    # ---- Intra: per-class concentration. Average across classes present. ----
    L_intra = (1.0 - Rbar).mean()                                      # scalar

    # ---- Inter: pairwise hinge on cos(φ_i − φ_j). Want them ≤ cos(margin). ----
    margin   = float(psl.MARGIN)                                       # radians
    cos_m    = float(np.cos(margin)) if margin > 0 else 1.0
    diff     = phi.unsqueeze(0) - phi.unsqueeze(1)                     # [V, V]
    cos_sim  = torch.cos(diff)                                          # [V, V]

    # Off-diagonal mask
    eye      = torch.eye(V, device=phi.device, dtype=phi.dtype)
    off      = 1.0 - eye

    # Optional upweight on bg-event pairs. We need to know which valid index
    # corresponds to bg (if present).
    weight   = off.clone()
    bg_w     = float(psl.BG_EVENT_WEIGHT)
    if bg_w != 1.0:
        idx_map  = torch.nonzero(valid_mask, as_tuple=False).squeeze(-1)  # [V] global class ids
        bg_pos   = (idx_map == bg_id).nonzero(as_tuple=False)
        if bg_pos.numel() > 0:
            j = bg_pos.item()
            weight[j, :] = weight[j, :] * bg_w
            weight[:, j] = weight[:, j] * bg_w
            weight[j, j] = 0.0

    # Hinge: only penalize pairs that are too close (cos_sim > cos_m).
    if margin > 0:
        hinge = (cos_sim - cos_m).clamp_min(0.0)
    else:
        hinge = (cos_sim + 1.0) * 0.5                                  # [0, 1], no margin

    L_spread = (hinge * weight).sum() / weight.sum().clamp_min(1.0)

    return float(psl.LAMBDA_INTRA) * L_intra + float(psl.LAMBDA_SPREAD) * L_spread


def apply_fccn_cli_overrides(args_obj, cfg_obj):
    """Apply --fccn_* CLI overrides to cfg.FCCN_LOSS (in place).
    Returns a one-line summary string for logging."""
    fl = cfg_obj.FCCN_LOSS
    if getattr(args_obj, 'fccn_enabled', None) is not None:
        fl.ENABLED = (args_obj.fccn_enabled.lower() == 'true')
    if getattr(args_obj, 'fccn_mode', None) is not None:
        fl.MODE = args_obj.fccn_mode
    if getattr(args_obj, 'fccn_lambda', None) is not None:
        fl.LAMBDA = float(args_obj.fccn_lambda)
    if getattr(args_obj, 'fccn_mask_bg', None) is not None:
        fl.MASK_BG = (args_obj.fccn_mask_bg.lower() == 'true')
    return (f"FCCN_LOSS: ENABLED={fl.ENABLED} MODE={fl.MODE} "
            f"LAMBDA={fl.LAMBDA} MASK_BG={fl.MASK_BG}")


# ============================================================================
# Phase-Aware Logit Gate
# ----------------------------------------------------------------------------
# Replaces the magnitude-only logit `|z|` with `|z|·(cos(angle(z) − θ_y) + 1)/2`
# where θ_y is a class-specific target phase: one of two learnable scalars
# (event_phase, bg_phase). Encodes the binary phase decomposition discovered
# in the Q3 diagnostic into the training objective itself — phase becomes a
# supervised gate on magnitude rather than an emergent property.
#
# Mathematically impossible on real-valued networks (no angle to gate on).
# ============================================================================
class PhaseAwareLogit(nn.Module):
    """Class-conditional phase gating of ComplexCNN's magnitude output.

    For each (b, t, y), the gated logit is:
        gate = (cos(phase − θ_y) + 1) / 2     ∈ [0, 1]
        logit = magnitude · gate

    θ_y = bg_phase if y is the background class else event_phase. Both phases
    are scalar nn.Parameters (when LEARNABLE=True) initialized from the Q3
    diagnostic measurements on a trained CReLU ComplexCNN (event ≈ +3.07 rad,
    bg ≈ −2.88 rad).

    When the network's phase at slot y disagrees with the expected target, the
    gate suppresses the magnitude there — explicitly exploiting the
    spontaneous decomposition (phase = event/bg, magnitude = per-class).
    """
    def __init__(self, num_classes=None, bg_class_idx=None,
                 event_phase_init=3.07, bg_phase_init=-2.88,
                 learnable=True):
        """num_classes and bg_class_idx are accepted only for backward-compat
        and logging; they are NOT used in forward. The bg slot is always
        taken to be the LAST index of the input — which is the convention in
        this codebase (TRAIN_BG_CLASS_ID and TOTAL_BG_CLASS_ID are both the
        last index of their respective vocab). This makes PAL size-agnostic:
        the same module handles 47-wide train logits and 68-wide eval logits."""
        super().__init__()
        self.num_classes = num_classes
        self.bg_class_idx = bg_class_idx
        if learnable:
            self.event_phase = nn.Parameter(torch.tensor(float(event_phase_init)))
            self.bg_phase    = nn.Parameter(torch.tensor(float(bg_phase_init)))
        else:
            self.register_buffer('event_phase', torch.tensor(float(event_phase_init)))
            self.register_buffer('bg_phase',    torch.tensor(float(bg_phase_init)))

    def forward(self, magnitude, phase):
        """
        Args:
            magnitude: [B, T, K] from ComplexCNN (pre-normalization).
                       K = num_train_classes+1 during train, num_total_classes+1
                       during eval. The bg slot is always at index K-1.
            phase:     [B, T, K] from ComplexCNN, in [-π, π].
        Returns:
            logits:    [B, T, K] phase-gated magnitudes.
        """
        K = phase.size(-1)
        # target[k] = bg_phase if k == K-1 else event_phase.
        # Built on-the-fly so the same module works for any K (47 train, 68 eval).
        bg_mask = torch.zeros(K, device=phase.device, dtype=phase.dtype)
        bg_mask[-1] = 1.0
        target = self.event_phase * (1.0 - bg_mask) + self.bg_phase * bg_mask
        gate = (torch.cos(phase - target) + 1.0) / 2.0
        return magnitude * gate


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
# Alternative learnable fusion families (2-stream only).
#
# Controls for whether complex fusion beats *other* learnable fusion
# designs, not merely a stacked real CNN. Each head below keeps RealCNN's
# 5-layer temporal-conv trunk and differs only in how simm_vt and simm_at are
# combined at layer 1, so the fusion family is the sole free variable:
#
#   real      channel-stack, then convolve         (additive, unconstrained)
#   mlp       pointwise (temporal_kernel=1) stack  (additive, no temporal context)
#   gated     z*h_v + (1-z)*h_a, z learned         (Arevalo et al., GMU)
#   bilinear  h_v ⊙ h_a                            (Kim et al., low-rank MLB)
#   complex   (W_R+iW_I)(x_R+ix_I)                 (multiplicative, weight-tied)
#
# gated and bilinear are the two multiplicative competitors: they get the same
# cross terms the complex head has, but free rather than tied. Widths are chosen
# so every head lands within 2% of ComplexCNN's 31,874 params (see
# _fusion_param_count in tools/check_fusion_params.py).
# ============================================================================

class _TrunkMixin(nn.Module):
    """Shared 4-layer h->h temporal trunk + 1x1 projection to logits."""

    def _build_trunk(self, hidden_channels, temporal_kernel):
        t_pad = temporal_kernel // 2
        self.bn1 = nn.BatchNorm2d(hidden_channels)
        self.conv2 = nn.Conv2d(hidden_channels, hidden_channels, (temporal_kernel, 1), padding=(t_pad, 0))
        self.bn2 = nn.BatchNorm2d(hidden_channels)
        self.conv3 = nn.Conv2d(hidden_channels, hidden_channels, (temporal_kernel, 1), padding=(t_pad, 0))
        self.bn3 = nn.BatchNorm2d(hidden_channels)
        self.conv4 = nn.Conv2d(hidden_channels, hidden_channels, (temporal_kernel, 1), padding=(t_pad, 0))
        self.bn4 = nn.BatchNorm2d(hidden_channels)
        self.conv5 = nn.Conv2d(hidden_channels, 1, kernel_size=(1, 1), padding=0)

    def _run_trunk(self, x):
        x = F.relu(self.bn1(x))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.conv5(x)
        # |x| mirrors ComplexCNN's magnitude output, so downstream min-max
        # normalisation and the loss see the same value range in every family.
        return torch.abs(x).squeeze(1)

    @staticmethod
    def _check_two_stream(simm_imag_vt, simm_pt):
        if simm_imag_vt is not None or simm_pt is not None:
            raise ValueError(
                "gated/bilinear fusion heads are defined for the 2-stream "
                "(simm_vt, simm_at) setting only; disable IHSV and PHASE.")


class GatedFusionCNN(_TrunkMixin):
    """Gated Multimodal Unit fusion: h = z*tanh(W_v x_v) + (1-z)*tanh(W_a x_a)."""

    def __init__(self, hidden_channels=45, temporal_kernel=5):
        super().__init__()
        t_pad = temporal_kernel // 2
        self.conv_v = nn.Conv2d(1, hidden_channels, (temporal_kernel, 1), padding=(t_pad, 0))
        self.conv_a = nn.Conv2d(1, hidden_channels, (temporal_kernel, 1), padding=(t_pad, 0))
        self.conv_z = nn.Conv2d(2, hidden_channels, (temporal_kernel, 1), padding=(t_pad, 0))
        self._build_trunk(hidden_channels, temporal_kernel)

    def forward(self, simm_vt, simm_imag_vt, simm_at, simm_pt=None):
        self._check_two_stream(simm_imag_vt, simm_pt)
        xv = simm_vt.unsqueeze(1)                      # [bs, 1, 10, C+1]
        xa = simm_at.unsqueeze(1)
        z = torch.sigmoid(self.conv_z(torch.cat([xv, xa], dim=1)))
        h = z * torch.tanh(self.conv_v(xv)) + (1.0 - z) * torch.tanh(self.conv_a(xa))
        return self._run_trunk(h)


class BilinearFusionCNN(_TrunkMixin):
    """Low-rank bilinear (MLB) fusion: h = tanh(W_v x_v) ⊙ tanh(W_a x_a).

    The Hadamard product gives every cross term x_v[i]*x_a[j] within a channel,
    with the two projections learned independently -- i.e. the complex head's
    cross terms without the weight tying.
    """

    def __init__(self, hidden_channels=45, temporal_kernel=5):
        super().__init__()
        t_pad = temporal_kernel // 2
        self.conv_v = nn.Conv2d(1, hidden_channels, (temporal_kernel, 1), padding=(t_pad, 0))
        self.conv_a = nn.Conv2d(1, hidden_channels, (temporal_kernel, 1), padding=(t_pad, 0))
        self._build_trunk(hidden_channels, temporal_kernel)

    def forward(self, simm_vt, simm_imag_vt, simm_at, simm_pt=None):
        self._check_two_stream(simm_imag_vt, simm_pt)
        xv = simm_vt.unsqueeze(1)
        xa = simm_at.unsqueeze(1)
        h = torch.tanh(self.conv_v(xv)) * torch.tanh(self.conv_a(xa))
        return self._run_trunk(h)


# ============================================================================
# DualPathRealCNN — param-matched real counterpart to ComplexCNN
# Mirrors ablation_study.py:DualPathRealCNN. Independent V and A paths,
# additive merge, magnitude output. ~31,841 params at hidden=32.
# ============================================================================
class DualPathRealCNN(nn.Module):
    def __init__(self, hidden_channels=32, temporal_kernel=5):
        super(DualPathRealCNN, self).__init__()
        t_pad = temporal_kernel // 2

        self.v_conv1 = nn.Conv2d(1, hidden_channels, kernel_size=(temporal_kernel, 1), padding=(t_pad, 0))
        self.v_bn1 = nn.BatchNorm2d(hidden_channels)
        self.v_conv2 = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=(temporal_kernel, 1), padding=(t_pad, 0))
        self.v_bn2 = nn.BatchNorm2d(hidden_channels)
        self.v_conv3 = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=(temporal_kernel, 1), padding=(t_pad, 0))
        self.v_bn3 = nn.BatchNorm2d(hidden_channels)
        self.v_conv4 = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=(temporal_kernel, 1), padding=(t_pad, 0))
        self.v_bn4 = nn.BatchNorm2d(hidden_channels)

        self.a_conv1 = nn.Conv2d(1, hidden_channels, kernel_size=(temporal_kernel, 1), padding=(t_pad, 0))
        self.a_bn1 = nn.BatchNorm2d(hidden_channels)
        self.a_conv2 = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=(temporal_kernel, 1), padding=(t_pad, 0))
        self.a_bn2 = nn.BatchNorm2d(hidden_channels)
        self.a_conv3 = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=(temporal_kernel, 1), padding=(t_pad, 0))
        self.a_bn3 = nn.BatchNorm2d(hidden_channels)
        self.a_conv4 = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=(temporal_kernel, 1), padding=(t_pad, 0))
        self.a_bn4 = nn.BatchNorm2d(hidden_channels)

        self.merge_conv = nn.Conv2d(hidden_channels, 1, kernel_size=(1, 1), padding=0)

    def forward(self, simm_vt, simm_at, simm_imag_vt=None, simm_pt=None,
                return_phase=False):
        """API-compatible with ComplexCNN.forward. The optional
        simm_imag_vt / simm_pt / return_phase kwargs are accepted only so the
        train_epoch dispatch (which calls all fusion CNNs with 4 positional
        args) doesn't error — they are SILENTLY IGNORED for this 2-stream
        real-valued architecture. return_phase=True returns (logits, None)
        since there is no phase in a real-valued network."""
        del simm_imag_vt, simm_pt  # documented as ignored
        v = simm_vt.unsqueeze(1)
        a = simm_at.unsqueeze(1)

        v = F.relu(self.v_bn1(self.v_conv1(v)))
        v = F.relu(self.v_bn2(self.v_conv2(v)))
        v = F.relu(self.v_bn3(self.v_conv3(v)))
        v = F.relu(self.v_bn4(self.v_conv4(v)))

        a = F.relu(self.a_bn1(self.a_conv1(a)))
        a = F.relu(self.a_bn2(self.a_conv2(a)))
        a = F.relu(self.a_bn3(self.a_conv3(a)))
        a = F.relu(self.a_bn4(self.a_conv4(a)))

        out = self.merge_conv(v + a)
        out = torch.abs(out)
        pred_logits = out.squeeze(1)
        if return_phase:
            return pred_logits, None
        return pred_logits


# ============================================================================
# ============================================================================
# Idea 1 — Cross-modal AV Phase Coherence Loss (novel for OV-AVEL)
# ----------------------------------------------------------------------------
# Operates on the *raw* similarity tensors (simm_vt, simm_at) BEFORE the
# ComplexCNN. At that stage phase is fully intact (no CReLU has run yet).
#
# For each (frame, class) we form
#       z[t, c] = simm_vt[t, c] + i · simm_at[t, c]
#       φ[t, c] = arg(z[t, c])              ∈ (-π, π]
#
# Within an event of class c*, the temporal phase φ[:, c*] should be
# COHERENT (e.g., AV evidence consistently points the same way). For
# non-event frames or non-GT classes, φ should be incoherent. We measure
# coherence with the Kuramoto order parameter R = | (1/W) Σ e^{iφ_t'} |
# over a small temporal window.
#
# The loss = - E[ R at GT class within events ] + λ · E[ R at non-GT classes ]
#
# Gradient flows back through simm_vt / simm_at → temporal SA blocks. CReLU
# is downstream and never touches this loss.
# ============================================================================

def compute_av_phase_coherence_loss(simm_vt, simm_at, full_labels, bg_flag,
                                    window_size=3, lambda_bg=0.5, margin=0.0):
    """Cross-modal AV phase coherence auxiliary loss (Idea 1).

    Args:
        simm_vt:     [B, T, K]  cosine sim of visual feats × text   (real part)
        simm_at:     [B, T, K]  cosine sim of audio feats  × text   (imag part)
        full_labels: [B, T]     long, class IDs in [0, K) ; bg_flag = bg index
        bg_flag:     int, the background class index
        window_size: int, temporal window for circular mean (odd ≥ 1)
        lambda_bg:   weight on the "non-GT class coherence should be low" term
        margin:      optional margin for the gap

    Returns:
        loss:  scalar (smaller = more coherent at GT, less at non-GT)
        stats: dict for monitoring
    """
    B, T, K = simm_vt.shape
    device  = simm_vt.device

    # Complex AV similarity per (frame, class)
    z   = simm_vt + 1j * simm_at                              # [B, T, K], complex
    eiphi = z / (z.abs() + 1e-8)                              # unit-norm: e^{iφ}

    # Temporal circular mean over a sliding window via 1-D conv.
    # eiphi_real, eiphi_imag each shape [B, K, T] → conv along time.
    pad = window_size // 2
    kernel = torch.ones(1, 1, window_size, device=device) / window_size
    er = eiphi.real.permute(0, 2, 1)                          # [B, K, T]
    ei = eiphi.imag.permute(0, 2, 1)                          # [B, K, T]
    # Grouped conv treats each class independently (groups=K).
    kernel_k = kernel.expand(K, 1, window_size).contiguous()
    mean_er = F.conv1d(er, kernel_k, padding=pad, groups=K)   # [B, K, T]
    mean_ei = F.conv1d(ei, kernel_k, padding=pad, groups=K)   # [B, K, T]
    R = torch.sqrt(mean_er ** 2 + mean_ei ** 2 + 1e-8)        # [B, K, T] in [0, 1]
    R = R.permute(0, 2, 1)                                    # [B, T, K]

    # Masks
    is_event = (full_labels != bg_flag).float()                                   # [B, T]
    gt_onehot = F.one_hot(full_labels.clamp_min(0), num_classes=K).float()        # [B, T, K]
    nongt_mask = 1.0 - gt_onehot                                                  # [B, T, K]

    # GT-class coherence within events
    gt_coh_per_frame = (R * gt_onehot).sum(dim=-1)                                # [B, T]
    n_event = is_event.sum().clamp_min(1.0)
    R_gt_event = (gt_coh_per_frame * is_event).sum() / n_event

    # Non-GT-class coherence (averaged across the K-1 wrong classes, all frames).
    # Goal: small — wrong classes shouldn't have a coherent AV story.
    nongt_coh_per_frame = (R * nongt_mask).sum(dim=-1) / (K - 1)                  # [B, T]
    R_nongt = nongt_coh_per_frame.mean()

    # Gap loss with optional margin: maximize R_gt_event - R_nongt.
    gap = R_gt_event - R_nongt
    loss = -(gap - margin) + lambda_bg * R_nongt              # encourage gap, push nongt down

    stats = {
        'R_gt_event':   R_gt_event.item(),
        'R_nongt':      R_nongt.item(),
        'coherence_gap': gap.item(),
    }
    return loss, stats


# ============================================================================
# Idea 4 — Phase-Variance FG/BG Gate (novel for OV-AVEL)
# ----------------------------------------------------------------------------
# Inference-time gate. Computes the circular variance of arg(simm_vt + i·simm_at)
# across the class axis at each frame. High variance means no class achieves a
# stable cross-modal phase angle ⇒ likely background. Two trainable scalars.
# ============================================================================

class PhaseVarianceGate(nn.Module):
    """Phase-variance background gate (Idea 4).

    fg_prob[t] = σ( -α · phase_variance[t] + β )

    where  phase_variance[t] = 1 - | (1/K) Σ_c e^{i·arg(z[t,c])} |   (circular var)
    and    z[t, c] = simm_vt[t, c] + i · simm_at[t, c].
    """
    def __init__(self, alpha_init=4.0, beta_init=2.0, learnable=True):
        super().__init__()
        if learnable:
            self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))
            self.beta  = nn.Parameter(torch.tensor(float(beta_init)))
        else:
            self.register_buffer('alpha', torch.tensor(float(alpha_init)))
            self.register_buffer('beta',  torch.tensor(float(beta_init)))

    @staticmethod
    def phase_variance(simm_vt, simm_at):
        """Return circular variance ∈ [0, 1] over the class axis. [B, T]."""
        z = simm_vt + 1j * simm_at                                # [B, T, K]
        eiphi = z / (z.abs() + 1e-8)
        m = eiphi.mean(dim=-1)                                    # [B, T] complex
        return 1.0 - m.abs()                                      # [B, T]

    def forward(self, pred_logits, simm_vt, simm_at, bg_flag):
        """Apply the gate. pred_logits [B, T, K] → final_logits [B, T, K]."""
        var = self.phase_variance(simm_vt, simm_at)               # [B, T]
        fg_prob = torch.sigmoid(-self.alpha * var + self.beta)    # [B, T]

        # Multiplicative gate: suppress fg, boost bg in proportion.
        K = pred_logits.size(-1)
        fg_mask = torch.ones(K, device=pred_logits.device)
        fg_mask[bg_flag] = 0.0                                    # 1 for fg cols, 0 for bg
        bg_mask = 1.0 - fg_mask                                   # 0 for fg cols, 1 for bg

        scale_fg = fg_prob.unsqueeze(-1)                          # [B, T, 1]
        scale_bg = (1.0 - fg_prob).unsqueeze(-1)                  # [B, T, 1]

        # Element-wise: fg cols × scale_fg, bg col × scale_bg
        gated = pred_logits * (scale_fg * fg_mask + scale_bg * bg_mask)
        return gated, fg_prob


# ============================================================================
# Boundary-aware loss + Viterbi decoding
# ----------------------------------------------------------------------------
# Both target the dominant failure mode of frame-level OV-AVEL: temporal
# boundary jitter and bg-holes inside events. Boundary-weighted CE pushes
# extra training gradient to event onset/offset frames; the structured
# decoder enforces "at most one contiguous fg segment per video" at decode
# time. Each is gated by its own config flag so they can be ablated on/off.
# ============================================================================

def boundary_weighted_ce(pred_logits, full_labels, alpha):
    """Per-frame CE up-weighted at event boundaries.
    A boundary frame is one where the class changes between consecutive frames
    (i.e., the frame is at an event onset or offset). Each frame's weight is
    `1 + alpha * is_boundary`.

    pred_logits: [B, T, K]
    full_labels: [B, T] (long, class IDs)
    alpha:        scalar
    Returns scalar loss.
    """
    fl = full_labels
    diff_left  = torch.zeros_like(fl, dtype=torch.bool)
    diff_left[:, 1:]   = fl[:, 1:]  != fl[:, :-1]
    diff_right = torch.zeros_like(fl, dtype=torch.bool)
    diff_right[:, :-1] = fl[:, :-1] != fl[:, 1:]
    is_boundary = (diff_left | diff_right).float()                       # [B, T]
    weights     = 1.0 + alpha * is_boundary                              # [B, T]

    ce_per_frame = F.cross_entropy(
        pred_logits.permute(0, 2, 1),                                    # [B, K, T]
        full_labels,                                                     # [B, T]
        reduction='none',
    )                                                                    # [B, T]
    return (ce_per_frame * weights).sum() / weights.sum().clamp_min(1.0)


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


def cartesian_complex_construct(primary_sim, complementary_sim):
    """
    Construct complex real/imag from Cartesian form: z = primary + i·complementary

    Both inputs are cosine similarities in [0, 1] (after min-max norm), so they
    live in the same space and scale. The complex convolution layers learn
    cross-channel correlations via (W_r + iW_i)(x_r + ix_i), acting as a
    structured regularizer that couples the two similarity streams.

    Args:
        primary_sim: main similarity [bs, 10, C+1] in [0, 1] (e.g., simm_at)
        complementary_sim: auxiliary similarity [bs, 10, C+1] in [0, 1] (e.g., simm_pt)

    Returns:
        real: primary_sim (unchanged)
        imag: complementary_sim (unchanged)
    """
    return primary_sim, complementary_sim


def temporal_smooth(pred_logits, kernel_size=3, sigma=1.0):
    """
    Gaussian smoothing along temporal axis (dim=1) only.

    Args:
        pred_logits: [bs, T, C+1] prediction logits
        kernel_size: odd integer (3 recommended)
        sigma: Gaussian std (1.0 recommended)

    Returns:
        smoothed: [bs, T, C+1]
    """
    bs, T, C = pred_logits.shape

    # Reshape: [bs, C, T] for conv1d
    x = pred_logits.permute(0, 2, 1)

    # Create Gaussian kernel
    coords = torch.arange(kernel_size, device=pred_logits.device, dtype=pred_logits.dtype)
    coords = coords - kernel_size // 2
    kernel = torch.exp(-coords**2 / (2 * sigma**2))
    kernel = kernel / kernel.sum()

    # Expand for grouped convolution: [C, 1, kernel_size]
    kernel = kernel.view(1, 1, kernel_size).expand(C, 1, kernel_size)

    # Apply smoothing independently per category
    padding = kernel_size // 2
    smoothed = F.conv1d(x, kernel, padding=padding, groups=C)

    return smoothed.permute(0, 2, 1)


# ============================================================================
# Phase Coherence Loss for F1 Improvement
# Based on: Mean Phase Coherence (MPC) across D features - properly implemented
# Reference: test_phase_coherence.py showed MPC event=0.239 vs background=0.181
# ============================================================================

def compute_phase_coherence_loss(phase_info, avc_labels, window_size=3):
    """
    Compute phase coherence loss using proper Mean Phase Coherence (MPC).

    MPC measures cross-modal phase synchronization across D=1024 features:
    MPC = |mean_D(exp(i * (phase_audio - phase_video)))|

    - High MPC (~0.24): Audio and video features are phase-locked (event)
    - Low MPC (~0.18): Random phase offsets (background)

    This is the CORRECT implementation matching test_phase_coherence.py.

    Args:
        phase_info: Dict containing:
            - audio_phase_raw: [B, T, D] per-feature phases
            - visual_phase_raw: [B, T, D] per-feature phases
        avc_labels: [B, T] binary labels (1=event, 0=background)
        window_size: Window for temporal smoothing (optional refinement)

    Returns:
        loss: Scalar loss value
        stats: Dict with monitoring statistics
    """
    # Get raw per-feature phases [B, T, D]
    audio_phase_raw = phase_info['audio_phase_raw']
    visual_phase_raw = phase_info['visual_phase_raw']

    B, T, D = audio_phase_raw.shape
    device = audio_phase_raw.device

    # =========================================================================
    # PROPER MPC COMPUTATION (matching test_phase_coherence.py)
    # =========================================================================
    # Phase difference per feature
    phase_diff = audio_phase_raw - visual_phase_raw  # [B, T, D]

    # Mean Phase Coherence: coherence across D features (NOT temporal window!)
    # MPC = |mean_D(exp(i * phase_diff))|
    complex_phases = torch.exp(1j * phase_diff.double())  # [B, T, D]
    mpc = torch.abs(complex_phases.mean(dim=-1)).float()  # [B, T]

    # MPC is now the actual audio-visual phase coherence per frame
    # Expected range: ~0.18 (background) to ~0.24 (events) based on analysis
    phase_coherence = mpc

    # Optional: Apply temporal smoothing for stability (like local_mean)
    if window_size > 1:
        smoothed_coherence = torch.zeros_like(phase_coherence)
        for t in range(T):
            start = max(0, t - window_size // 2)
            end = min(T, t + window_size // 2 + 1)
            smoothed_coherence[:, t] = phase_coherence[:, start:end].mean(dim=1)
        phase_coherence = smoothed_coherence

    # =========================================================================
    # Loss components
    # =========================================================================
    # 1. Maximize MPC during events (events should have higher A-V sync)
    event_mask = avc_labels.float()  # [B, T]
    n_event_frames = event_mask.sum() + 1e-8
    event_coherence = (phase_coherence * event_mask).sum() / n_event_frames

    # 2. Minimize MPC during background (or just maximize the gap)
    bg_mask = 1 - event_mask
    n_bg_frames = bg_mask.sum() + 1e-8
    bg_coherence = (phase_coherence * bg_mask).sum() / n_bg_frames

    # 3. Boundary detection via MPC gradient
    coherence_grad = torch.zeros_like(phase_coherence)
    coherence_grad[:, 1:] = torch.abs(phase_coherence[:, 1:] - phase_coherence[:, :-1])

    label_diff = torch.zeros_like(avc_labels.float())
    label_diff[:, 1:] = torch.abs(avc_labels[:, 1:].float() - avc_labels[:, :-1].float())
    boundary_mask = (label_diff > 0).float()
    n_boundaries = boundary_mask.sum() + 1e-8
    boundary_gradient = (coherence_grad * boundary_mask).sum() / n_boundaries

    # Combined loss: maximize (event_coherence - bg_coherence) + boundary gradient
    # This encourages discriminative MPC between events and background
    coherence_gap = event_coherence - bg_coherence
    loss = -coherence_gap - 0.5 * boundary_gradient

    stats = {
        'event_coherence': event_coherence.item(),
        'bg_coherence': bg_coherence.item(),
        'coherence_delta': coherence_gap.item(),
        'boundary_gradient': boundary_gradient.item(),
        'mpc_mean': phase_coherence.mean().item(),  # Overall MPC for monitoring
    }

    return loss, stats


# ============================================================================
from utils.Recorder import Recorder
from utils.eval_metrics import segment_level, event_level


from imagebind_finetune.models import imagebind_model_fully
from imagebind_finetune.models.imagebind_model_fully import ModalityType
from einops import rearrange, repeat, reduce

from dataloader_fully import OVAVE_Dataset
from config import cfg
from diagnose_segment_acc import Diagnoser
from layers.phase_encoder import PhaseEncoder, PhaseToMelProjector, compute_if_gd, compute_if
from pix2pix_phase2mel import load_trained_generator
from cyclegan_phase2mel import load_trained_cyclegan, translate_phase_to_mel
from cyclegan_phase2mel_conditional import load_trained as load_trained_cond_cyclegan, translate_phase_to_mel as translate_phase_to_mel_cond
from layers.lora import PhaseAudioLoRA
from layers.temporal_av_attn_layer import TransformerEncoder, ComplexTransformerEncoder
from copy import deepcopy as dcp
import pdb


# ============================================================================
# ImageBind audio normalization (must match imagebind.data.load_and_transform_audio_data)
# ----------------------------------------------------------------------------
# The CycleGAN / pix2pix generators were trained on RAW mel spectrograms saved
# by extract_spectrograms.py (no normalization applied). At inference, their
# output is fed to ImageBind's audio encoder, which was trained on mels that
# HAVE been normalized by Normalize(mean=-4.268, std=9.138) inside
# load_and_transform_audio_data. Without this normalization step, the encoder
# receives input on a ~10x larger scale than expected.
# This helper re-applies the missing normalization to bring CycleGAN / pix2pix
# outputs into ImageBind's expected input distribution.
# ============================================================================
IMAGEBIND_AUDIO_MEAN = -4.268
IMAGEBIND_AUDIO_STD  = 9.138


def apply_imagebind_audio_norm(mel):
    """Normalize a raw-scale mel tensor to ImageBind's expected input scale.

    Gated by `cfg.APPLY_IMAGEBIND_AUDIO_NORM`. Default is False so existing
    pre-patch runs are exactly reproducible. Set to True via
    `CFG_OVERRIDES_JSON='{"APPLY_IMAGEBIND_AUDIO_NORM": true}'` to enable the
    calibration fix.
    """
    if not getattr(cfg, "APPLY_IMAGEBIND_AUDIO_NORM", False):
        return mel
    return (mel - IMAGEBIND_AUDIO_MEAN) / IMAGEBIND_AUDIO_STD


# configs
# dataset_configs = get_and_save_args(parser)
# parser.set_defaults(**dataset_configs)
# args = parser.parse_args()
args = get_and_save_args(parser)
# CLI overrides for the FCCN loss sweep (no-op if --fccn_* flags aren't passed).
_fccn_summary = apply_fccn_cli_overrides(args, cfg)
print(_fccn_summary)
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

if cfg.SA.SA_ATTN_FLAG:
    if cfg.SA.SA_ATTN_TYPE == 'bothFirst':
        ext = f"_Sattn_bothFirstK{cfg.SA.SA_BOTH_LAYER_K}"
        audio_layer_ids = list(range(cfg.SA.SA_BOTH_LAYER_K))
        visual_layer_ids = list(range(cfg.SA.SA_BOTH_LAYER_K))
        sattn_type = 'bothFirst'
    elif cfg.SA.SA_ATTN_TYPE == 'bothLast':
        ext = f"_Sattn_bothLastK{cfg.SA.SA_BOTH_LAYER_K}"
        audio_layer_ids = list(range(12-cfg.SA.SA_BOTH_LAYER_K, 12))
        visual_layer_ids = list(range(32-cfg.SA.SA_BOTH_LAYER_K, 32))
        sattn_type = 'bothLast'
    elif cfg.SA.SA_ATTN_TYPE == 'evenFirst':
        ext = f"_Sattn_evenly_firstK{cfg.SA.SA_EVEN_LAYER_K}"
        audio_layer_ids  = list(range(0, 12, 3)) + [11] # [0, 3, 6, 9, 11]
        visual_layer_ids = list(range(0, 32, 8)) + [31] # [0, 8, 16, 24, 31]
        audio_layer_ids = audio_layer_ids[:cfg.SA.SA_EVEN_LAYER_K]
        visual_layer_ids = visual_layer_ids[:cfg.SA.SA_EVEN_LAYER_K]
        sattn_type = 'evenFirst'
    elif cfg.SA.SA_ATTN_TYPE == 'evenLast':
        ext = f"_Sattn_evenly_lastK{cfg.SA.SA_EVEN_LAYER_K}"
        audio_layer_ids  = list(range(0, 12, 3)) + [11] # [0, 3, 6, 9, 11]
        visual_layer_ids = list(range(0, 32, 8)) + [31] # [0, 8, 16, 24, 31]
        audio_layer_ids = audio_layer_ids[-cfg.SA.SA_EVEN_LAYER_K:]
        visual_layer_ids = visual_layer_ids[-cfg.SA.SA_EVEN_LAYER_K:]
        sattn_type = 'evenLast'
    elif cfg.SA.SA_ATTN_TYPE == 'fixedBlkids':
        ext = f"_Sattn_fixedBlkids"
        audio_layer_ids = cfg.SA.SA_FIXED_LAYER_A
        visual_layer_ids = cfg.SA.SA_FIXED_LAYER_V
        sattn_type = 'fixedBlkids'
        assert len(audio_layer_ids) == len(visual_layer_ids)
    else:
        raise NotImplementedError
else:
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



def compute_train_class_log_priors_total_space():
    """
    Per Menon et al. 2021 ('Long-Tail Learning via Logit Adjustment'), build
    log class priors from training-set frame labels, mapped into the TOTAL
    (test-side) class space. Open-set classes (in total CSV but unseen at
    train) get Laplace-smoothed priors via cfg.LOGIT_ADJ.PRIOR_SMOOTHING_EPS.

    Returns:
      log_pi: torch.FloatTensor [TOTAL_BG_CLASS_ID + 1]
      counts: torch.DoubleTensor of raw frame counts (for logging)
    """
    import pandas as _pd
    import json as _json

    num_total = cfg.TOTAL_BG_CLASS_ID + 1
    train_bg = cfg.TRAIN_BG_CLASS_ID
    total_bg = cfg.TOTAL_BG_CLASS_ID

    train_cats = _pd.read_csv(cfg.TRAIN_CLOSE_CATEGORY_CSV_PATH, header=None)[0].tolist()
    total_cats = _pd.read_csv(cfg.TOTAL_CLOSE_OPEN_CATEGORY_CSV_PATH, header=None)[0].tolist()
    # train fg-class index -> total fg-class index (skip the train bg row)
    train_to_total = {}
    for ti, cat in enumerate(train_cats):
        if ti == train_bg:
            continue
        if cat in total_cats:
            train_to_total[ti] = total_cats.index(cat)

    meta_df = _pd.read_csv(cfg.META_CSV_PATH)
    train_df = meta_df[meta_df['split'] == 'train'].reset_index(drop=True)
    with open(cfg.ANNO_JSON_PATH, 'r') as fr:
        anno_data = _json.load(fr)

    counts = torch.zeros(num_total, dtype=torch.float64)
    for idx in range(len(train_df)):
        # Same positional unpack as OVAVE_Dataset: [split, category, cls_type, vid_name]
        category = train_df.iloc[idx, 1]
        vid_name = train_df.iloc[idx, 3]
        avc_label = _json.loads(anno_data[vid_name]['label'])  # list of 10 in {0,1}
        n_fg = sum(int(x) for x in avc_label)
        n_bg = len(avc_label) - n_fg
        # Background frames -> total bg slot
        counts[total_bg] += n_bg
        # Foreground frames -> mapped category in total space (or bg if bg-only video)
        if n_fg > 0 and category in total_cats:
            cat_total_idx = total_cats.index(category)
            counts[cat_total_idx] += n_fg
        else:
            counts[total_bg] += n_fg  # bg-only video or unknown category

    # Open-set classes (in total CSV but never appear in train) have count=0.
    # Pure Laplace smoothing would make them look ~33× rarer than the rarest
    # seen class -> artificially huge tau*log_pi boost. Fill them according to
    # cfg.LOGIT_ADJ.UNSEEN_FG_MODE so the prior correction targets bg-collapse,
    # not unseen-class amplification.
    fg_mask = torch.ones(num_total, dtype=torch.bool)
    fg_mask[total_bg] = False
    seen_fg_mask = fg_mask & (counts > 0)
    unseen_fg_mask = fg_mask & (counts == 0)
    mode = str(cfg.LOGIT_ADJ.UNSEEN_FG_MODE).lower()
    if unseen_fg_mask.any() and seen_fg_mask.any():
        if mode == 'mean_fg':
            fill = counts[seen_fg_mask].mean()
        elif mode == 'min_fg':
            fill = counts[seen_fg_mask].min()
        elif mode == 'laplace':
            fill = torch.tensor(0.0, dtype=counts.dtype)
        else:
            raise ValueError(f"Unknown LOGIT_ADJ.UNSEEN_FG_MODE: {mode!r}")
        counts[unseen_fg_mask] = fill

    eps = float(cfg.LOGIT_ADJ.PRIOR_SMOOTHING_EPS)
    pi = (counts + eps) / (counts.sum() + eps * num_total)
    log_pi = torch.log(pi).float()
    return log_pi, counts


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
    val_dataloader = DataLoader(
        OVAVE_Dataset(split='val', test_data_type=args.val_data_type, debug=args.debug),
        batch_size=args.val_batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True
    )
    test_dataloader = DataLoader(
        OVAVE_Dataset(split='test', test_data_type=args.test_data_type, debug=args.debug),
        batch_size=args.test_batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True
    )
    # pdb.set_trace()

    # === Logit adjustment priors (computed once from train labels) ============
    log_pi_total = None
    if cfg.LOGIT_ADJ.ENABLED:
        log_pi_total, prior_counts = compute_train_class_log_priors_total_space()
        num_total = cfg.TOTAL_BG_CLASS_ID + 1
        total_bg = cfg.TOTAL_BG_CLASS_ID
        fg_mask = torch.ones(num_total, dtype=torch.bool)
        fg_mask[total_bg] = False
        n_seen = int((prior_counts > 0).sum().item())
        bg_log = log_pi_total[total_bg].item()
        fg_mean_log = log_pi_total[fg_mask].mean().item()
        delta = bg_log - fg_mean_log
        tau = float(cfg.LOGIT_ADJ.TAU)
        logger.info(f"==> Logit Adjustment: ENABLED (tau={tau}, smoothing_eps={cfg.LOGIT_ADJ.PRIOR_SMOOTHING_EPS})")
        logger.info(f"    seen-at-train classes: {n_seen}/{num_total}  (unseen open-set get smoothed prior)")
        logger.info(f"    log_pi[bg]={bg_log:+.4f}, mean log_pi[fg]={fg_mean_log:+.4f}, Δ(bg - mean fg)={delta:+.4f}")
        logger.info(f"    At inference: bg score is shifted by {-tau*delta:+.4f} relative to mean fg "
                    f"(positive = bg discouraged)")
    else:
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
        # text projection tuning
        text_tune_flag=cfg.TEXT_TUNE_FLAG,
        # complex temporal attention
        use_complex_temporal_attn=cfg.TA.USE_COMPLEX_ATTN,
        # dual stream temporal attention (separate audio/visual)
        use_dual_stream_temporal_attn=cfg.TA.USE_DUAL_STREAM_ATTN,
        # cross-modal complex attention (self-attn + cross-attn)
        use_cross_modal_attn=cfg.TA.USE_CROSS_MODAL_ATTN,
        # complex projection self-attention (complex 1x1 conv + real self-attn)
        use_complex_proj_attn=cfg.TA.USE_COMPLEX_PROJ_ATTN,
        # complex temporal projection self-attention (complex 5x1 conv + real self-attn)
        use_complex_temporal_proj_attn=cfg.TA.USE_COMPLEX_TEMPORAL_PROJ_ATTN,
        temporal_kernel=cfg.TA.TEMPORAL_KERNEL,
        # text-headed temporal attention (text categories as query heads)
        use_text_headed_attn=cfg.TA.USE_TEXT_HEADED_ATTN,
        use_text_headed_complex_attn=cfg.TA.USE_TEXT_HEADED_COMPLEX_ATTN,
        text_headed_refinement_layers=cfg.TA.TEXT_HEADED_REFINEMENT_LAYERS,
        # dual complex self-attention with text as imaginary
        use_dual_complex_text_imag=cfg.TA.USE_DUAL_COMPLEX_TEXT_IMAG,
        dual_complex_num_layers=cfg.TA.DUAL_COMPLEX_NUM_LAYERS,
        dual_complex_text_broadcast=cfg.TA.DUAL_COMPLEX_TEXT_BROADCAST,
        dual_complex_output_mode=cfg.TA.DUAL_COMPLEX_OUTPUT_MODE,
        # identity complex ablation (real=imag=modality)
        use_identity_complex_ablation=cfg.TA.USE_IDENTITY_COMPLEX_ABLATION,
        # split complex attention (split 1024 -> 512 real + 512 imag, concat back)
        use_split_complex_attn=cfg.TA.USE_SPLIT_COMPLEX_ATTN,
        # temporal diff complex attention (real=frame, imag=temporal difference)
        use_temporal_diff_complex_attn=cfg.TA.USE_TEMPORAL_DIFF_COMPLEX_ATTN,
        temporal_diff_padding_mode=cfg.TA.TEMPORAL_DIFF_PADDING_MODE,
        # rotary complex attention (phase = temporal position)
        use_rotary_complex_attn=cfg.TA.USE_ROTARY_COMPLEX_ATTN,
        rotary_learnable_freq=cfg.TA.ROTARY_LEARNABLE_FREQ,
        rotary_num_freq_bands=cfg.TA.ROTARY_NUM_FREQ_BANDS,
        # projected complex attention (linear projection from real to complex)
        use_projected_complex_attn=cfg.TA.USE_PROJECTED_COMPLEX_ATTN,
        projected_complex_output_mode=cfg.TA.PROJECTED_COMPLEX_OUTPUT_MODE,
        # aligned temporal diff complex attention (for paper - with alignment loss)
        use_aligned_temporal_diff_attn=cfg.TA.USE_ALIGNED_TEMPORAL_DIFF_ATTN,
        aligned_residual_weight=cfg.TA.ALIGNED_RESIDUAL_WEIGHT,
        aligned_loss_weight=cfg.TA.ALIGNED_LOSS_WEIGHT,
        # phase-aware complex attention (Hilbert projection + ModReLU)
        use_phase_aware_complex_attn=cfg.TA.USE_PHASE_AWARE_COMPLEX_ATTN,
        phase_aware_use_temporal_bias=cfg.TA.PHASE_AWARE_USE_TEMPORAL_BIAS,
        phase_aware_output_mode=cfg.TA.PHASE_AWARE_OUTPUT_MODE,
        phase_aware_modrelu_bias_init=cfg.TA.PHASE_AWARE_MODRELU_BIAS_INIT,
        # TAPA (Token-Aware Phase Attention) parameters
        use_tapa=cfg.TA.USE_TAPA,
        tapa_theta=cfg.TA.TAPA_THETA,
        tapa_alpha=cfg.TA.TAPA_ALPHA,
        tapa_imag_mode=cfg.TA.TAPA_IMAG_MODE,
        # MoE temporal attention (boundary-aware expert routing)
        use_moe_temporal_attn=cfg.MOE.ENABLED,
        moe_num_experts=cfg.MOE.NUM_EXPERTS,
        moe_expert_hidden_dim=cfg.MOE.EXPERT_HIDDEN_DIM,
        moe_router_hidden_dim=cfg.MOE.ROUTER_HIDDEN_DIM,
        moe_top_k=cfg.MOE.TOP_K,
        moe_use_temporal_diff_routing=cfg.MOE.USE_TEMPORAL_DIFF_ROUTING,
        moe_load_balance_weight=cfg.MOE.LOAD_BALANCE_WEIGHT,
    )
    # mainModel.eval()
    mainModel.to(device)

    # Log temporal attention status
    if cfg.TA.USE_PHASE_AWARE_COMPLEX_ATTN:
        logger.info(f"==> PHASE-AWARE COMPLEX TEMPORAL ATTENTION: ENABLED")
        logger.info(f"    Hilbert Projection (learnable complex lift) + ModReLU (phase-preserving)")
        logger.info(f"    use_temporal_bias={cfg.TA.PHASE_AWARE_USE_TEMPORAL_BIAS}")
        logger.info(f"    output_mode='{cfg.TA.PHASE_AWARE_OUTPUT_MODE}'")
        logger.info(f"    modrelu_bias_init={cfg.TA.PHASE_AWARE_MODRELU_BIAS_INIT}")
        logger.info(f"    TAPA: use_tapa={cfg.TA.USE_TAPA}, theta={cfg.TA.TAPA_THETA}, alpha={cfg.TA.TAPA_ALPHA}, imag_mode='{cfg.TA.TAPA_IMAG_MODE}'")
    elif cfg.TA.USE_PROJECTED_COMPLEX_ATTN:
        logger.info(f"==> PROJECTED COMPLEX TEMPORAL ATTENTION: ENABLED")
        logger.info(f"    Projects 1024 real -> 1024 real + 1024 imag via learned linear layers")
        logger.info(f"    output_mode='{cfg.TA.PROJECTED_COMPLEX_OUTPUT_MODE}'")
    elif cfg.TA.USE_ROTARY_COMPLEX_ATTN:
        logger.info(f"==> ROTARY COMPLEX TEMPORAL ATTENTION: ENABLED")
        logger.info(f"    Phase encodes temporal position (like RoPE), magnitude encodes content")
        logger.info(f"    learnable_freq={cfg.TA.ROTARY_LEARNABLE_FREQ}, num_freq_bands={cfg.TA.ROTARY_NUM_FREQ_BANDS}")
    elif cfg.TA.USE_TEMPORAL_DIFF_COMPLEX_ATTN:
        logger.info(f"==> TEMPORAL DIFF COMPLEX ATTENTION: ENABLED")
        logger.info(f"    real=static features, imag=temporal differences (motion/dynamics)")
        logger.info(f"    padding_mode='{cfg.TA.TEMPORAL_DIFF_PADDING_MODE}'")
    elif cfg.TA.USE_SPLIT_COMPLEX_ATTN:
        logger.info(f"==> SPLIT COMPLEX TEMPORAL ATTENTION: ENABLED")
        logger.info(f"    Split 1024-dim -> 512 real + 512 imag -> concat back to 1024")
        logger.info(f"    Genuine heterogeneity, preserves ALL information")
    elif cfg.TA.USE_IDENTITY_COMPLEX_ABLATION:
        logger.info(f"==> IDENTITY COMPLEX ABLATION: ENABLED (real=imag=modality)")
        logger.info(f"    DUAL_COMPLEX_NUM_LAYERS: {cfg.TA.DUAL_COMPLEX_NUM_LAYERS}")
        logger.info(f"    DUAL_COMPLEX_OUTPUT_MODE: {cfg.TA.DUAL_COMPLEX_OUTPUT_MODE}")
    elif cfg.TA.USE_DUAL_COMPLEX_TEXT_IMAG:
        logger.info(f"==> Dual Complex Self-Attention with Text as Imaginary: ENABLED")
        logger.info(f"    DUAL_COMPLEX_NUM_LAYERS: {cfg.TA.DUAL_COMPLEX_NUM_LAYERS}")
        logger.info(f"    DUAL_COMPLEX_TEXT_BROADCAST: {cfg.TA.DUAL_COMPLEX_TEXT_BROADCAST}")
        logger.info(f"    DUAL_COMPLEX_OUTPUT_MODE: {cfg.TA.DUAL_COMPLEX_OUTPUT_MODE}")
    elif cfg.TA.USE_TEXT_HEADED_ATTN or cfg.TA.USE_TEXT_HEADED_COMPLEX_ATTN:
        logger.info(f"==> Text-Headed Temporal Attention: ENABLED")
        logger.info(f"    USE_TEXT_HEADED_ATTN: {cfg.TA.USE_TEXT_HEADED_ATTN}")
        logger.info(f"    USE_TEXT_HEADED_COMPLEX_ATTN: {cfg.TA.USE_TEXT_HEADED_COMPLEX_ATTN}")
        logger.info(f"    TEXT_HEADED_REFINEMENT_LAYERS: {cfg.TA.TEXT_HEADED_REFINEMENT_LAYERS}")
    elif cfg.TA.USE_ALIGNED_TEMPORAL_DIFF_ATTN:
        logger.info(f"==> ALIGNED TEMPORAL DIFF COMPLEX ATTENTION: ENABLED")
        logger.info(f"    real=static features, imag=temporal differences")
        logger.info(f"    residual_weight={cfg.TA.ALIGNED_RESIDUAL_WEIGHT}, loss_weight={cfg.TA.ALIGNED_LOSS_WEIGHT}")
        logger.info(f"    Gated residual + alignment loss for ImageBind space preservation")
    elif cfg.MOE.ENABLED:
        logger.info(f"==> MOE TEMPORAL ATTENTION: ENABLED (MoETemporalAttentionModule)")
        logger.info(f"    Boundary-aware expert routing with {cfg.MOE.NUM_EXPERTS} experts")
        logger.info(f"    Expert types: steady (k=5), onset (causal diff), offset (anti-causal diff), boundary (k=1)")
        logger.info(f"    expert_hidden_dim={cfg.MOE.EXPERT_HIDDEN_DIM}, router_hidden_dim={cfg.MOE.ROUTER_HIDDEN_DIM}")
        logger.info(f"    top_k={cfg.MOE.TOP_K} (0=soft routing), load_balance_weight={cfg.MOE.LOAD_BALANCE_WEIGHT}")
        logger.info(f"    use_temporal_diff_routing={cfg.MOE.USE_TEMPORAL_DIFF_ROUTING}")
    elif cfg.TA.TA_ATTN_FLAG:
        logger.info(f"==> REAL TEMPORAL ATTENTION: ENABLED (TemporalAttentionModule)")
        logger.info(f"    Standard real-valued self-attention on temporal dimension")
        logger.info(f"    sa_layer_num={cfg.TA.SA_LAYER_NUM}, head_num={cfg.TA.HEAD_NUM}")

    #! =====================================================================
    #! FUSION CNN (can be disabled via config)
    #! =====================================================================
    fusion_cnn_type = getattr(cfg, 'FUSION_CNN_TYPE', 'complex')
    # Legacy `use_real_cnn` flag selects the 4-channel stacked RealCNN forward.
    # DualPathRealCNN shares ComplexCNN's (simm_vt, simm_at) signature, so it goes
    # through the ComplexCNN branch in train/validate. The alternative fusion
    # families reuse RealCNN's (vt, imag_vt, at, pt) signature.
    USE_REAL_CNN = (fusion_cnn_type in ('real', 'mlp', 'gated', 'bilinear'))

    if cfg.FUSION_CNN_ENABLED:
        if fusion_cnn_type == 'gated':
            ''' Gated Multimodal Unit fusion (fusion-family control, 2-stream only) '''
            gated_hidden = int(getattr(cfg, 'GATED_CNN_HIDDEN', 45))
            fusionCNN = GatedFusionCNN(hidden_channels=gated_hidden, temporal_kernel=5)
            fusionCNN.to(device)
            n_par = sum(p.numel() for p in fusionCNN.parameters())
            logger.info(f"==> Gated CNN for AV Fusion: ENABLED (GMU, hidden_channels={gated_hidden}, temporal_kernel=5, {n_par} params)")
            logger.info(f"    h = z*tanh(W_v simm_vt) + (1-z)*tanh(W_a simm_at),  z = sigmoid(W_z [simm_vt; simm_at])")
        elif fusion_cnn_type == 'bilinear':
            ''' Low-rank bilinear (MLB) fusion (fusion-family control, 2-stream only) '''
            bil_hidden = int(getattr(cfg, 'BILINEAR_CNN_HIDDEN', 45))
            fusionCNN = BilinearFusionCNN(hidden_channels=bil_hidden, temporal_kernel=5)
            fusionCNN.to(device)
            n_par = sum(p.numel() for p in fusionCNN.parameters())
            logger.info(f"==> Bilinear CNN for AV Fusion: ENABLED (MLB, hidden_channels={bil_hidden}, temporal_kernel=5, {n_par} params)")
            logger.info(f"    h = tanh(W_v simm_vt) * tanh(W_a simm_at)   (free cross terms, no weight tying)")
        elif fusion_cnn_type == 'mlp':
            ''' Pointwise MLP fusion: RealCNN with temporal_kernel=1 (no temporal context) '''
            mlp_hidden = int(getattr(cfg, 'MLP_CNN_HIDDEN', 100))
            mlp_in_channels = 2
            fusionCNN = RealCNN(hidden_channels=mlp_hidden, in_channels=mlp_in_channels,
                                temporal_kernel=1)
            fusionCNN.to(device)
            n_par = sum(p.numel() for p in fusionCNN.parameters())
            logger.info(f"==> MLP for AV Fusion: ENABLED (pointwise, in_channels={mlp_in_channels}, hidden_channels={mlp_hidden}, temporal_kernel=1, {n_par} params)")
        elif fusion_cnn_type == 'real_dual':
            ''' DualPathRealCNN: param-matched real counterpart to ComplexCNN '''
            fusionCNN = DualPathRealCNN(hidden_channels=32, temporal_kernel=5)
            fusionCNN.to(device)
            logger.info(f"==> DualPath Real CNN for AV Fusion: ENABLED (independent V/A paths, hidden_channels=32, temporal_kernel=5)")
            logger.info(f"    pred = |merge(V_path(simm_vt) + A_path(simm_at))|   (~31,841 params, matches ComplexCNN)")
        elif fusion_cnn_type == 'real':
            ''' Stacked-channel RealCNN: up to 4 channels (simm_vt, simm_at,
                + simm_vis_imag if iHSV/Depth enabled, + simm_pt if Phase enabled). '''
            # NOTE: RealCNN.forward stacks in order [simm_vt, simm_at, simm_imag_vt?, simm_pt?]
            real_in_channels = 2  # simm_vt, simm_at always present
            ch_desc = ["simm_vt (RGB visual)", "simm_at (mel audio)"]
            if cfg.IHSV.ENABLED or cfg.DEPTH.ENABLED:
                real_in_channels += 1
                ch_desc.append("simm_vis_imag (depth or iHSV)")
            if cfg.PHASE.ENABLED or cfg.COMPLEX_REPR.ENABLED:
                real_in_channels += 1
                ch_desc.append("simm_pt (phase audio)")
            real_hidden = int(getattr(cfg, 'REAL_CNN_HIDDEN', 45))
            fusionCNN = RealCNN(hidden_channels=real_hidden, in_channels=real_in_channels)
            fusionCNN.to(device)
            logger.info(f"==> Real CNN for AV Fusion: ENABLED (5 layers, in_channels={real_in_channels}, hidden_channels={real_hidden}, temporal_kernel=5)")
            for i, d in enumerate(ch_desc):
                logger.info(f"    ch{i}: {d}")
        else:
            ''' Complex CNN for AV fusion: 2-ch dual-complex when iHSV+phase enabled, else 1-ch '''
            # Use 2 input channels whenever AT LEAST ONE companion stream
            # (iHSV or phase) is enabled. ComplexCNN.forward populates the
            # missing companion with zeros; the layer dim must match the
            # input tensor's channel count. This was previously `and`, which
            # silently broke half-paired ablations (V-only or A-only).
            complex_in_ch = 2 if (cfg.IHSV.ENABLED or cfg.PHASE.ENABLED) else 1
            cplx_act = getattr(cfg, 'COMPLEX_CNN_ACTIVATION', 'crelu')
            cplx_modrelu_bias = float(getattr(cfg, 'COMPLEX_CNN_MODRELU_BIAS_INIT', -0.1))
            cplx_tk = int(getattr(cfg, 'COMPLEX_CNN_TEMPORAL_KERNEL', 5))
            cplx_hidden = int(getattr(cfg, 'COMPLEX_CNN_HIDDEN', 32))
            fusionCNN = ComplexCNN(hidden_channels=cplx_hidden, in_channels=complex_in_ch,
                                   temporal_kernel=cplx_tk,
                                   activation=cplx_act,
                                   modrelu_bias_init=cplx_modrelu_bias)
            fusionCNN.to(device)
            logger.info(f"==> Complex CNN for AV Fusion: ENABLED (5 layers, in_channels={complex_in_ch}, hidden_channels={cplx_hidden}, temporal_kernel={cplx_tk}, activation='{cplx_act}')")
            if cplx_act == 'modrelu':
                logger.info(f"    ModReLU bias init: {cplx_modrelu_bias}  (phase-preserving; learned dead zone)")
            if complex_in_ch == 2:
                logger.info(f"    FUSION_CHANNEL_MODE='{cfg.FUSION_CHANNEL_MODE}'")
                logger.info(f"    ch0 = simm_vt + i·simm_imag_vt   (visual real + visual imag)  [complex_pair]")
                logger.info(f"    ch1 = simm_at + i·simm_pt        (audio real  + audio imag)   [complex_pair]")
            else:
                logger.info(f"    z   = simm_vt + i·simm_at        (real=RGB visual, imag=audio mel)")
    else:
        fusionCNN = None
        logger.info(f"==> Fusion CNN: DISABLED (using direct similarity fusion)")

    # ---------------------------------------------------------------------
    # Phase-Aware Logit (optional) — exploits the Q3 binary-phase decomposition.
    # Only active when fusionCNN is ComplexCNN. Adds 2 learnable scalars.
    # ---------------------------------------------------------------------
    phase_aware_logit = None
    _pal_cfg = getattr(cfg, 'PHASE_AWARE_LOGIT', None)
    if (_pal_cfg is not None and _pal_cfg.ENABLED
            and cfg.FUSION_CNN_ENABLED and fusion_cnn_type == 'complex'):
        num_classes_total = cfg.TOTAL_BG_CLASS_ID + 1  # C+1 including bg
        phase_aware_logit = PhaseAwareLogit(
            num_classes=num_classes_total,
            bg_class_idx=cfg.TOTAL_BG_CLASS_ID,
            event_phase_init=float(_pal_cfg.EVENT_PHASE_INIT),
            bg_phase_init=float(_pal_cfg.BG_PHASE_INIT),
            learnable=bool(_pal_cfg.LEARNABLE),
        ).to(device)
        n_pal = sum(p.numel() for p in phase_aware_logit.parameters() if p.requires_grad)
        logger.info(f"==> Phase-Aware Logit: ENABLED (learnable={_pal_cfg.LEARNABLE}, "
                    f"+{n_pal} params)")
        logger.info(f"    event_phase init = {_pal_cfg.EVENT_PHASE_INIT:+.3f} rad, "
                    f"bg_phase init = {_pal_cfg.BG_PHASE_INIT:+.3f} rad")
        logger.info(f"    pred_logits = (simm_vt + simm_at) / 2 (simple average)")

    #! =====================================================================
    #! PHASE MODULE (optional, controlled by cfg.PHASE)
    #! =====================================================================
    phase_model = None
    phase_projector = None
    phase_temporal_attn = None
    pix2pix_generator = None
    pix2pix_normalizer = None
    cyclegan_generator = None
    cyclegan_if_range = None
    cyclegan_mel_range = None
    if cfg.PHASE.ENABLED:
        in_channels = 2 if cfg.PHASE.INPUT_TYPE == 'if_gd' else 1
        if cfg.PHASE.MODE == 'encoder':
            phase_model = PhaseEncoder(
                in_channels=in_channels,
                hidden_channels=cfg.PHASE.ENCODER_CHANNELS,
                num_layers=cfg.PHASE.ENCODER_LAYERS,
                output_dim=1024,
            ).to(device)
            logger.info(f"==> Phase Encoder: ENABLED (mode='encoder', input_type='{cfg.PHASE.INPUT_TYPE}')")
            logger.info(f"    in_channels={in_channels}, hidden_channels={cfg.PHASE.ENCODER_CHANNELS}, layers={cfg.PHASE.ENCODER_LAYERS}")
        elif cfg.PHASE.MODE == 'lora':
            # LoRA-adapted audio encoder for phase IF input
            # PhaseAudioLoRA: trainable stem + block adapters on frozen audio encoder
            phase_model = PhaseAudioLoRA(
                model=mainModel,
                rank=cfg.PHASE.LORA_RANK,
                alpha=cfg.PHASE.LORA_ALPHA,
                init_stem_from_pretrained=True,
                stem_init_scale=0.1,
            ).to(device)
            # Phase projector: project IF [1, 257, 204] → mel shape [1, 128, 204] before LoRA encoder
            phase_projector = PhaseToMelProjector(in_freq=257, out_freq=128).to(device)
            # Phase gets its own temporal self-attention
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
            logger.info(f"==> Phase via LoRA: ENABLED (mode='lora', input_type='{cfg.PHASE.INPUT_TYPE}')")
            logger.info(f"    Pipeline: phase IF [1,257,204] → proj [1,128,204] → LoRA-adapted audio encoder → [1024]")
            logger.info(f"    LoRA rank={cfg.PHASE.LORA_RANK}, alpha={cfg.PHASE.LORA_ALPHA}")
            logger.info(f"    Trainable params: {phase_model.num_trainable_params():,} (stem + adapters)")
            logger.info(f"    Phase temporal self-attention: {cfg.TA.SA_LAYER_NUM} layers")
        elif cfg.PHASE.MODE == 'imagebind':
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
            logger.info(f"==> Phase via ImageBind: ENABLED (mode='imagebind', input_type='{cfg.PHASE.INPUT_TYPE}')")
            logger.info(f"    Pipeline: phase [1,257,204] → proj [1,128,204] → ImageBind audio encoder → [1024]")
            logger.info(f"    Phase temporal self-attention: {cfg.TA.SA_LAYER_NUM} layers, hid={cfg.TA.HIDDEN_DIM}, heads={cfg.TA.HEAD_NUM}")
            logger.info(f"    Output: simm_pt [bs, 10, C+1] (phase-text similarity)")
        elif cfg.PHASE.MODE == 'hilbert':
            # Hilbert analytic signal: imag part is already [1, 128, 204] (same as mel)
            # No projector needed — feed directly through frozen ImageBind audio encoder
            phase_projector = None  # not needed
            # Temporal self-attention for Hilbert features (same as other modes)
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
            logger.info(f"==> Phase via Hilbert: ENABLED (mode='hilbert')")
            logger.info(f"    Pipeline: mel → Hilbert analytic imag [1,128,204] → ImageBind audio encoder (frozen) → [1024]")
            logger.info(f"    No projector needed (Hilbert imag is already mel-shaped)")
            logger.info(f"    Phase temporal self-attention: {cfg.TA.SA_LAYER_NUM} layers, hid={cfg.TA.HIDDEN_DIM}, heads={cfg.TA.HEAD_NUM}")
            logger.info(f"    Output: simm_pt [bs, 10, C+1] (hilbert-text similarity)")
        elif cfg.PHASE.MODE == 'pix2pix':
            # Pix2Pix GAN: pretrained generator translates IF phase → mel-like spectrogram
            # Generator is FROZEN — only temporal attention is trainable
            pix2pix_generator, pix2pix_normalizer = load_trained_generator(
                cfg.PHASE.PIX2PIX_CKPT, device=device)
            pix2pix_generator.eval()
            for p in pix2pix_generator.parameters():
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
            logger.info(f"==> Phase via Pix2Pix: ENABLED (mode='pix2pix')")
            logger.info(f"    Checkpoint: {cfg.PHASE.PIX2PIX_CKPT}")
            logger.info(f"    Pipeline: phase [1,257,204] → compute_if → Pix2Pix GAN (frozen) → mel-like [1,128,204] → ImageBind audio encoder (frozen) → [1024]")
            logger.info(f"    Phase temporal self-attention: {cfg.TA.SA_LAYER_NUM} layers, hid={cfg.TA.HIDDEN_DIM}, heads={cfg.TA.HEAD_NUM}")

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

        elif cfg.PHASE.MODE == 'cond_cyclegan':
            # Conditional CycleGAN: G_IF2Mel takes (IF, mel_cond) → mel-like spectrogram
            # Generator is FROZEN — only temporal attention is trainable
            cyclegan_generator, _G_Mel2IF_unused, cyclegan_if_range, cyclegan_mel_range = load_trained_cond_cyclegan(
                cfg.PHASE.COND_CYCLEGAN_CKPT, device=device)
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
            logger.info(f"==> Phase via conditional CycleGAN: ENABLED (mode='cond_cyclegan')")
            logger.info(f"    Checkpoint: {cfg.PHASE.COND_CYCLEGAN_CKPT}")
            logger.info(f"    Pipeline: (phase [1,257,204]→IF, mel_cond [1,128,204]) → CondCycleGAN G_IF2Mel(IF, mel_cond) (frozen) → mel-like [1,128,204] → ImageBind audio encoder (frozen) → [1024]")
            logger.info(f"    Phase temporal self-attention: {cfg.TA.SA_LAYER_NUM} layers, hid={cfg.TA.HIDDEN_DIM}, heads={cfg.TA.HEAD_NUM}")

    else:
        logger.info(f"==> Phase: DISABLED")

    #! =====================================================================
    #! DEPTH TEMPORAL SELF-ATTENTION (optional, when DEPTH is enabled)
    #! =====================================================================
    depth_temporal_attn = None
    if cfg.DEPTH.ENABLED:
        depth_sa_block = TransformerEncoder(
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
        depth_temporal_attn = nn.Sequential(
            *[dcp(depth_sa_block) for _ in range(cfg.TA.SA_LAYER_NUM)]
        ).to(device)
        logger.info(f"==> Depth Temporal Self-Attention: ENABLED")
        logger.info(f"    {cfg.TA.SA_LAYER_NUM} layers, hid={cfg.TA.HIDDEN_DIM}, heads={cfg.TA.HEAD_NUM}")
    else:
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

    #! =====================================================================
    #! Idea 4 — Phase-variance fg/bg gate (inference-time, ~2 trainable scalars)
    #! =====================================================================
    phase_variance_gate = None
    if getattr(cfg, 'PHASE_VARIANCE_GATE', None) is not None and cfg.PHASE_VARIANCE_GATE.ENABLED:
        phase_variance_gate = PhaseVarianceGate(
            alpha_init=cfg.PHASE_VARIANCE_GATE.ALPHA_INIT,
            beta_init =cfg.PHASE_VARIANCE_GATE.BETA_INIT,
            learnable =cfg.PHASE_VARIANCE_GATE.LEARNABLE,
        ).to(device)
        logger.info(f"==> Phase-Variance FG/BG Gate (Idea 4): ENABLED  "
                    f"(α_init={cfg.PHASE_VARIANCE_GATE.ALPHA_INIT}, "
                    f"β_init={cfg.PHASE_VARIANCE_GATE.BETA_INIT}, "
                    f"learnable={cfg.PHASE_VARIANCE_GATE.LEARNABLE})")
    else:
        logger.info(f"==> Phase-Variance FG/BG Gate (Idea 4): DISABLED")

    #! =====================================================================
    #! Idea 1 — AV phase coherence loss flag (no module; just an auxiliary loss)
    #! =====================================================================
    if getattr(cfg, 'AV_PHASE_COHERENCE', None) is not None and cfg.AV_PHASE_COHERENCE.ENABLED:
        logger.info(f"==> AV Phase Coherence Loss (Idea 1): ENABLED  "
                    f"(weight={cfg.AV_PHASE_COHERENCE.WEIGHT}, "
                    f"window={cfg.AV_PHASE_COHERENCE.WINDOW}, "
                    f"λ_bg={cfg.AV_PHASE_COHERENCE.LAMBDA_BG})")
    else:
        logger.info(f"==> AV Phase Coherence Loss (Idea 1): DISABLED")


    #! =====================================================================
    #! COMPLEX TEMPORAL ATTENTION (2 ComplexTransformerEncoders: visual + audio)
    #! =====================================================================
    complex_visual_temporal_attn = None
    complex_audio_temporal_attn = None
    if cfg.COMPLEX_TEMPORAL.ENABLED:
        complex_vis_block = ComplexTransformerEncoder(
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
        complex_visual_temporal_attn = nn.Sequential(
            *[dcp(complex_vis_block) for _ in range(cfg.TA.SA_LAYER_NUM)]
        ).to(device)
        complex_aud_block = ComplexTransformerEncoder(
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
        complex_audio_temporal_attn = nn.Sequential(
            *[dcp(complex_aud_block) for _ in range(cfg.TA.SA_LAYER_NUM)]
        ).to(device)
        logger.info(f"==> Complex Temporal Attention: ENABLED")
        logger.info(f"    Visual: ComplexTransformerEncoder(real=real_iHSV, imag=imag_iHSV)")
        logger.info(f"    Audio:  ComplexTransformerEncoder(real=mel, imag=CycleGAN_phase)")
        logger.info(f"    {cfg.TA.SA_LAYER_NUM} layers, hid={cfg.TA.HIDDEN_DIM}, heads={cfg.TA.HEAD_NUM}")

    #! =====================================================================
    #! COMPLEX REPRESENTATION MODE LOGGING
    #! =====================================================================
    if cfg.COMPLEX_REPR.ENABLED:
        logger.info(f"==> COMPLEX REPRESENTATION MODE: ENABLED (FCCNs-style)")
        logger.info(f"    Vision: RGB → HSV → Complex → hsv_to_rgb(real) & hsv_to_rgb(imag) → ViT-L")
        if cfg.COMPLEX_REPR.AUDIO_MODE == 'fft_real_imag':
            logger.info(f"    Audio:  waveform → FFT → |real| → mel & |imag| → mel → AudioEnc")
            logger.info(f"    No PhaseToMelProjector needed — audio imag mel is already [1, 128, 204]")
        else:
            logger.info(f"    Audio:  standard magnitude mel → AudioEnc (real) + phase IF/GD → proj → AudioEnc (imag)")
            logger.info(f"    PhaseToMelProjector: phase [1,257,204] → mel [1,128,204] → frozen AudioEnc")

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
    # Add phase model parameters if enabled
    if phase_model is not None:
        if isinstance(phase_model, PhaseAudioLoRA):
            print(f"PhaseAudioLoRA trainable parameters ({phase_model.num_trainable_params():,} total):")
            for p in phase_model.trainable_parameters():
                print(f"  [shape={list(p.shape)}]")
            trainable_params = trainable_params + phase_model.trainable_parameters()
        else:
            print("Phase Encoder trainable parameters:")
            for name, params in phase_model.named_parameters():
                print(f"  phase_encoder.{name}")
            trainable_params = trainable_params + list(phase_model.parameters())
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
    if depth_temporal_attn is not None:
        print("Depth Temporal Self-Attention trainable parameters:")
        for name, params in depth_temporal_attn.named_parameters():
            print(f"  depth_temporal_attn.{name}")
        trainable_params = trainable_params + list(depth_temporal_attn.parameters())
    if ihsv_temporal_attn is not None:
        print("iHSV Temporal Self-Attention trainable parameters:")
        for name, params in ihsv_temporal_attn.named_parameters():
            print(f"  ihsv_temporal_attn.{name}")
        trainable_params = trainable_params + list(ihsv_temporal_attn.parameters())
    if complex_visual_temporal_attn is not None:
        print("Complex Visual Temporal Attention trainable parameters:")
        for name, params in complex_visual_temporal_attn.named_parameters():
            print(f"  complex_visual_temporal_attn.{name}")
        trainable_params = trainable_params + list(complex_visual_temporal_attn.parameters())
    if complex_audio_temporal_attn is not None:
        print("Complex Audio Temporal Attention trainable parameters:")
        for name, params in complex_audio_temporal_attn.named_parameters():
            print(f"  complex_audio_temporal_attn.{name}")
        trainable_params = trainable_params + list(complex_audio_temporal_attn.parameters())
    if phase_variance_gate is not None and cfg.PHASE_VARIANCE_GATE.LEARNABLE:
        print("Phase-Variance Gate trainable parameters:")
        for name, _ in phase_variance_gate.named_parameters():
            print(f"  phase_variance_gate.{name}")
        trainable_params = trainable_params + list(phase_variance_gate.parameters())
    if phase_aware_logit is not None and any(p.requires_grad for p in phase_aware_logit.parameters()):
        print("Phase-Aware Logit trainable parameters:")
        for name, _ in phase_aware_logit.named_parameters():
            print(f"  phase_aware_logit.{name}")
        trainable_params = trainable_params + list(phase_aware_logit.parameters())

    # -----------------------------------------------------------------------
    # Modality-balance loss: optional learnable / per_class phase prototypes.
    # 'fixed' mode uses cfg constants; 'learnable' and 'per_class' need
    # nn.Parameter objects registered with the optimizer here.
    # -----------------------------------------------------------------------
    mbl_learnable_event = None
    mbl_learnable_bg    = None
    mbl_per_class       = None
    _mbl_cfg = getattr(cfg, 'MODALITY_BALANCE_LOSS', None)
    if _mbl_cfg is not None and _mbl_cfg.ENABLED:
        _mbl_mode = getattr(_mbl_cfg, 'MODE', 'fixed')
        if _mbl_mode == 'learnable':
            mbl_learnable_event = nn.Parameter(
                torch.tensor(float(_mbl_cfg.EVENT_TARGET), dtype=torch.float32, device=device))
            mbl_learnable_bg = nn.Parameter(
                torch.tensor(float(_mbl_cfg.BG_TARGET), dtype=torch.float32, device=device))
            trainable_params = trainable_params + [mbl_learnable_event, mbl_learnable_bg]
            logger.info(f"==> MBL learnable θ: event init={float(_mbl_cfg.EVENT_TARGET):+.4f} rad, "
                        f"bg init={float(_mbl_cfg.BG_TARGET):+.4f} rad (2 trainable scalars)")
        elif _mbl_mode == 'per_class':
            # One learnable scalar per event class (indices 0..bg-1).
            num_event_classes = cfg.TRAIN_BG_CLASS_ID  # event ids are 0..bg-1
            mbl_per_class = nn.Parameter(
                torch.full((num_event_classes + 1,),
                           float(_mbl_cfg.EVENT_TARGET),
                           dtype=torch.float32, device=device))
            # Note: index bg is unused for event-target, but kept in vector so
            #       gather(idx=full_label) works without re-indexing.
            trainable_params = trainable_params + [mbl_per_class]
            logger.info(f"==> MBL per-class θ: {num_event_classes} learnable scalars, "
                        f"init={float(_mbl_cfg.EVENT_TARGET):+.4f} rad each")
        else:
            logger.info(f"==> MBL fixed θ: event={float(_mbl_cfg.EVENT_TARGET):+.4f} rad, "
                        f"bg={float(_mbl_cfg.BG_TARGET):+.4f} rad")

    optimizer = torch.optim.Adam(trainable_params, lr=args.lr)
    scheduler = MultiStepLR(optimizer, milestones=[10, 20, 30], gamma=0.5)

    """loss"""
    criterion = nn.BCEWithLogitsLoss().cuda()
    # Focal-loss branch disabled — using plain CrossEntropy (matches original baseline).
    # if getattr(cfg, 'LOSS', None) is not None and cfg.LOSS.TYPE == 'focal':
    #     criterion_event = MulticlassFocalLoss(
    #         gamma=cfg.LOSS.FOCAL_GAMMA,
    #         alpha=cfg.LOSS.FOCAL_ALPHA,
    #     ).cuda()
    #     logger.info(f"==> Classification loss: FOCAL  (gamma={cfg.LOSS.FOCAL_GAMMA}, "
    #                 f"alpha={'None' if cfg.LOSS.FOCAL_ALPHA is None else 'per-class'})")
    # else:
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
            if depth_temporal_attn is not None and 'depth_temporal_attn' in ckpt:
                depth_temporal_attn.load_state_dict(ckpt['depth_temporal_attn'])
                logger.info("  Loaded depth_temporal_attn from checkpoint")
            if phase_model is not None and 'phase_model' in ckpt:
                phase_model.load_state_dict(ckpt['phase_model'])
                logger.info("  Loaded phase_model from checkpoint")
            if phase_projector is not None and 'phase_projector' in ckpt:
                phase_projector.load_state_dict(ckpt['phase_projector'])
                logger.info("  Loaded phase_projector from checkpoint")
            if complex_visual_temporal_attn is not None and 'complex_visual_temporal_attn' in ckpt:
                complex_visual_temporal_attn.load_state_dict(ckpt['complex_visual_temporal_attn'])
                logger.info("  Loaded complex_visual_temporal_attn from checkpoint")
            if complex_audio_temporal_attn is not None and 'complex_audio_temporal_attn' in ckpt:
                complex_audio_temporal_attn.load_state_dict(ckpt['complex_audio_temporal_attn'])
                logger.info("  Loaded complex_audio_temporal_attn from checkpoint")
            if phase_variance_gate is not None and 'phase_variance_gate' in ckpt:
                phase_variance_gate.load_state_dict(ckpt['phase_variance_gate'])
                logger.info("  Loaded phase_variance_gate from checkpoint")
        else:
            mainModel.load_state_dict(ckpt)
    elif args.resume != "" and (not os.path.isfile(args.resume)):
        print(f"resume argument: {args.resume}")
        raise FileNotFoundError


    '''Only Evaluate'''
    if args.evaluate:
        logger.info(f"\nStart testing..")
        _score, dump = validate_epoch(mainModel, fusionCNN, test_dataloader, criterion, criterion_event, epoch=0, eval_only=True, use_real_cnn=USE_REAL_CNN,
                       phase_model=phase_model, phase_projector=phase_projector, phase_temporal_attn=phase_temporal_attn,
                       pix2pix_generator=pix2pix_generator, pix2pix_normalizer=pix2pix_normalizer,
                       cyclegan_generator=cyclegan_generator, cyclegan_if_range=cyclegan_if_range, cyclegan_mel_range=cyclegan_mel_range,
                       depth_temporal_attn=depth_temporal_attn, ihsv_temporal_attn=ihsv_temporal_attn,
                       complex_visual_temporal_attn=complex_visual_temporal_attn, complex_audio_temporal_attn=complex_audio_temporal_attn,
                       phase_variance_gate=phase_variance_gate,
                       log_pi_total=log_pi_total,
                       collect=True,
                       phase_aware_logit=phase_aware_logit)
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
                           phase_model=phase_model, phase_projector=phase_projector, phase_temporal_attn=phase_temporal_attn,
                           pix2pix_generator=pix2pix_generator, pix2pix_normalizer=pix2pix_normalizer,
                           cyclegan_generator=cyclegan_generator, cyclegan_if_range=cyclegan_if_range, cyclegan_mel_range=cyclegan_mel_range,
                           depth_temporal_attn=depth_temporal_attn, ihsv_temporal_attn=ihsv_temporal_attn,
                           complex_visual_temporal_attn=complex_visual_temporal_attn, complex_audio_temporal_attn=complex_audio_temporal_attn,
                           phase_variance_gate=phase_variance_gate,
                           phase_aware_logit=phase_aware_logit,
                           mbl_learnable_event=mbl_learnable_event, mbl_learnable_bg=mbl_learnable_bg,
                           mbl_per_class=mbl_per_class)

        if ((epoch) % args.eval_freq == 0) or (epoch == args.n_epoch - 1):
            # acc = validate_epoch(mainModel, val_dataloader, criterion, criterion_event, epoch)
            acc, dump = validate_epoch(mainModel, fusionCNN, test_dataloader, criterion, criterion_event, epoch, use_real_cnn=USE_REAL_CNN,
                                 phase_model=phase_model, phase_projector=phase_projector, phase_temporal_attn=phase_temporal_attn,
                                 pix2pix_generator=pix2pix_generator, pix2pix_normalizer=pix2pix_normalizer,
                                 cyclegan_generator=cyclegan_generator, cyclegan_if_range=cyclegan_if_range, cyclegan_mel_range=cyclegan_mel_range,
                                 depth_temporal_attn=depth_temporal_attn, ihsv_temporal_attn=ihsv_temporal_attn,
                           complex_visual_temporal_attn=complex_visual_temporal_attn, complex_audio_temporal_attn=complex_audio_temporal_attn,
                           phase_variance_gate=phase_variance_gate,
                           log_pi_total=log_pi_total,
                           collect=True,
                           phase_aware_logit=phase_aware_logit) # may change to test
            if acc > best_accuracy:
                best_accuracy = acc
                best_dump = dump
                logger.info(f'best average result of three metrics at epoch-{epoch}: {best_accuracy:.4f}')
                best_accuracy_epoch = epoch
                checkpoint_dict = {'mainModel': mainModel.state_dict()}
                if fusionCNN is not None:
                    checkpoint_dict['fusionCNN'] = fusionCNN.state_dict()
                if phase_model is not None:
                    checkpoint_dict['phase_model'] = phase_model.state_dict()
                if phase_projector is not None:
                    checkpoint_dict['phase_projector'] = phase_projector.state_dict()
                if phase_temporal_attn is not None:
                    checkpoint_dict['phase_temporal_attn'] = phase_temporal_attn.state_dict()
                if depth_temporal_attn is not None:
                    checkpoint_dict['depth_temporal_attn'] = depth_temporal_attn.state_dict()
                if ihsv_temporal_attn is not None:
                    checkpoint_dict['ihsv_temporal_attn'] = ihsv_temporal_attn.state_dict()
                if complex_visual_temporal_attn is not None:
                    checkpoint_dict['complex_visual_temporal_attn'] = complex_visual_temporal_attn.state_dict()
                if complex_audio_temporal_attn is not None:
                    checkpoint_dict['complex_audio_temporal_attn'] = complex_audio_temporal_attn.state_dict()
                if phase_variance_gate is not None:
                    checkpoint_dict['phase_variance_gate'] = phase_variance_gate.state_dict()
                if phase_aware_logit is not None:
                    checkpoint_dict['phase_aware_logit'] = phase_aware_logit.state_dict()
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
                phase_model=None, phase_projector=None, phase_temporal_attn=None,
                pix2pix_generator=None, pix2pix_normalizer=None,
                cyclegan_generator=None, cyclegan_if_range=None, cyclegan_mel_range=None,
                depth_temporal_attn=None, ihsv_temporal_attn=None,
                complex_visual_temporal_attn=None, complex_audio_temporal_attn=None,
                phase_variance_gate=None,
                phase_aware_logit=None,
                mbl_learnable_event=None, mbl_learnable_bg=None, mbl_per_class=None):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    train_acc = AverageMeter()
    train_seg_fscore = AverageMeter()
    train_eve_fscore = AverageMeter()
    end_time = time.time()

    # Track gate values and phase signals for AlignedTemporalDiffComplexAttention
    gate_audio_values = []
    gate_visual_values = []
    phase_audio_pre_values = []
    phase_audio_post_values = []
    phase_visual_pre_values = []
    phase_visual_post_values = []
    # Track phase coherence stats (for PhaseAwareComplexAttention with local_mean)
    phase_coherence_event_values = []
    phase_coherence_bg_values = []
    phase_coherence_boundary_grad_values = []

    model.train()
    if fusion_cnn is not None:
        fusion_cnn.train()
    if phase_model is not None:
        phase_model.train()
    if phase_projector is not None:
        phase_projector.train()
    if phase_temporal_attn is not None:
        phase_temporal_attn.train()
    if depth_temporal_attn is not None:
        depth_temporal_attn.train()
    if ihsv_temporal_attn is not None:
        ihsv_temporal_attn.train()
    if complex_visual_temporal_attn is not None:
        complex_visual_temporal_attn.train()
    if complex_audio_temporal_attn is not None:
        complex_audio_temporal_attn.train()
    if phase_variance_gate is not None:
        phase_variance_gate.train()
    # Note: here we set the model to a double type precision,
    # since the extracted features are in a double type.
    # This will also lead to the size of the model double increases.
    # model.double()
    optimizer.zero_grad()

    for n_iter, batch_data in enumerate(train_dataloader):

        data_time.update(time.time() - end_time)
        '''Feed input to model'''
        depth_inputs = None
        if cfg.COMPLEX_REPR.ENABLED:
            audio, visual, visual_imag, text, full_label, avc_label, category_label, vid_name, audio_phase_data = batch_data
            visual_imag_inputs = visual_imag.cuda()
            if cfg.COMPLEX_REPR.AUDIO_MODE == 'fft_real_imag':
                #! audio_phase_data is audio_imag mel [bs, 10, 1, 128, 204]
                audio_imag_inputs = audio_phase_data.squeeze(1).cuda()
                phase_raw = None
            else:
                #! audio_phase_data is phase IF/GD [bs, 10, 1, 257, 204]
                phase_raw = audio_phase_data.squeeze(1).cuda()
                audio_imag_inputs = None
        elif cfg.COMPLEX_TEMPORAL.ENABLED:
            audio, visual, visual_imag, text, full_label, avc_label, category_label, vid_name, phase_raw = batch_data
            visual_imag_inputs = visual_imag.cuda()
            phase_raw = phase_raw.squeeze(1).cuda()
            audio_imag_inputs = None
        elif cfg.DEPTH.ENABLED and cfg.PHASE.ENABLED:
            audio, visual, depth, text, full_label, avc_label, category_label, vid_name, phase_raw = batch_data
            depth_inputs = depth.cuda()  # [bs, 10, 1, 224, 224]
            visual_imag_inputs = None
            phase_raw = phase_raw.squeeze(1).cuda()
            audio_imag_inputs = None
        elif cfg.DEPTH.ENABLED:
            audio, visual, depth, text, full_label, avc_label, category_label, vid_name = batch_data
            depth_inputs = depth.cuda()  # [bs, 10, 1, 224, 224]
            visual_imag_inputs = None
            phase_raw = None
            audio_imag_inputs = None
        elif cfg.IHSV.ENABLED and cfg.PHASE.ENABLED:
            audio, visual, visual_imag, text, full_label, avc_label, category_label, vid_name, phase_raw = batch_data
            visual_imag_inputs = visual_imag.cuda()
            phase_raw = phase_raw.squeeze(1).cuda()
            audio_imag_inputs = None
        elif cfg.IHSV.ENABLED:
            audio, visual, visual_imag, text, full_label, avc_label, category_label, vid_name = batch_data
            visual_imag_inputs = visual_imag.cuda()
            phase_raw = None
            audio_imag_inputs = None
        elif cfg.PHASE.ENABLED:
            audio, visual, text, full_label, avc_label, category_label, vid_name, phase_raw = batch_data
            visual_imag_inputs = None
            phase_raw = phase_raw.squeeze(1).cuda()  # [bs, 10, 1, 257, 204]
            audio_imag_inputs = None
        else:
            audio, visual, text, full_label, avc_label, category_label, vid_name = batch_data
            visual_imag_inputs = None
            phase_raw = None
            audio_imag_inputs = None
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
        if depth_inputs is not None:
            inputs[ModalityType.DEPTH] = depth_inputs  # [bs, 10, 1, 224, 224]
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

        #! DEPTH: depth x text similarity (replaces iHSV when DEPTH.ENABLED)
        simm_dt = None
        if ModalityType.DEPTH in embeddings:
            depth_feas = embeddings[ModalityType.DEPTH]  # [bs, 10, 1024]
            simm_dt = compute_cross_modal_similarity(depth_feas, text_feas)  # [bs, 10, C+1]

        #! =====================================================================
        #! AUDIO IMAGINARY / PHASE FEATURE COMPUTATION
        #! =====================================================================
        simm_pt = None
        if cfg.COMPLEX_REPR.ENABLED and audio_imag_inputs is not None:
            #! Complex repr mode: audio imag mel is already [bs, 10, 1, 128, 204]
            #! Feed directly through frozen audio encoder — no projector needed
            audio_imag_input = {ModalityType.AUDIO: audio_imag_inputs}
            with torch.no_grad():
                audio_imag_embeddings = model(audio_imag_input)
            audio_imag_feas = audio_imag_embeddings['audio']  # [bs, 10, 1024]

            # Temporal self-attention (trainable)
            if phase_temporal_attn is not None:
                for blk in phase_temporal_attn:
                    audio_imag_feas = blk(audio_imag_feas, audio_imag_feas, audio_imag_feas)

            # Audio-imag × text similarity
            simm_pt = compute_cross_modal_similarity(audio_imag_feas, text_feas)  # [bs, 10, C+1]

        elif phase_raw is not None and phase_model is not None and isinstance(phase_model, PhaseAudioLoRA):
            #! LoRA phase pipeline: phase IF → projector → LoRA-adapted audio encoder → temporal attn
            B_ph, T_ph = phase_raw.shape[:2]
            phase_flat = phase_raw.reshape(B_ph * T_ph, *phase_raw.shape[2:])  # [bs*10, 1, 257, 204]
            # Compute IF (instantaneous frequency)
            phase_if = compute_if(phase_flat)  # [bs*10, 1, 257, 204]

            # Step 1: Project IF to mel shape
            phase_mel = phase_projector(phase_if)  # [bs*10, 1, 128, 204]
            phase_mel_5d = phase_mel.reshape(B_ph, T_ph, 1, 128, 204)  # [bs, 10, 1, 128, 204]

            # Step 2: Feed through LoRA-adapted audio encoder (trainable adapters + stem)
            phase_feas = phase_model(phase_mel_5d)  # [bs, 10, 1024]

            # Step 3: Phase temporal self-attention (trainable)
            if phase_temporal_attn is not None:
                for blk in phase_temporal_attn:
                    phase_feas = blk(phase_feas, phase_feas, phase_feas)

            # Step 4: Phase-text similarity
            simm_pt = compute_cross_modal_similarity(phase_feas, text_feas)  # [bs, 10, C+1]

        elif phase_raw is not None and cfg.PHASE.MODE == 'hilbert':
            #! Hilbert pipeline: hilbert imag [bs, 10, 1, 128, 204] → frozen audio encoder → temporal attn
            #! No projector needed — already mel-shaped
            hilbert_input = {ModalityType.AUDIO: phase_raw}  # [bs, 10, 1, 128, 204]
            with torch.no_grad():
                hilbert_embeddings = model(hilbert_input)
            phase_feas = hilbert_embeddings['audio']  # [bs, 10, 1024]

            # Temporal self-attention (trainable)
            if phase_temporal_attn is not None:
                for blk in phase_temporal_attn:
                    phase_feas = blk(phase_feas, phase_feas, phase_feas)

            # Hilbert-text similarity
            simm_pt = compute_cross_modal_similarity(phase_feas, text_feas)  # [bs, 10, C+1]

        elif phase_raw is not None and cfg.PHASE.MODE == 'pix2pix':
            #! Pix2Pix pipeline: phase → IF → Pix2Pix GAN (frozen) → mel-like → frozen audio encoder → temporal attn
            B_ph, T_ph = phase_raw.shape[:2]
            phase_flat = phase_raw.reshape(B_ph * T_ph, *phase_raw.shape[2:])  # [bs*10, 1, 257, 204]

            # Step 1: Compute IF (instantaneous frequency)
            phase_if = compute_if(phase_flat)  # [bs*10, 1, 257, 204]

            # Step 2: Translate IF → mel-like via frozen Pix2Pix generator
            with torch.no_grad():
                phase_input_norm = phase_if / torch.pi  # normalize to [-1, 1]
                phase_mel_norm = pix2pix_generator(phase_input_norm)  # [bs*10, 1, 128, 204]
                phase_mel = pix2pix_normalizer.denormalize(phase_mel_norm)  # restore mel distribution
                phase_mel = apply_imagebind_audio_norm(phase_mel)  # FIX: re-apply Normalize that load_and_transform_audio_data uses

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

        elif phase_raw is not None and cfg.PHASE.MODE == 'cyclegan':
            #! Vanilla CycleGAN pipeline: phase IF → G_IF2Mel (frozen, unconditional) → mel-like → frozen audio encoder → temporal attn
            # TRAIN-time phase corruption. Eval-time controls showed the
            # converged model is insensitive to phase *content*; this asks whether the
            # +1.5 Eve-F1 that the phase stream contributes during TRAINING requires
            # the phase to be the clip's own. 'noise' trains on G(uniform random phase):
            # if it matches the real run, the stream is a content-free regulariser.
            _tm = getattr(cfg.PHASE, 'TRAIN_MISMATCH', 'none')
            if _tm == 'noise':
                phase_raw = (torch.rand_like(phase_raw) * 2.0 - 1.0) * torch.pi
            elif _tm != 'none':
                raise ValueError(f"unknown cfg.PHASE.TRAIN_MISMATCH={_tm!r}")
            B_ph, T_ph = phase_raw.shape[:2]
            phase_flat = phase_raw.reshape(B_ph * T_ph, *phase_raw.shape[2:])  # [bs*10, 1, 257, 204]

            # Step 1: Compute IF (instantaneous frequency)
            phase_if = compute_if(phase_flat)  # [bs*10, 1, 257, 204]

            # Step 2: Translate IF → mel-like via frozen vanilla CycleGAN generator
            with torch.no_grad():
                phase_mel = translate_phase_to_mel(
                    cyclegan_generator, cyclegan_if_range, cyclegan_mel_range, phase_if
                )  # [bs*10, 1, 128, 204]
                phase_mel = apply_imagebind_audio_norm(phase_mel)  # FIX: re-apply Normalize that load_and_transform_audio_data uses

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

        elif phase_raw is not None and cfg.PHASE.MODE == 'cond_cyclegan':
            #! Conditional CycleGAN pipeline: (phase IF, mel_cond) → G_IF2Mel(IF, mel_cond) (frozen) → mel-like → frozen audio encoder → temporal attn
            B_ph, T_ph = phase_raw.shape[:2]
            phase_flat = phase_raw.reshape(B_ph * T_ph, *phase_raw.shape[2:])  # [bs*10, 1, 257, 204]
            mel_cond = audio_inputs.reshape(B_ph * T_ph, 1, 128, 204)            # [bs*10, 1, 128, 204]

            # Step 1: Compute IF (instantaneous frequency)
            phase_if = compute_if(phase_flat)  # [bs*10, 1, 257, 204]

            # Step 2: Translate (IF, mel_cond) → mel-like via frozen conditional CycleGAN generator
            with torch.no_grad():
                phase_mel = translate_phase_to_mel_cond(
                    cyclegan_generator, cyclegan_if_range, cyclegan_mel_range, phase_if, mel_cond, device=phase_if.device
                )  # [bs*10, 1, 128, 204]
                phase_mel = apply_imagebind_audio_norm(phase_mel)  # FIX: re-apply Normalize that load_and_transform_audio_data uses

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
            if cfg.PHASE.INPUT_TYPE == 'if_gd':
                phase_flat = compute_if_gd(phase_flat)
                phase_for_proj = phase_flat[:, 0:1, :, :]
            elif cfg.PHASE.INPUT_TYPE == 'if':
                phase_for_proj = compute_if(phase_flat)
            else:
                phase_for_proj = phase_flat

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

        #! DISABLED: Compute audio-visual weights via element-wise product of audio and visual feature vector
        # av_product = audio_feas * visual_feas  # [bs, 10, 1024]
        # av_weights = av_product.sum(dim=-1)  # [bs, 10]
        # av_weights_max = av_weights.max(dim=-1, keepdim=True)[0]  # [bs, 1]
        # av_weights = av_weights / (av_weights_max + 1e-8)  # [bs, 10]
        # av_weights = av_weights.unsqueeze(-1)  # [bs, 10, 1]


        #! =====================================================================
        #! COMPLEX TEMPORAL ATTENTION MODE
        #! =====================================================================
        if cfg.COMPLEX_TEMPORAL.ENABLED and complex_visual_temporal_attn is not None and complex_audio_temporal_attn is not None:
            imag_visual_feas = embeddings.get(ModalityType.VISION_IMAG)
            if imag_visual_feas is None:
                imag_visual_feas = torch.zeros_like(visual_feas)

            # Get phase features (without separate temporal attn — complex temporal attn handles it)
            phase_feas_for_complex = torch.zeros_like(audio_feas)
            if phase_raw is not None and cfg.PHASE.MODE == 'cyclegan':
                B_ph, T_ph = phase_raw.shape[:2]
                phase_flat = phase_raw.reshape(B_ph * T_ph, *phase_raw.shape[2:])
                phase_if = compute_if(phase_flat)
                with torch.no_grad():
                    phase_mel = translate_phase_to_mel(
                        cyclegan_generator, cyclegan_if_range, cyclegan_mel_range, phase_if)
                    phase_mel = apply_imagebind_audio_norm(phase_mel)  # FIX: re-apply Normalize that load_and_transform_audio_data uses
                phase_input = {ModalityType.AUDIO: phase_mel.reshape(B_ph, T_ph, 1, 128, 204)}
                with torch.no_grad():
                    phase_embeddings = model(phase_input)
                phase_feas_for_complex = phase_embeddings['audio']
            elif phase_raw is not None and cfg.PHASE.MODE == 'cond_cyclegan':
                B_ph, T_ph = phase_raw.shape[:2]
                phase_flat = phase_raw.reshape(B_ph * T_ph, *phase_raw.shape[2:])
                mel_cond = audio_inputs.reshape(B_ph * T_ph, 1, 128, 204)
                phase_if = compute_if(phase_flat)
                with torch.no_grad():
                    phase_mel = translate_phase_to_mel_cond(
                        cyclegan_generator, cyclegan_if_range, cyclegan_mel_range, phase_if, mel_cond, device=phase_if.device)
                    phase_mel = apply_imagebind_audio_norm(phase_mel)  # FIX: re-apply Normalize that load_and_transform_audio_data uses
                phase_input = {ModalityType.AUDIO: phase_mel.reshape(B_ph, T_ph, 1, 128, 204)}
                with torch.no_grad():
                    phase_embeddings = model(phase_input)
                phase_feas_for_complex = phase_embeddings['audio']

            # Complex Visual Temporal Attention: real=real_iHSV, imag=imag_iHSV
            vis_real, vis_imag = visual_feas, imag_visual_feas
            for blk in complex_visual_temporal_attn:
                vis_real, vis_imag = blk(vis_real, vis_imag)

            # Complex Audio Temporal Attention: real=mel_audio, imag=cyclegan_phase
            aud_real, aud_imag = audio_feas, phase_feas_for_complex
            for blk in complex_audio_temporal_attn:
                aud_real, aud_imag = blk(aud_real, aud_imag)

            # 4 similarity matrices from complex temporal attention outputs
            simm_vt = compute_cross_modal_similarity(vis_real, text_feas)
            simm_imag_vt = compute_cross_modal_similarity(vis_imag, text_feas)
            simm_at = compute_cross_modal_similarity(aud_real, text_feas)
            simm_pt = compute_cross_modal_similarity(aud_imag, text_feas)
        else:
            #! GEN1: generate AVE prediction logits (without AV similarity modulation)
            simm_at = compute_cross_modal_similarity(audio_feas, text_feas) # [bs, 10, 46+1]
            simm_vt = compute_cross_modal_similarity(visual_feas, text_feas) # [bs, 10, 46+1]

        #! DISABLED: Apply AV weights to audio-text and video-text similarities
        # simm_at = simm_at * av_weights  # [bs, 10, C+1]
        # simm_vt = simm_vt * av_weights  # [bs, 10, C+1]

        #! =====================================================================
        #! FUSION: RealCNN, ComplexCNN, or simple average (when disabled)
        #! =====================================================================
        # Normalize all available similarity matrices
        simm_vt_norm = min_max_normalize(simm_vt, dim=-1)  # [bs, 10, C+1] -> [0, 1]
        simm_at_norm = min_max_normalize(simm_at, dim=-1)  # [bs, 10, C+1] -> [0, 1]
        simm_imag_vt_norm = min_max_normalize(simm_imag_vt, dim=-1) if simm_imag_vt is not None else None
        simm_dt_norm = min_max_normalize(simm_dt, dim=-1) if simm_dt is not None else None
        simm_pt_norm = min_max_normalize(simm_pt, dim=-1) if simm_pt is not None else None
        # Use depth as visual-imaginary channel when available, else fall back to iHSV
        simm_vis_imag_norm = simm_dt_norm if simm_dt_norm is not None else simm_imag_vt_norm

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
            #! ComplexCNN: 2 complex channels in complex_pair mode (visual / audio domains)
            _use_fccn = (getattr(cfg, 'FCCN_LOSS', None) is not None
                         and cfg.FCCN_LOSS.ENABLED)
            _use_pal  = (phase_aware_logit is not None)
            _use_mbl  = (getattr(cfg, 'MODALITY_BALANCE_LOSS', None) is not None
                         and cfg.MODALITY_BALANCE_LOSS.ENABLED)
            _use_psl  = (getattr(cfg, 'PHASE_SPREAD_LOSS', None) is not None
                         and cfg.PHASE_SPREAD_LOSS.ENABLED)
            if _use_fccn or _use_pal or _use_mbl or _use_psl:
                pred_logits, fusion_phase = fusion_cnn(
                    simm_vt_norm, simm_at_norm,
                    simm_vis_imag_norm, simm_pt_norm,
                    return_phase=True)                          # [bs,10,C+1] each
            else:
                pred_logits = fusion_cnn(simm_vt_norm, simm_at_norm,
                                         simm_vis_imag_norm, simm_pt_norm)  # [bs, 10, C+1]
                fusion_phase = None
            # Phase-aware logit gating BEFORE min-max normalization.
            # Suppresses logits at slots where phase disagrees with class-target.
            if _use_pal:
                pred_logits = phase_aware_logit(pred_logits, fusion_phase)
            pred_logits = min_max_normalize(pred_logits, dim=-1)  # output min-max norm

        #! Apply temporal smoothing on predictions (Gaussian smoothing along time axis)
        if cfg.SMOOTH.ENABLED:
            pred_logits = temporal_smooth(pred_logits,
                                          kernel_size=cfg.SMOOTH.KERNEL_SIZE,
                                          sigma=cfg.SMOOTH.SIGMA)

        # # OLD ComplexCNN code (commented out for reference - train):
        # #! Case 4: input(min-max) -> complex -> mag -> min-max -> CE
        # simm_vt_norm = min_max_normalize(simm_vt, dim=-1)  # [bs, 10, C+1] -> [0, 1]
        # simm_at_norm = min_max_normalize(simm_at, dim=-1)  # [bs, 10, C+1] -> [0, 1]
        # pred_logits = complex_cnn(simm_vt_norm, simm_at_norm)  # [bs, 10, C+1]
        # pred_logits = min_max_normalize(pred_logits, dim=-1)  # output min-max norm

        #! Case 1: input(raw) -> complex -> mag -> min-max -> CE
        # pred_logits = complex_cnn(simm_vt, simm_at)  # [bs, 10, C+1]
        # pred_logits = min_max_normalize(pred_logits, dim=-1)  # output min-max norm

        #! Case 2: input(raw) -> complex -> mag -> sigmoid -> CE
        # pred_logits = complex_cnn(simm_vt, simm_at)  # [bs, 10, C+1]
        # pred_logits = torch.sigmoid(pred_logits)

        #! Case 3: input(raw) -> complex -> mag -> CE (no normalization)
        # pred_logits = complex_cnn(simm_vt, simm_at)  # [bs, 10, C+1]

        # pdb.set_trace()
        #! OLD GEN1: sqrt(relu) fusion (replaced by ComplexCNN)
        # pred_logits = torch.sqrt(F.relu(simm_at * simm_vt)) # [bs, 10, 46+1]
        #! GEN2: generate AVE prediction logits
        # simm_at = compute_cross_modal_similarity(audio_feas, text_feas) # [bs, 10, 46+1]
        # simm_vt = compute_cross_modal_similarity(visual_feas, text_feas) # [bs, 10, 46+1]
        # pred_logits = (simm_at + simm_vt) * 0.5 # [bs, 10, 46+1]
        #! GEN3: generate AVE prediction logits
        # av_feas = (audio_feas + visual_feas) * 0.5
        # simm_avt = compute_cross_modal_similarity(av_feas, text_feas) # [bs, 10, 46+1]
        # pred_logits = simm_avt # [bs, 10, 46+1]


        if  torch.sum(pred_logits.isnan()) > 0:
            continue
            pdb.set_trace()


        # Boundary-weighted CE re-weights frames at event onsets/offsets.
        # Targets the boundary-vs-interior accuracy gap (~12 pp in our diag).
        if cfg.BOUNDARY.ENABLED:
            loss = boundary_weighted_ce(pred_logits, full_labels, cfg.BOUNDARY.ALPHA)
        else:
            loss = criterion_event(pred_logits.permute(0, 2, 1), full_labels)

        # FCCN auxiliary phase loss — mode selected by cfg.FCCN_LOSS.MODE.
        # Active only in the ComplexCNN branch (fusion_phase is populated there).
        if 'fusion_phase' in dir() and fusion_phase is not None:
            loss = apply_fccn_aux_loss(loss, fusion_phase, full_labels, cfg)

        # Modality-balance loss (NEW) — exploits §3's finding that events live
        # at arg z ≈ π/4 in raw similarities. Pushes fused phase at GT class
        # toward π/4 for events (and optionally 0 for bg).
        if ('fusion_phase' in dir() and fusion_phase is not None
                and getattr(cfg, 'MODALITY_BALANCE_LOSS', None) is not None
                and cfg.MODALITY_BALANCE_LOSS.ENABLED):
            mb_aux = compute_modality_balance_loss(
                fusion_phase, full_labels, cfg,
                learnable_event=mbl_learnable_event,
                learnable_bg=mbl_learnable_bg,
                per_class_targets=mbl_per_class)
            loss = loss + mb_aux
            if n_iter % args.print_iter_freq == 0:
                with torch.no_grad():
                    y_idx_dbg = full_labels.long().unsqueeze(-1)
                    theta_y_dbg = fusion_phase.gather(-1, y_idx_dbg).squeeze(-1)
                    fg = (full_labels != cfg.TRAIN_BG_CLASS_ID)
                    if fg.any():
                        mu_evt = torch.atan2(torch.sin(theta_y_dbg[fg]).mean(),
                                             torch.cos(theta_y_dbg[fg]).mean())
                    else:
                        mu_evt = torch.tensor(float('nan'))
                    _mode = getattr(cfg.MODALITY_BALANCE_LOSS, 'MODE', 'fixed')
                    if _mode == 'learnable' and mbl_learnable_event is not None:
                        target_str = f"θ_evt(learn)={mbl_learnable_event.item():+.4f}"
                    elif _mode == 'per_class' and mbl_per_class is not None:
                        target_str = (f"θ_pc μ={mbl_per_class.mean().item():+.4f} "
                                      f"σ={mbl_per_class.std().item():.4f}")
                    else:
                        target_str = f"target={cfg.MODALITY_BALANCE_LOSS.EVENT_TARGET:.4f}"
                    logger.info(f"[mod-bal] aux={mb_aux.item():.4f}  μ_evt={mu_evt.item():+.3f} rad  "
                                f"({target_str})")

        # Phase-Spread Loss (NEW) — O3-faithful angular-margin loss:
        # intra-class vMF concentration + inter-class hyperspherical spread on
        # the fused phase. Open-vocab safe (no per-class params).
        if ('fusion_phase' in dir() and fusion_phase is not None
                and getattr(cfg, 'PHASE_SPREAD_LOSS', None) is not None
                and cfg.PHASE_SPREAD_LOSS.ENABLED):
            psl_aux = compute_phase_spread_loss(fusion_phase, full_labels, cfg)
            loss = loss + psl_aux
            if n_iter % args.print_iter_freq == 0:
                logger.info(f"[ph-spread] aux={psl_aux.item():.4f}  "
                            f"λ_intra={cfg.PHASE_SPREAD_LOSS.LAMBDA_INTRA}  "
                            f"λ_spread={cfg.PHASE_SPREAD_LOSS.LAMBDA_SPREAD}  "
                            f"margin={cfg.PHASE_SPREAD_LOSS.MARGIN}")

        # Idea 1: Cross-modal AV phase coherence loss (operates on raw simm tensors).
        # Pushes the temporal phase of (simm_vt + i·simm_at) at the GT class to
        # be coherent within events while keeping non-GT classes incoherent.
        if getattr(cfg, 'AV_PHASE_COHERENCE', None) is not None and cfg.AV_PHASE_COHERENCE.ENABLED:
            av_phase_loss, av_phase_stats = compute_av_phase_coherence_loss(
                simm_vt, simm_at, full_labels,
                bg_flag=cfg.TRAIN_BG_CLASS_ID,
                window_size=cfg.AV_PHASE_COHERENCE.WINDOW,
                lambda_bg=cfg.AV_PHASE_COHERENCE.LAMBDA_BG,
                margin=cfg.AV_PHASE_COHERENCE.MARGIN,
            )
            loss = loss + cfg.AV_PHASE_COHERENCE.WEIGHT * av_phase_loss
            if n_iter % args.print_iter_freq == 0:
                logger.info(f"[av-phase-coh] R_gt_event={av_phase_stats['R_gt_event']:.4f}  "
                            f"R_nongt={av_phase_stats['R_nongt']:.4f}  "
                            f"gap={av_phase_stats['coherence_gap']:.4f}")

        # Add alignment loss if using AlignedTemporalDiffComplexAttention
        if 'alignment_loss' in embeddings:
            align_loss = embeddings['alignment_loss']
            loss = loss + cfg.TA.ALIGNED_LOSS_WEIGHT * align_loss

        # Add phase coherence loss if using PhaseAwareComplexAttention
        if 'phase_info' in embeddings and cfg.TA.PHASE_COHERENCE_LOSS_ENABLED:
            phase_coherence_loss, phase_stats = compute_phase_coherence_loss(
                embeddings['phase_info'],
                avc_labels,
                window_size=cfg.TA.PHASE_COHERENCE_WINDOW
            )
            loss = loss + cfg.TA.PHASE_COHERENCE_LOSS_WEIGHT * phase_coherence_loss
            # Track phase coherence stats for monitoring
            phase_coherence_event_values.append(phase_stats['event_coherence'])
            phase_coherence_bg_values.append(phase_stats['bg_coherence'])
            phase_coherence_boundary_grad_values.append(phase_stats['boundary_gradient'])

        # Track gate values and phase signals for explainability
        if 'audio_gate' in embeddings:
            gate_audio_values.append(embeddings['audio_gate'])
            gate_visual_values.append(embeddings['visual_gate'])
        if 'explainability' in embeddings:
            explain = embeddings['explainability']
            # Store mean phase values per batch (averaged over time and batch)
            phase_audio_pre_values.append(explain['audio_phase_pre'].mean().item())
            phase_audio_post_values.append(explain['audio_phase_post'].mean().item())
            phase_visual_pre_values.append(explain['visual_phase_pre'].mean().item())
            phase_visual_post_values.append(explain['visual_phase_post'].mean().item())

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

    # Print gate values and phase signals at end of epoch
    if len(gate_audio_values) > 0:
        avg_gate_audio = np.mean(gate_audio_values)
        avg_gate_visual = np.mean(gate_visual_values)
        logger.info(f'\n{"="*60}')
        logger.info(f'Epoch {epoch} - Gate Values & Phase Signals Summary')
        logger.info(f'{"="*60}')
        logger.info(f'  Audio Gate:  {avg_gate_audio:.4f}  (>{0.5:.1f} favors original, <{0.5:.1f} favors attention)')
        logger.info(f'  Visual Gate: {avg_gate_visual:.4f}  (>{0.5:.1f} favors original, <{0.5:.1f} favors attention)')

        if len(phase_audio_pre_values) > 0:
            avg_phase_audio_pre = np.mean(phase_audio_pre_values)
            avg_phase_audio_post = np.mean(phase_audio_post_values)
            avg_phase_visual_pre = np.mean(phase_visual_pre_values)
            avg_phase_visual_post = np.mean(phase_visual_post_values)
            phase_change_audio = avg_phase_audio_post - avg_phase_audio_pre
            phase_change_visual = avg_phase_visual_post - avg_phase_visual_pre

            logger.info(f'  ----------------------------------------')
            logger.info(f'  Phase (arctan(imag/real) = "event-ness"):')
            logger.info(f'    Audio  - Pre: {avg_phase_audio_pre:.4f}, Post: {avg_phase_audio_post:.4f}, Change: {phase_change_audio:+.4f}')
            logger.info(f'    Visual - Pre: {avg_phase_visual_pre:.4f}, Post: {avg_phase_visual_post:.4f}, Change: {phase_change_visual:+.4f}')
        logger.info(f'{"="*60}\n')

    # Print phase coherence stats (from local_mean in phase domain)
    if len(phase_coherence_event_values) > 0:
        avg_event_coh = np.mean(phase_coherence_event_values)
        avg_bg_coh = np.mean(phase_coherence_bg_values)
        avg_boundary_grad = np.mean(phase_coherence_boundary_grad_values)
        logger.info(f'\n{"="*60}')
        logger.info(f'Epoch {epoch} - Phase Coherence (local_mean equivalent)')
        logger.info(f'{"="*60}')
        logger.info(f'  Event Coherence:    {avg_event_coh:.4f}  (higher = better A-V sync during events)')
        logger.info(f'  Background Coh:     {avg_bg_coh:.4f}  (lower = good discrimination)')
        logger.info(f'  Delta (E-BG):       {avg_event_coh - avg_bg_coh:+.4f}  (positive = good!)')
        logger.info(f'  Boundary Gradient:  {avg_boundary_grad:.4f}  (higher = better transition detection)')
        logger.info(f'{"="*60}\n')

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


_EVAL_MISMATCH_CACHE = {}


def _apply_eval_mismatch(phase_raw):
    """Eval-time corruption of the phase stream (content controls).

    The alternative hypothesis is that the CycleGAN emits generic mel-like texture
    that acts as augmentation rather than carrying clip-specific phase content. G
    only ever sees the IF, so that hypothesis is exactly "G's output is insensitive
    to its input" -- which these arms test end-to-end, with no retraining.

    NOTE: the test DataLoader is `shuffle=False` and META_CSV is class-ordered, so a
    within-batch permutation would hand each clip the phase of a *same-class* clip
    and spuriously show no effect. 'foreign' therefore caches the first batch's phase
    and reuses it for every later batch, guaranteeing a cross-class mismatch for all
    but the first batch.
    """
    mode = getattr(cfg.PHASE, 'EVAL_MISMATCH', 'none')
    if not _EVAL_MISMATCH_CACHE.get('logged'):
        _EVAL_MISMATCH_CACHE['logged'] = True
        logger.info(f"[eval-mismatch] mode={mode!r}  phase_raw mean={phase_raw.mean():.6f} "
                    f"std={phase_raw.std():.6f} sum={phase_raw.double().sum():.4f}")
    # 'mute' is the plumbing control: it leaves phase_raw alone and instead zeroes the
    # phase *similarity* downstream. If muting the stream does not move the metrics,
    # the trained model ignores phase and no phase-content experiment can say anything.
    if mode in ('none', 'mute'):
        return phase_raw
    if mode == 'noise':
        return (torch.rand_like(phase_raw) * 2.0 - 1.0) * torch.pi
    if mode == 'timerev':
        return torch.flip(phase_raw, dims=[-1])          # last dim = time frames
    if mode == 'foreign':
        cached = _EVAL_MISMATCH_CACHE.get('phase')
        if cached is None:
            _EVAL_MISMATCH_CACHE['phase'] = phase_raw.detach().clone()
            # First batch has no foreign donor yet; roll it so no clip keeps its own
            # phase. This batch is same-class, and is ~2% of the split.
            return torch.roll(phase_raw, shifts=1, dims=0)
        B = phase_raw.shape[0]
        reps = (B + cached.shape[0] - 1) // cached.shape[0]
        donor = cached.repeat(reps, *([1] * (cached.dim() - 1)))[:B]
        return donor.to(phase_raw.device, non_blocking=True)
    raise ValueError(f"unknown cfg.PHASE.EVAL_MISMATCH={mode!r}")


def _apply_ihsv_mismatch(feas):
    """Same content-destruction controls as _apply_eval_mismatch, for the iHSV stream.

    Applied to the pooled iHSV embeddings rather than the raw frames -- equivalent for
    the question asked (destroy clip correspondence, preserve the marginal
    distribution) and far cheaper, since it skips a second vision-trunk forward.

    Lets us ask, on ONE checkpoint, which auxiliary stream's *content* the 4-stream
    model actually consumes: run this with 'foreign' and compare the Event-F1 drop to
    the phase stream's (which is ~0).
    """
    mode = getattr(cfg.IHSV, 'EVAL_MISMATCH', 'none')
    if not _EVAL_MISMATCH_CACHE.get('ihsv_logged'):
        _EVAL_MISMATCH_CACHE['ihsv_logged'] = True
        logger.info(f"[ihsv-mismatch] mode={mode!r}  feas mean={feas.mean():.6f} std={feas.std():.6f}")
    if mode == 'none':
        return feas
    if mode == 'mute':
        return torch.zeros_like(feas)
    if mode == 'foreign':
        cached = _EVAL_MISMATCH_CACHE.get('ihsv')
        if cached is None:
            _EVAL_MISMATCH_CACHE['ihsv'] = feas.detach().clone()
            return torch.roll(feas, shifts=1, dims=0)   # first batch: no donor yet
        B = feas.shape[0]
        reps = (B + cached.shape[0] - 1) // cached.shape[0]
        return cached.repeat(reps, *([1] * (cached.dim() - 1)))[:B].to(feas.device)
    raise ValueError(f"unknown cfg.IHSV.EVAL_MISMATCH={mode!r}")


@torch.no_grad()
def validate_epoch(model, fusion_cnn, val_dataloader, criterion, criterion_event, epoch, eval_only=False, use_real_cnn=True,
                   phase_model=None, phase_projector=None, phase_temporal_attn=None,
                   pix2pix_generator=None, pix2pix_normalizer=None,
                cyclegan_generator=None, cyclegan_if_range=None, cyclegan_mel_range=None,
                   depth_temporal_attn=None, ihsv_temporal_attn=None,
                   complex_visual_temporal_attn=None, complex_audio_temporal_attn=None,
                   phase_variance_gate=None,
                   log_pi_total=None,
                   collect=False,
                   phase_aware_logit=None):
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

    # Track gate values and phase signals for AlignedTemporalDiffComplexAttention
    gate_audio_values = []
    gate_visual_values = []
    phase_audio_pre_values = []
    phase_audio_post_values = []
    phase_visual_pre_values = []
    phase_visual_post_values = []

    model.eval()
    if fusion_cnn is not None:
        fusion_cnn.eval()
    if phase_model is not None:
        phase_model.eval()
    if phase_projector is not None:
        phase_projector.eval()
    if phase_temporal_attn is not None:
        phase_temporal_attn.eval()
    if complex_visual_temporal_attn is not None:
        complex_visual_temporal_attn.eval()
    if complex_audio_temporal_attn is not None:
        complex_audio_temporal_attn.eval()
    if phase_variance_gate is not None:
        phase_variance_gate.eval()

    for n_iter, batch_data in enumerate(val_dataloader):
        data_time.update(time.time() - end_time)

        '''Feed input to model'''
        depth_inputs = None
        if cfg.COMPLEX_REPR.ENABLED:
            audio, visual, visual_imag, text, full_label, avc_label, category_label, vid_name, audio_phase_data = batch_data
            visual_imag_inputs = visual_imag.cuda()
            if cfg.COMPLEX_REPR.AUDIO_MODE == 'fft_real_imag':
                audio_imag_inputs = audio_phase_data.squeeze(1).cuda()
                phase_raw = None
            else:
                phase_raw = audio_phase_data.squeeze(1).cuda()
                audio_imag_inputs = None
        elif cfg.COMPLEX_TEMPORAL.ENABLED:
            audio, visual, visual_imag, text, full_label, avc_label, category_label, vid_name, phase_raw = batch_data
            visual_imag_inputs = visual_imag.cuda()
            phase_raw = phase_raw.squeeze(1).cuda()
            audio_imag_inputs = None
        elif cfg.DEPTH.ENABLED and cfg.PHASE.ENABLED:
            audio, visual, depth, text, full_label, avc_label, category_label, vid_name, phase_raw = batch_data
            depth_inputs = depth.cuda()  # [bs, 10, 1, 224, 224]
            visual_imag_inputs = None
            phase_raw = phase_raw.squeeze(1).cuda()
            audio_imag_inputs = None
        elif cfg.DEPTH.ENABLED:
            audio, visual, depth, text, full_label, avc_label, category_label, vid_name = batch_data
            depth_inputs = depth.cuda()  # [bs, 10, 1, 224, 224]
            visual_imag_inputs = None
            phase_raw = None
            audio_imag_inputs = None
        elif cfg.IHSV.ENABLED and cfg.PHASE.ENABLED:
            audio, visual, visual_imag, text, full_label, avc_label, category_label, vid_name, phase_raw = batch_data
            visual_imag_inputs = visual_imag.cuda()
            phase_raw = phase_raw.squeeze(1).cuda()
            audio_imag_inputs = None
        elif cfg.IHSV.ENABLED:
            audio, visual, visual_imag, text, full_label, avc_label, category_label, vid_name = batch_data
            visual_imag_inputs = visual_imag.cuda()
            phase_raw = None
            audio_imag_inputs = None
        elif cfg.PHASE.ENABLED:
            audio, visual, text, full_label, avc_label, category_label, vid_name, phase_raw = batch_data
            visual_imag_inputs = None
            phase_raw = phase_raw.squeeze(1).cuda()
            audio_imag_inputs = None
        else:
            audio, visual, text, full_label, avc_label, category_label, vid_name = batch_data
            visual_imag_inputs = None
            phase_raw = None
            audio_imag_inputs = None
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
        if depth_inputs is not None:
            inputs[ModalityType.DEPTH] = depth_inputs  # [bs, 10, 1, 224, 224]
        if visual_imag_inputs is not None:
            inputs[ModalityType.VISION_IMAG] = visual_imag_inputs
        with torch.no_grad():
            embeddings = model(inputs)
        audio_feas = embeddings['audio'] # [bs, 10, 1024]
        visual_feas = embeddings['vision'] # [bs, 10, 1024]
        text_feas = embeddings['text'] # [67+1, 1024]
        text_feas = text_feas.unsqueeze(0).repeat(bs, 1, 1) # [bs, 67+1, 1024]
        # pred_logits = embeddings['pred'] # [bs, 10, 67+1]

        # Track gate values and phase signals for explainability
        if 'explainability' in embeddings:
            explain = embeddings['explainability']
            gate_audio_values.append(explain['audio_gate'])
            gate_visual_values.append(explain['visual_gate'])
            phase_audio_pre_values.append(explain['audio_phase_pre'].mean().item())
            phase_audio_post_values.append(explain['audio_phase_post'].mean().item())
            phase_visual_pre_values.append(explain['visual_phase_pre'].mean().item())
            phase_visual_post_values.append(explain['visual_phase_post'].mean().item())

        #! =====================================================================
        #! AUDIO IMAGINARY / PHASE FEATURE COMPUTATION — same as train_epoch
        #! =====================================================================
        simm_pt = None
        if cfg.COMPLEX_REPR.ENABLED and audio_imag_inputs is not None:
            audio_imag_input = {ModalityType.AUDIO: audio_imag_inputs}
            with torch.no_grad():
                audio_imag_embeddings = model(audio_imag_input)
                audio_imag_feas = audio_imag_embeddings['audio']
                if phase_temporal_attn is not None:
                    for blk in phase_temporal_attn:
                        audio_imag_feas = blk(audio_imag_feas, audio_imag_feas, audio_imag_feas)
                simm_pt = compute_cross_modal_similarity(audio_imag_feas, text_feas)

        elif phase_raw is not None and phase_model is not None and isinstance(phase_model, PhaseAudioLoRA):
            #! LoRA phase pipeline (eval)
            B_ph, T_ph = phase_raw.shape[:2]
            phase_flat = phase_raw.reshape(B_ph * T_ph, *phase_raw.shape[2:])
            phase_if = compute_if(phase_flat)
            with torch.no_grad():
                phase_mel = phase_projector(phase_if)
                phase_mel_5d = phase_mel.reshape(B_ph, T_ph, 1, 128, 204)
                phase_feas = phase_model(phase_mel_5d)
                if phase_temporal_attn is not None:
                    for blk in phase_temporal_attn:
                        phase_feas = blk(phase_feas, phase_feas, phase_feas)
                simm_pt = compute_cross_modal_similarity(phase_feas, text_feas)

        elif phase_raw is not None and cfg.PHASE.MODE == 'hilbert':
            #! Hilbert pipeline (eval): hilbert imag [bs, 10, 1, 128, 204] → frozen encoder
            with torch.no_grad():
                hilbert_input = {ModalityType.AUDIO: phase_raw}
                hilbert_embeddings = model(hilbert_input)
                phase_feas = hilbert_embeddings['audio']
                if phase_temporal_attn is not None:
                    for blk in phase_temporal_attn:
                        phase_feas = blk(phase_feas, phase_feas, phase_feas)
                simm_pt = compute_cross_modal_similarity(phase_feas, text_feas)

        elif phase_raw is not None and cfg.PHASE.MODE == 'pix2pix':
            #! Pix2Pix pipeline (eval)
            B_ph, T_ph = phase_raw.shape[:2]
            phase_flat = phase_raw.reshape(B_ph * T_ph, *phase_raw.shape[2:])
            phase_if = compute_if(phase_flat)
            with torch.no_grad():
                phase_input_norm = phase_if / torch.pi
                phase_mel_norm = pix2pix_generator(phase_input_norm)
                phase_mel = pix2pix_normalizer.denormalize(phase_mel_norm)
                phase_mel = apply_imagebind_audio_norm(phase_mel)  # FIX: re-apply Normalize that load_and_transform_audio_data uses
                phase_input = {ModalityType.AUDIO: phase_mel.reshape(B_ph, T_ph, 1, 128, 204)}
                phase_embeddings = model(phase_input)
                phase_feas = phase_embeddings['audio']
                if phase_temporal_attn is not None:
                    for blk in phase_temporal_attn:
                        phase_feas = blk(phase_feas, phase_feas, phase_feas)
                simm_pt = compute_cross_modal_similarity(phase_feas, text_feas)

        elif phase_raw is not None and cfg.PHASE.MODE == 'cyclegan':
            #! Vanilla CycleGAN pipeline (eval): unconditional IF → mel translation
            phase_raw = _apply_eval_mismatch(phase_raw)
            B_ph, T_ph = phase_raw.shape[:2]
            phase_flat = phase_raw.reshape(B_ph * T_ph, *phase_raw.shape[2:])
            phase_if = compute_if(phase_flat)
            with torch.no_grad():
                phase_mel = translate_phase_to_mel(
                    cyclegan_generator, cyclegan_if_range, cyclegan_mel_range, phase_if
                )
                phase_mel = apply_imagebind_audio_norm(phase_mel)  # FIX: re-apply Normalize that load_and_transform_audio_data uses
                phase_input = {ModalityType.AUDIO: phase_mel.reshape(B_ph, T_ph, 1, 128, 204)}
                phase_embeddings = model(phase_input)
                phase_feas = phase_embeddings['audio']
                if phase_temporal_attn is not None:
                    for blk in phase_temporal_attn:
                        phase_feas = blk(phase_feas, phase_feas, phase_feas)
                simm_pt = compute_cross_modal_similarity(phase_feas, text_feas)
                if getattr(cfg.PHASE, 'EVAL_MISMATCH', 'none') == 'mute':
                    simm_pt = torch.zeros_like(simm_pt)
                if not _EVAL_MISMATCH_CACHE.get('logged_pt'):
                    _EVAL_MISMATCH_CACHE['logged_pt'] = True
                    logger.info(f"[eval-mismatch] phase_mel mean={phase_mel.mean():.6f} "
                                f"std={phase_mel.std():.6f} | simm_pt mean={simm_pt.mean():.6f} "
                                f"std={simm_pt.std():.6f}")

        elif phase_raw is not None and cfg.PHASE.MODE == 'cond_cyclegan':
            #! Conditional CycleGAN pipeline (eval): (IF, mel_cond) → mel translation
            B_ph, T_ph = phase_raw.shape[:2]
            phase_flat = phase_raw.reshape(B_ph * T_ph, *phase_raw.shape[2:])
            mel_cond = audio_inputs.reshape(B_ph * T_ph, 1, 128, 204)
            phase_if = compute_if(phase_flat)
            with torch.no_grad():
                phase_mel = translate_phase_to_mel_cond(
                    cyclegan_generator, cyclegan_if_range, cyclegan_mel_range, phase_if, mel_cond, device=phase_if.device
                )
                phase_mel = apply_imagebind_audio_norm(phase_mel)  # FIX: re-apply Normalize that load_and_transform_audio_data uses
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
            if cfg.PHASE.INPUT_TYPE == 'if_gd':
                phase_flat = compute_if_gd(phase_flat)
                phase_for_proj = phase_flat[:, 0:1, :, :]
            elif cfg.PHASE.INPUT_TYPE == 'if':
                phase_for_proj = compute_if(phase_flat)
            else:
                phase_for_proj = phase_flat
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
            imag_visual_feas = _apply_ihsv_mismatch(imag_visual_feas)
            # iHSV temporal self-attention (trainable)
            if ihsv_temporal_attn is not None:
                for blk in ihsv_temporal_attn:
                    imag_visual_feas = blk(imag_visual_feas, imag_visual_feas, imag_visual_feas)
            simm_imag_vt = compute_cross_modal_similarity(imag_visual_feas, text_feas)  # [bs, 10, C+1]

        #! DEPTH: depth x text similarity
        simm_dt = None
        if ModalityType.DEPTH in embeddings:
            depth_feas = embeddings[ModalityType.DEPTH]  # [bs, 10, 1024]
            if depth_temporal_attn is not None:
                for blk in depth_temporal_attn:
                    depth_feas = blk(depth_feas, depth_feas, depth_feas)
            simm_dt = compute_cross_modal_similarity(depth_feas, text_feas)  # [bs, 10, C+1]

        #! =====================================================================
        #! COMPLEX TEMPORAL ATTENTION MODE (eval)
        #! =====================================================================
        if cfg.COMPLEX_TEMPORAL.ENABLED and complex_visual_temporal_attn is not None and complex_audio_temporal_attn is not None:
            imag_visual_feas = embeddings.get(ModalityType.VISION_IMAG)
            if imag_visual_feas is None:
                imag_visual_feas = torch.zeros_like(visual_feas)

            phase_feas_for_complex = torch.zeros_like(audio_feas)
            if phase_raw is not None and cfg.PHASE.MODE == 'cyclegan':
                B_ph, T_ph = phase_raw.shape[:2]
                phase_flat = phase_raw.reshape(B_ph * T_ph, *phase_raw.shape[2:])
                phase_if = compute_if(phase_flat)
                phase_mel = translate_phase_to_mel(
                    cyclegan_generator, cyclegan_if_range, cyclegan_mel_range, phase_if)
                phase_mel = apply_imagebind_audio_norm(phase_mel)  # FIX: re-apply Normalize that load_and_transform_audio_data uses
                phase_input = {ModalityType.AUDIO: phase_mel.reshape(B_ph, T_ph, 1, 128, 204)}
                phase_embeddings = model(phase_input)
                phase_feas_for_complex = phase_embeddings['audio']
            elif phase_raw is not None and cfg.PHASE.MODE == 'cond_cyclegan':
                B_ph, T_ph = phase_raw.shape[:2]
                phase_flat = phase_raw.reshape(B_ph * T_ph, *phase_raw.shape[2:])
                mel_cond = audio_inputs.reshape(B_ph * T_ph, 1, 128, 204)
                phase_if = compute_if(phase_flat)
                phase_mel = translate_phase_to_mel_cond(
                    cyclegan_generator, cyclegan_if_range, cyclegan_mel_range, phase_if, mel_cond, device=phase_if.device)
                phase_mel = apply_imagebind_audio_norm(phase_mel)  # FIX: re-apply Normalize that load_and_transform_audio_data uses
                phase_input = {ModalityType.AUDIO: phase_mel.reshape(B_ph, T_ph, 1, 128, 204)}
                phase_embeddings = model(phase_input)
                phase_feas_for_complex = phase_embeddings['audio']

            vis_real, vis_imag = visual_feas, imag_visual_feas
            for blk in complex_visual_temporal_attn:
                vis_real, vis_imag = blk(vis_real, vis_imag)

            aud_real, aud_imag = audio_feas, phase_feas_for_complex
            for blk in complex_audio_temporal_attn:
                aud_real, aud_imag = blk(aud_real, aud_imag)

            simm_vt = compute_cross_modal_similarity(vis_real, text_feas)
            simm_imag_vt = compute_cross_modal_similarity(vis_imag, text_feas)
            simm_at = compute_cross_modal_similarity(aud_real, text_feas)
            simm_pt = compute_cross_modal_similarity(aud_imag, text_feas)
        else:
            #! GEN1: generate AVE prediction logits (without AV similarity modulation)
            simm_at = compute_cross_modal_similarity(audio_feas, text_feas) # [bs, 10, 67+1]
            simm_vt = compute_cross_modal_similarity(visual_feas, text_feas) # [bs, 10, 67+1]

        #! =====================================================================
        #! FUSION: RealCNN, ComplexCNN, or simple average (when disabled)
        #! =====================================================================
        # Normalize all available similarity matrices
        simm_vt_norm = min_max_normalize(simm_vt, dim=-1)
        simm_at_norm = min_max_normalize(simm_at, dim=-1)
        simm_imag_vt_norm = min_max_normalize(simm_imag_vt, dim=-1) if simm_imag_vt is not None else None
        simm_dt_norm = min_max_normalize(simm_dt, dim=-1) if simm_dt is not None else None
        simm_pt_norm = min_max_normalize(simm_pt, dim=-1) if simm_pt is not None else None
        # Use depth as visual-imaginary channel when available, else fall back to iHSV
        simm_vis_imag_norm = simm_dt_norm if simm_dt_norm is not None else simm_imag_vt_norm

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
            #! ComplexCNN: 2 complex channels in complex_pair mode (visual / audio domains)
            _use_pal_v = (phase_aware_logit is not None)
            if _use_pal_v:
                pred_logits, fusion_phase = fusion_cnn(
                    simm_vt_norm, simm_at_norm,
                    simm_vis_imag_norm, simm_pt_norm,
                    return_phase=True)
                pred_logits = phase_aware_logit(pred_logits, fusion_phase)
            else:
                pred_logits = fusion_cnn(simm_vt_norm, simm_at_norm,
                                         simm_vis_imag_norm, simm_pt_norm)  # [bs, 10, C+1]
            pred_logits = min_max_normalize(pred_logits, dim=-1)  # output min-max norm

        #! Apply temporal smoothing on predictions (Gaussian smoothing along time axis)
        if cfg.SMOOTH.ENABLED:
            pred_logits = temporal_smooth(pred_logits,
                                          kernel_size=cfg.SMOOTH.KERNEL_SIZE,
                                          sigma=cfg.SMOOTH.SIGMA)

        #! Case 1: input(raw) -> complex -> mag -> min-max -> CE
        # pred_logits = complex_cnn(simm_vt, simm_at)  # [bs, 10, C+1]
        # pred_logits = min_max_normalize(pred_logits, dim=-1)  # output min-max norm

        #! Case 2: input(raw) -> complex -> mag -> sigmoid -> CE
        # pred_logits = complex_cnn(simm_vt, simm_at)  # [bs, 10, C+1]
        # pred_logits = torch.sigmoid(pred_logits)

        #! Case 3: input(raw) -> complex -> mag -> CE (no normalization)
        # pred_logits = complex_cnn(simm_vt, simm_at)  # [bs, 10, C+1]

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

            # Post-hoc logit adjustment (Menon et al. 2021): subtract tau * log_pi
            # only on the metric path. CE loss above is computed on raw logits.
            if cfg.LOGIT_ADJ.ENABLED and log_pi_total is not None and float(cfg.LOGIT_ADJ.TAU) != 0.0:
                pred_logits_eval = pred_logits - float(cfg.LOGIT_ADJ.TAU) * log_pi_total.to(pred_logits.device)
            else:
                pred_logits_eval = pred_logits

            # Idea 4: Phase-variance fg/bg gate. Suppresses fg classes when
            # cross-modal phase across classes is incoherent (likely background).
            if phase_variance_gate is not None:
                pred_logits_eval, _fg_prob = phase_variance_gate(
                    pred_logits_eval, simm_vt, simm_at, bg_flag=cfg.TOTAL_BG_CLASS_ID
                )

            acc = compute_accuracy_supervised_v1(pred_logits_eval, full_labels, bg_flag=cfg.TOTAL_BG_CLASS_ID)
            accuracy.update(acc.item(), bs * 10)
            diag.update(pred_logits_eval.argmax(-1), full_labels)
            seg_f, eve_f = compute_seg_eve_fscores_v1(pred_logits_eval, avc_labels, category_labels, bg_flag=cfg.TOTAL_BG_CLASS_ID)
            seg_fscore.update(seg_f, n=1)
            eve_fscore.update(eve_f, n=1)

            if collect:
                argmax = pred_logits_eval.argmax(dim=-1)  # [B, T] — adjusted argmax
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

    # Print gate values and phase signals at end of validation
    if len(gate_audio_values) > 0:
        avg_gate_audio = np.mean(gate_audio_values)
        avg_gate_visual = np.mean(gate_visual_values)
        logger.info(f'\n{"="*60}')
        logger.info(f'Validation Epoch {epoch} - Gate Values & Phase Signals')
        logger.info(f'{"="*60}')
        logger.info(f'  Audio Gate:  {avg_gate_audio:.4f}  (>{0.5:.1f} favors original, <{0.5:.1f} favors attention)')
        logger.info(f'  Visual Gate: {avg_gate_visual:.4f}  (>{0.5:.1f} favors original, <{0.5:.1f} favors attention)')

        if len(phase_audio_pre_values) > 0:
            avg_phase_audio_pre = np.mean(phase_audio_pre_values)
            avg_phase_audio_post = np.mean(phase_audio_post_values)
            avg_phase_visual_pre = np.mean(phase_visual_pre_values)
            avg_phase_visual_post = np.mean(phase_visual_post_values)
            phase_change_audio = avg_phase_audio_post - avg_phase_audio_pre
            phase_change_visual = avg_phase_visual_post - avg_phase_visual_pre

            logger.info(f'  ----------------------------------------')
            logger.info(f'  Phase (arctan(imag/real) = "event-ness"):')
            logger.info(f'    Audio  - Pre: {avg_phase_audio_pre:.4f}, Post: {avg_phase_audio_post:.4f}, Change: {phase_change_audio:+.4f}')
            logger.info(f'    Visual - Pre: {avg_phase_visual_pre:.4f}, Post: {avg_phase_visual_post:.4f}, Change: {phase_change_visual:+.4f}')
        logger.info(f'{"="*60}\n')

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
