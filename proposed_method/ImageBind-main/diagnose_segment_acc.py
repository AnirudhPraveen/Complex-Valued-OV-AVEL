"""
Diagnose why segment-accuracy is flat while event-F1 improves.

Usage: from validate_epoch, accumulate per-batch tensors then call report().

    from diagnose_segment_acc import Diagnoser
    diag = Diagnoser(num_classes=cfg.TOTAL_BG_CLASS_ID + 1, bg_flag=cfg.TOTAL_BG_CLASS_ID)
    # ... inside the val loop, after `pred = pred_logits.argmax(-1)` (shape [B,T]):
    diag.update(pred, full_labels)         # both [B, T] long tensors on any device
    # ... after the loop:
    diag.report(logger=logger)             # or print
"""

import numpy as np
import torch


class Diagnoser:
    def __init__(self, num_classes, bg_flag):
        self.K = num_classes        # C + 1
        self.bg = bg_flag
        self.cm = np.zeros((self.K, self.K), dtype=np.int64)  # rows=GT, cols=pred
        self.boundary_correct = 0
        self.boundary_total   = 0
        self.interior_correct = 0
        self.interior_total   = 0
        self.bg_correct = 0
        self.bg_total   = 0
        self.fg_correct = 0
        self.fg_total   = 0

    @torch.no_grad()
    def update(self, pred, target):
        # pred, target: [B, T] long
        p = pred.detach().cpu().numpy().astype(np.int64)
        t = target.detach().cpu().numpy().astype(np.int64)
        B, T = t.shape

        # confusion matrix (vectorized)
        idx = t * self.K + p
        binc = np.bincount(idx.ravel(), minlength=self.K * self.K)
        self.cm += binc.reshape(self.K, self.K)

        # bg vs fg accuracy
        bg_mask = (t == self.bg)
        fg_mask = ~bg_mask
        self.bg_total   += int(bg_mask.sum())
        self.fg_total   += int(fg_mask.sum())
        self.bg_correct += int(((p == t) & bg_mask).sum())
        self.fg_correct += int(((p == t) & fg_mask).sum())

        # boundary vs interior on FG runs
        # boundary = first or last segment of a contiguous same-class fg run in GT
        for b in range(B):
            row = t[b]
            in_run = False
            run_class = -1
            run_start = 0
            for i in range(T):
                c = row[i]
                fg = (c != self.bg)
                if fg and (not in_run or c != run_class):
                    if in_run:
                        self._tally_run(p[b], t[b], run_start, i - 1)
                    in_run = True
                    run_class = c
                    run_start = i
                elif (not fg) and in_run:
                    self._tally_run(p[b], t[b], run_start, i - 1)
                    in_run = False
            if in_run:
                self._tally_run(p[b], t[b], run_start, T - 1)

    def _tally_run(self, p_row, t_row, s, e):
        # boundary segments are s and e (may coincide if length 1)
        for i in range(s, e + 1):
            is_boundary = (i == s or i == e)
            correct = int(p_row[i] == t_row[i])
            if is_boundary:
                self.boundary_total += 1
                self.boundary_correct += correct
            else:
                self.interior_total += 1
                self.interior_correct += correct

    def report(self, logger=None, top_confusions=10):
        log = logger.info if logger is not None else print

        total = self.cm.sum()
        diag = np.trace(self.cm)
        overall_acc = diag / max(total, 1)
        log(f"[diag] segments={total}  overall_acc={overall_acc:.4f}")

        log(f"[diag] bg_acc = {self.bg_correct}/{self.bg_total} "
            f"= {self.bg_correct / max(self.bg_total,1):.4f}   "
            f"fg_acc = {self.fg_correct}/{self.fg_total} "
            f"= {self.fg_correct / max(self.fg_total,1):.4f}")

        log(f"[diag] boundary_acc = {self.boundary_correct}/{self.boundary_total} "
            f"= {self.boundary_correct / max(self.boundary_total,1):.4f}   "
            f"interior_acc = {self.interior_correct}/{self.interior_total} "
            f"= {self.interior_correct / max(self.interior_total,1):.4f}")

        # bg<->fg leakage
        bg_pred_when_fg_gt = self.cm[:self.bg, self.bg].sum()         # GT fg, pred bg
        fg_pred_when_bg_gt = self.cm[self.bg, :self.bg].sum()         # GT bg, pred fg
        fg_to_fg_wrong    = self.cm[:self.bg, :self.bg].sum() - np.trace(self.cm[:self.bg, :self.bg])
        log(f"[diag] errors: GTfg->predbg = {bg_pred_when_fg_gt}   "
            f"GTbg->predfg = {fg_pred_when_bg_gt}   "
            f"fg->wrong_fg = {fg_to_fg_wrong}")

        # per-class accuracy (sorted, worst first)
        per_class_acc = np.divide(
            np.diag(self.cm).astype(np.float64),
            self.cm.sum(axis=1).clip(min=1).astype(np.float64),
        )
        support = self.cm.sum(axis=1)
        order = np.argsort(per_class_acc)
        log("[diag] worst per-class acc (class, acc, support):")
        shown = 0
        for c in order:
            if support[c] == 0:
                continue
            tag = " (bg)" if c == self.bg else ""
            log(f"        class {int(c):3d}{tag}: acc={per_class_acc[c]:.3f}  n={int(support[c])}")
            shown += 1
            if shown >= top_confusions:
                break

        # top off-diagonal confusions
        cm_off = self.cm.copy()
        np.fill_diagonal(cm_off, 0)
        flat = cm_off.ravel()
        top = np.argsort(flat)[::-1][:top_confusions]
        log(f"[diag] top {top_confusions} confusions (gt -> pred : count):")
        for f in top:
            gt, pr = divmod(int(f), self.K)
            if cm_off[gt, pr] == 0:
                break
            tag_gt = " (bg)" if gt == self.bg else ""
            tag_pr = " (bg)" if pr == self.bg else ""
            log(f"        {gt}{tag_gt} -> {pr}{tag_pr} : {int(cm_off[gt, pr])}")
