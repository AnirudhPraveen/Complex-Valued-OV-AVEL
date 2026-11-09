import numpy as np
import pdb


def Precision(X_pre, X_gt):

    N = len(X_pre)
    p = 0.0
    for i in range(N):
        x = X_pre[i, :]
        y = X_gt[i, :]
        p += np.sum(x*y)/np.sum(x)
    return p/N


def Recall(X_pre, X_gt):
    N = len(X_pre)
    p = 0.0
    for i in range(N):
        x = X_pre[i, :]
        y = X_gt[i, :]
        p += np.sum(x * y) / np.sum(y)
    return p/N


def F1(X_pre, X_gt):
    N = len(X_pre)
    p = 0
    for i in range(N):
        x = X_pre[i, :]
        y = X_gt[i, :]
        p += 2*np.sum(x * y) / (np.sum(x) + np.sum(y))
    return p/N

def event_level(SO_av, GT_av, length=None):
    # extract events
    N = SO_av.shape[0]
    if length is None:
        length = SO_av.shape[1]
    event_p_av = [None for n in range(N)]
    event_gt_av = [None for n in range(N)]

    TP_av = np.zeros(N)
    FP_av = np.zeros(N)
    FN_av = np.zeros(N)

    for n in range(N):
        seq_pred = SO_av[n, :]
        if np.sum(seq_pred) != 0:
            x = extract_event(seq_pred, n, length)
            event_p_av[n] = x

        seq_gt = GT_av[n, :]
        if np.sum(seq_gt) != 0:
            x = extract_event(seq_gt, n, length)
            event_gt_av[n] = x

        tp, fp, fn = event_wise_metric(event_p_av[n], event_gt_av[n])
        TP_av[n] += tp
        FP_av[n] += fp
        FN_av[n] += fn

    # n = len(FP_av)
    F_av = []
    for ii in range(N):
        if (TP_av + FP_av)[ii] != 0 or (TP_av + FN_av)[ii] != 0:
            F_av.append(2 * TP_av[ii] / (2 * TP_av[ii] + (FN_av + FP_av)[ii]))

    if len(F_av) == 0:
        f_av = 1.0 # all true negatives
    else:
        f_av = (sum(F_av)/len(F_av))
    # pdb.set_trace()

    return f_av


def segment_level(SO_av, GT_av):
    # SO_av, GT_av: [46/67+1, 10]
    # compute F scores
    TP_av = np.sum(SO_av * GT_av, axis=1)
    FN_av = np.sum((1 - SO_av) * GT_av, axis=1)
    FP_av = np.sum(SO_av * (1 - GT_av), axis=1)
    n = len(FP_av)
    # pdb.set_trace()
    F_av = []
    for ii in range(n):
        if (TP_av + FP_av)[ii] != 0 or (TP_av + FN_av)[ii] != 0:
            F_av.append(2 * TP_av[ii] / (2 * TP_av[ii] + (FN_av + FP_av)[ii]))
    # pdb.set_trace()

    if len(F_av) == 0:
        f_av = 1.0 # all true negatives
    else:
        f_av = (sum(F_av)/len(F_av))
    # pdb.set_trace()

    return f_av


def to_vec(start, end, length=10):
    x = np.zeros(length)
    for i in range(start, end):
        x[i] = 1
    return x

def extract_event(seq, n, length=None):
    if length is None:
        length = len(seq)
    x = []
    i = 0
    while i < length:
        if seq[i] == 1:
            start = i
            if i + 1 == length:
                i = i + 1
                end = i
                x.append(to_vec(start, end, length))
                break

            for j in range(i + 1, length):
                if seq[j] != 1:
                    i = j + 1
                    end = j
                    x.append(to_vec(start, end, length))
                    break
                else:
                    i = j + 1
                    if i == length:
                        end = i
                        x.append(to_vec(start, end, length))
                        break
        else:
            i += 1
    return x

def event_wise_metric(event_p, event_gt, tiou=0.5):
    TP = 0
    FP = 0
    FN = 0

    if event_p is not None:
        num_event = len(event_p)
        for i in range(num_event):
            x1 = event_p[i]
            if event_gt is not None:
                nn = len(event_gt)
                flag = True
                for j in range(nn):
                    x2 = event_gt[j]
                    if np.sum(x1 * x2) >= tiou * np.sum(x1 + x2 - x1 * x2):
                        TP += 1
                        flag = False
                        break
                if flag:
                    FP += 1
            else:
                FP += 1

    if event_gt is not None:
        num_event = len(event_gt)
        for i in range(num_event):
            x1 = event_gt[i]
            if event_p is not None:
                nn = len(event_p)
                flag = True
                for j in range(nn):
                    x2 = event_p[j]
                    if np.sum(x1 * x2) >= tiou * np.sum(x1 + x2 - x1 * x2):
                        flag = False
                        break
                if flag:
                    FN += 1
            else:
                FN += 1
    return TP, FP, FN


def event_level_all(SO_av, GT_av, length, tiou_list, num_classes=100):
    """
    Event-level F1 at multiple tIoU thresholds (matches PreFM's event_level_all_UnAV).

    Args:
        SO_av: [N, T] predicted binary matrix (N=num_classes)
        GT_av: [N, T] ground truth binary matrix
        length: temporal length T
        tiou_list: list/array of tIoU thresholds (e.g., np.arange(0.1, 1.0, 0.1))
        num_classes: number of event classes (default 100)

    Returns:
        list of F1 scores, one per tIoU threshold
    """
    N = num_classes

    # Pre-extract events for each class (avoid recomputing per threshold)
    event_p_av = [None for _ in range(N)]
    event_gt_av = [None for _ in range(N)]

    for n in range(N):
        seq_pred = SO_av[n, :]
        if np.sum(seq_pred) != 0:
            event_p_av[n] = extract_event(seq_pred, n, length)
        seq_gt = GT_av[n, :]
        if np.sum(seq_gt) != 0:
            event_gt_av[n] = extract_event(seq_gt, n, length)

    f_av_list = []
    for threshold in tiou_list:
        TP_av = np.zeros(N)
        FP_av = np.zeros(N)
        FN_av = np.zeros(N)

        for n in range(N):
            tp, fp, fn = event_wise_metric(event_p_av[n], event_gt_av[n], threshold)
            TP_av[n] += tp
            FP_av[n] += fp
            FN_av[n] += fn

        F_av = []
        for ii in range(N):
            if (TP_av[ii] + FP_av[ii]) != 0 or (TP_av[ii] + FN_av[ii]) != 0:
                F_av.append(2 * TP_av[ii] / (2 * TP_av[ii] + (FN_av[ii] + FP_av[ii])))

        if len(F_av) == 0:
            f_av = 1.0
        else:
            f_av = sum(F_av) / len(F_av)
        f_av_list.append(f_av)

    return f_av_list