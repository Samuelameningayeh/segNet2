# metrics_camvid.py
import torch


def compute_confusion_matrix(pred, target, num_classes,
                             ignore_index=None, device=None):
    """
    pred, target: (N, H, W) int64 tensors
    """
    if device is None:
        device = pred.device

    pred = pred.view(-1)
    target = target.view(-1)

    if ignore_index is not None:
        mask = target != ignore_index
        pred = pred[mask]
        target = target[mask]

    cm = torch.zeros(num_classes, num_classes, dtype=torch.int64, device=device)
    k = (target >= 0) & (target < num_classes)
    indices = num_classes * target[k] + pred[k]
    cm += torch.bincount(indices, minlength=num_classes ** 2).view(
        num_classes, num_classes
    )
    return cm


def metrics_from_confusion_matrix(cm):
    tp = cm.diag().float()
    total_gt = cm.sum(dim=1).float()
    total_pred = cm.sum(dim=0).float()
    union = total_gt + total_pred - tp

    global_acc = tp.sum() / cm.sum().clamp(min=1)
    class_acc = (tp / total_gt.clamp(min=1)).mean()
    miou = (tp / union.clamp(min=1)).mean()

    return {
        "global_acc": global_acc.item(),
        "class_acc": class_acc.item(),
        "miou": miou.item(),
    }


def compute_class_weights(loader, num_classes,
                          ignore_index=None, device="cpu"):
    """
    Median frequency balancing over the training loader.
    """
    counts = torch.zeros(num_classes, dtype=torch.float64)

    with torch.no_grad():
        for _, masks in loader:
            masks = masks.view(-1)
            if ignore_index is not None:
                masks = masks[masks != ignore_index]
            for c in range(num_classes):
                counts[c] += (masks == c).sum().item()

    counts[counts == 0] = 1.0
    freqs = counts / counts.sum()
    median_freq = freqs.median()
    weights = median_freq / freqs
    return weights.float().to(device)
