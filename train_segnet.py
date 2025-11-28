# train_segnet_camvid.py
import time
import torch
import torch.nn as nn
import torch.optim as optim

from model_segnet import SegNet
from data import get_camvid_loaders
from metrics import (
    compute_confusion_matrix,
    metrics_from_confusion_matrix,
    compute_class_weights,
)


def train_one_epoch(model, loader, optimizer, criterion,
                    device, num_classes, ignore_index=None):
    model.train()
    running_loss = 0.0
    cm_total = torch.zeros(num_classes, num_classes,
                           dtype=torch.int64, device=device)

    for imgs, masks in loader:
        imgs = imgs.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        optimizer.zero_grad()
        logits = model(imgs)
        loss = criterion(logits, masks)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * imgs.size(0)

        preds = torch.argmax(logits, dim=1)
        cm_total += compute_confusion_matrix(
            preds, masks, num_classes, ignore_index, device=device
        )

    epoch_loss = running_loss / len(loader.dataset)
    metrics = metrics_from_confusion_matrix(cm_total)
    return epoch_loss, metrics


def evaluate(model, loader, criterion,
             device, num_classes, ignore_index=None):
    model.eval()
    running_loss = 0.0
    cm_total = torch.zeros(num_classes, num_classes,
                           dtype=torch.int64, device=device)

    with torch.no_grad():
        for imgs, masks in loader:
            imgs = imgs.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            logits = model(imgs)
            loss = criterion(logits, masks)
            running_loss += loss.item() * imgs.size(0)

            preds = torch.argmax(logits, dim=1)
            cm_total += compute_confusion_matrix(
                preds, masks, num_classes, ignore_index, device=device
            )

    epoch_loss = running_loss / len(loader.dataset)
    metrics = metrics_from_confusion_matrix(cm_total)
    return epoch_loss, metrics


def main():
    # ----- config -----
    train_txt = "CamVid/train.txt"
    val_txt   = "CamVid/val.txt"
    test_txt  = "CamVid/test.txt"


    img_size = (360, 480)
    num_classes = 12          # your masks have values 0..11
    batch_size = 4
    num_workers = 4
    num_epochs = 50

    lr = 1e-3
    momentum = 0.9
    weight_decay = 5e-4
    ignore_index = None
    use_median_freq = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # ----- data -----
    train_loader, val_loader = get_camvid_loaders(
        train_txt, val_txt, img_size, batch_size, num_workers
    )

    # ----- model -----
    model = SegNet(num_classes=num_classes, in_channels=3).to(device)

    # ----- Loss -----
    if use_median_freq:
        print("Computing class weights...")
        class_weights = compute_class_weights(
            train_loader, num_classes, ignore_index, device=device
        )
        print("Class weights:", class_weights.cpu().numpy())
    else:
        class_weights = None

    # Only pass ignore_index if it's not None
    if ignore_index is not None:
        criterion = nn.CrossEntropyLoss(
            weight=class_weights,
            ignore_index=ignore_index
        )
    else:
        criterion = nn.CrossEntropyLoss(
            weight=class_weights
        )

    # ----- optimizer & scheduler -----
    optimizer = optim.SGD(
        model.parameters(), lr=lr,
        momentum=momentum, weight_decay=weight_decay
    )
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, step_size=20, gamma=0.1
    )

    best_miou = 0.0
    save_path = "segnet_camvid_best.pth"

    for epoch in range(1, num_epochs + 1):
        start = time.time()

        train_loss, train_metrics = train_one_epoch(
            model, train_loader, optimizer, criterion,
            device, num_classes, ignore_index
        )
        val_loss, val_metrics = evaluate(
            model, val_loader, criterion,
            device, num_classes, ignore_index
        )

        scheduler.step()
        elapsed = time.time() - start

        print(
            f"[Epoch {epoch:03d}/{num_epochs:03d}] "
            f"Time {elapsed:.1f}s | "
            f"TrainLoss {train_loss:.4f} ValLoss {val_loss:.4f} | "
            f"TrainG {train_metrics['global_acc']:.3f} "
            f"ValG {val_metrics['global_acc']:.3f} "
            f"ValC {val_metrics['class_acc']:.3f} "
            f"ValmIoU {val_metrics['miou']:.3f}"
        )

        if val_metrics["miou"] > best_miou:
            best_miou = val_metrics["miou"]
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_metrics": val_metrics,
            }, save_path)
            print(f"  --> New best mIoU={best_miou:.3f}, saved to {save_path}")


if __name__ == "__main__":
    main()
