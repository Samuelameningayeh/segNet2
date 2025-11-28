# eval_segnet_camvid.py
import torch
import torch.nn as nn

from model_segnet import SegNet
from data import get_camvid_test_loader
from metrics import compute_confusion_matrix, metrics_from_confusion_matrix


def evaluate_model(checkpoint_path, test_txt,
                   img_size=(360, 480), num_classes=12,
                   batch_size=4, num_workers=4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # data
    test_loader = get_camvid_test_loader(
        test_txt, img_size, batch_size, num_workers
    )

    # model
    model = SegNet(num_classes=num_classes, in_channels=3).to(device)

    # load weights
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    criterion = nn.CrossEntropyLoss()

    running_loss = 0.0
    cm_total = torch.zeros(num_classes, num_classes,
                           dtype=torch.int64, device=device)

    with torch.no_grad():
        for imgs, masks in test_loader:
            imgs = imgs.to(device)
            masks = masks.to(device)

            logits = model(imgs)
            loss = criterion(logits, masks)
            running_loss += loss.item() * imgs.size(0)

            preds = torch.argmax(logits, dim=1)
            cm_total += compute_confusion_matrix(
                preds, masks, num_classes, device=device
            )

    test_loss = running_loss / len(test_loader.dataset)
    metrics = metrics_from_confusion_matrix(cm_total)

    print("Test Loss:", test_loss)
    print("Test Global Acc:", metrics["global_acc"])
    print("Test Class Avg Acc:", metrics["class_acc"])
    print("Test mIoU:", metrics["miou"])


if __name__ == "__main__":
    checkpoint = "segnet_camvid_best.pth"
    test_txt = "SegNet/CamVid/test.txt"
    evaluate_model(checkpoint, test_txt)
