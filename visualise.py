# visualize_segnet_camvid.py
import os
import numpy as np
from PIL import Image

import torch

from model_segnet import SegNet
from data_camvid import get_camvid_test_loader


def get_camvid_palette(num_classes):
    # Simple 12-color palette – adjust as you like
    palette = [
        (128, 128, 128),  # 0
        (128, 0, 0),      # 1
        (192, 192, 128),  # 2
        (128, 64, 128),   # 3
        (0, 0, 192),      # 4
        (128, 128, 0),    # 5
        (192, 128, 128),  # 6
        (64, 64, 128),    # 7
        (64, 0, 128),     # 8
        (64, 64, 0),      # 9
        (0, 128, 192),    # 10
        (0, 0, 0),        # 11
    ]
    return palette[:num_classes]


def colorize_mask(mask, palette):
    if isinstance(mask, torch.Tensor):
        mask = mask.cpu().numpy()
    h, w = mask.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    for cls_id, color in enumerate(palette):
        out[mask == cls_id] = color
    return Image.fromarray(out)


def denormalize_image(img_tensor):
    mean = torch.tensor([0.485, 0.456, 0.406],
                        device=img_tensor.device).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225],
                       device=img_tensor.device).view(3, 1, 1)
    img = img_tensor * std + mean
    img = torch.clamp(img, 0, 1)
    img = img.cpu().permute(1, 2, 0).numpy()
    img = (img * 255).astype(np.uint8)
    return Image.fromarray(img)


def visualize_predictions(checkpoint_path, list_txt,
                          out_dir="vis", num_classes=12,
                          img_size=(360, 480),
                          max_batches=3, batch_size=4, num_workers=4):
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    loader = get_camvid_test_loader(list_txt, img_size,
                                    batch_size=batch_size,
                                    num_workers=num_workers)

    model = SegNet(num_classes=num_classes, in_channels=3).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    palette = get_camvid_palette(num_classes)

    with torch.no_grad():
        batch_idx = 0
        for imgs, masks in loader:
            imgs = imgs.to(device)
            masks = masks.to(device)

            logits = model(imgs)
            preds = torch.argmax(logits, dim=1)

            bs = imgs.size(0)
            for i in range(bs):
                img_vis = denormalize_image(imgs[i])
                gt_vis = colorize_mask(masks[i], palette)
                pred_vis = colorize_mask(preds[i], palette)

                w, h = img_vis.size
                canvas = Image.new("RGB", (w * 3, h))
                canvas.paste(img_vis, (0, 0))
                canvas.paste(gt_vis, (w, 0))
                canvas.paste(pred_vis, (2 * w, 0))

                canvas.save(os.path.join(
                    out_dir, f"batch{batch_idx:03d}_img{i:02d}.png"
                ))

            batch_idx += 1
            if batch_idx >= max_batches:
                break


if __name__ == "__main__":
    checkpoint = "segnet_camvid_best.pth"
    list_txt = "SegNet/CamVid/val.txt"  # or test.txt
    visualize_predictions(checkpoint, list_txt, out_dir="vis_val")
