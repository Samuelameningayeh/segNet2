# data_camvid.py
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as F_vision


class CamVidDataset(Dataset):
    """
    Each line in txt file has:
        /SegNet/CamVid/train/img.png   /SegNet/CamVid/trainannot/img.png
    
    We must remove the leading '/SegNet/' so it works on the user's computer.
    """
    def __init__(self, list_path, img_size=(360, 480)):
        self.img_size = img_size

        self.samples = []
        with open(list_path, "r") as f:
            for line in f:
                if not line.strip():
                    continue
                img_path, mask_path = line.strip().split()

                # === FIX 1: remove "/SegNet/" for local paths ===
                if img_path.startswith('/SegNet/'):
                    img_path = img_path.replace('/SegNet/', '')
                if mask_path.startswith('/SegNet/'):
                    mask_path = mask_path.replace('/SegNet/', '')

                self.samples.append((img_path, mask_path))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mask_path = self.samples[idx]

        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        # Resize consistently (SegNet standard)
        img = F_vision.resize(img, self.img_size,
                              interpolation=F_vision.InterpolationMode.BILINEAR)
        mask = F_vision.resize(mask, self.img_size,
                               interpolation=F_vision.InterpolationMode.NEAREST)

        # Image → tensor + normalize
        img = F_vision.to_tensor(img)
        img = F_vision.normalize(
            img,
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

        mask_np = np.array(mask, dtype=np.int64)
        mask_tensor = torch.from_numpy(mask_np).long()

        return img, mask_tensor


def get_camvid_loaders(train_txt, val_txt, img_size=(360, 480),
                       batch_size=4, num_workers=4):
    train_ds = CamVidDataset(train_txt, img_size)
    val_ds = CamVidDataset(val_txt, img_size)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader


def get_camvid_test_loader(test_txt, img_size=(360, 480),
                           batch_size=4, num_workers=4):
    test_ds = CamVidDataset(test_txt, img_size)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)
    return test_loader
