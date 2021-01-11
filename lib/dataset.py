from albumentations import Compose, GaussNoise, Blur, Normalize
import cv2
import numpy as np
import pathlib as pl
import torch
from torch.utils.data import Dataset


class SUN360(Dataset):

    def __init__(self, folder, augmentation=False):

        if augmentation:
            self.aug = Compose([GaussNoise(p=0.4),
                                Blur(p=0.3),
                                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), ])
        else:
            self.aug = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.folder = pl.Path(folder)
        self.filenames = []
        self.gts = []
        anns_fname = self.folder / "gt.csv"
        with open(anns_fname, "r") as f:
            lines = f.readlines()

        for l in lines:
            l = l.strip().split(",")
            self.filenames.append(l[0])
            angles = [
                float(l[1]),
                float(l[2])
            ]
            self.gts.append(angles)

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, item):
        filename = self.folder / self.filenames[item]

        img_np = cv2.imread(str(filename))

        assert img_np.shape[0] == 512
        assert img_np.shape[1] == 1024

        img_np = cv2.resize(img_np, (442, 221))
        img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)

        if self.aug is not None:
            data = {"image": img_np}
            img_np = self.aug(**data)["image"]

        img_pt = torch.from_numpy(img_np).permute(2, 0, 1)
        rx, ry = self.gts[item]

        label = torch.from_numpy(np.asarray([rx/180, ry/90])).float()

        return img_pt, label
