import torch.utils.data as data
import numpy as np
import os
from PIL import Image


def pascal_part_cmap(N=256, normalized=False):
    def bitget(byteval, idx):
        return (byteval & (1 << idx)) != 0

    dtype = "float32" if normalized else "uint8"
    cmap = np.zeros((N, 3), dtype=dtype)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7 - j)
            g = g | (bitget(c, 1) << 7 - j)
            b = b | (bitget(c, 2) << 7 - j)
            c = c >> 3

        cmap[i] = np.array([r, g, b])

    cmap = cmap / 255 if normalized else cmap
    return cmap


class PascalPart(data.Dataset):
    """Pascal Part dataset
    Args:
        root (string): Root directory of the Pascal Part Dataset.
        image_set (string, optional): Select the image_set to use, ``train``, ``trainval`` or ``val``
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
    """

    cmap = pascal_part_cmap()

    def __init__(self, root, image_set="train", transform=None):
        self.root = os.path.expanduser(root)
        self.img_dir = os.path.join(root, "img_dir", image_set)
        self.ann_dir = os.path.join(root, "ann_dir", image_set)

        self.imgs = [
            img_name
            for img_name in os.listdir(self.img_dir)
            if img_name.endswith(".jpg")
        ]
        self.anns = [
            ann_name
            for ann_name in os.listdir(self.ann_dir)
            if ann_name.endswith(".png")
        ]

        assert len(self.imgs) == len(
            self.anns
        ), "Incomplete dataset. Number of images and annotations does not match"

        self.anns = [img_name[:-4] + ".png" for img_name in self.imgs]

        self.transform = transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is the image segmentation.
        """
        img = Image.open(os.path.join(self.img_dir, self.imgs[index])).convert("RGB")
        target = Image.open(os.path.join(self.ann_dir, self.anns[index]))
        if self.transform is not None:
            img, target = self.transform(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)

    @classmethod
    def decode_target(cls, mask):
        """Decode semantic mask to RGB image"""
        return cls.cmap[mask]
