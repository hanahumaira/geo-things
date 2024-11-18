import os
import numpy as np
import cv2
from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data import Dataset

class LabelMe(Dataset):
    def __init__(self, data_folder, transform=None, img_size = (448, 448)):
        self.img_folder = os.path.join(data_folder, 'images')
        self.mask_folder = os.path.join(data_folder, 'labels')
        self.img_size = img_size
        with open(os.path.join(data_folder, 'class_names.txt'), 'r') as f:
            labels = f.readlines()
        self.label = {}
        for i, label in enumerate(labels):
            self.label[i] = label.strip()
        self.transforms = transform
        if transform is None:
            self.transforms = transforms.Compose([
                                        transforms.RandomHorizontalFlip(p = 0.5),
                                        transforms.Resize(img_size),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                                    ])
        self.img_name = os.listdir(self.img_folder)

    
    def __getitem__(self, idx):
        img = Image.open(os.path.join(self.img_folder, self.img_name[idx]))
        img = self.transforms(img)
        mask_name = '.'.join(self.img_name[idx].split('.', maxsplit = 1)[:-1]) + '.npy'
        mask = np.load(os.path.join(self.mask_folder, mask_name)).astype(np.uint8)
        mask = cv2.resize(mask, self.img_size, cv2.INTER_AREA)

        return img, torch.from_numpy(mask.astype(np.int64))

    def __len__(self):
        return len(self.img_name)