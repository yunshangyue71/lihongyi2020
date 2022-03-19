import torchvision.transforms as transforms
import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
import os

class ImgDataset(Dataset):
    def __init__(self, img_dir, wh=[128, 128], mode="train"):
        self.mode = mode
        self.img_dir = img_dir
        self.image_pathes = sorted(os.listdir(img_dir))
        self.wh = wh



    def __len__(self):
        return len(self.image_pathes)

    def get_img(self, index, mode):
        img_path = self.image_pathes[index]
        img = cv2.imread(os.path.join(self.img_dir, img_path))
        try:
            img = cv2.resize(img, (self.wh))

        except:
            print(img_path)
        return img

    def get_label(self, index):
        img_path = self.image_pathes[index]
        label = int(img_path.split("_")[0])
        return label

    def train_augument(self, x):
        img = transforms.ToPILImage()(x)
        img = transforms.ToTensor()(img)

        img = transforms.RandomResizedCrop(224)(img)
        img = transforms.RandomHorizontalFlip()(img)
        img = transforms.RandomRotation(30)(img)
        img = transforms.ColorJitter(brightness=0.3)(img)
        img = transforms.ColorJitter(contrast=0.3)(img)
        img = transforms.ColorJitter(saturation=0.5)(img)
        img = transforms.ColorJitter(hue=0.5)(img)
        img = transforms.GaussianBlur(3)(img)
        img = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img)
        return img
    def test_augument(self, x):
        img = transforms.ToPILImage()(x)
        img = transforms.ToTensor()(img)
        img = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img)
        return  img
    def __getitem__(self, index):

        X = self.get_img(index, self.mode)
        if self.mode == "train":
            X = self.train_augument(X)
        elif self.mode == "val" or self.mode == "test":
            X = self.test_augument(X)
        else:
            print("mode is wrong")

        if self.mode == "train" or self.mode == "val":
            Y = self.get_label(index)
            return X, Y
        elif self.mode == "test":
            return X

