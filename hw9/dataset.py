import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

root = "/media/q/data/lihongyi2020data/hw9/"

def preprocess(image_list):
    """ Normalize Image and Permute (N,H,W,C) to (N,C,H,W)
    Args:
      image_list: List of images (9000, 32, 32, 3)
    Returns:
      image_list: List of images (9000, 3, 32, 32)
    """
    image_list = np.array(image_list)
    image_list = np.transpose(image_list, (0, 3, 1, 2))
    image_list = (image_list / 255.0) * 2 - 1
    image_list = image_list.astype(np.float32)
    return image_list



class Image_Dataset(Dataset):
    def __init__(self, image_list):
        self.image_list = image_list
    def __len__(self):
        return len(self.image_list)

    def __train_aug(self):
        return transforms.Compose([
            transforms.ToPILImage(),

            # transforms.RandomHorizontalFlip(),
            # transforms.RandomAffine(degrees=30,  # 旋转
            #                         scale=(0.8, 1),
            #                         shear=(20, 20),  # -20 20
            #                         translate=(0.1, 0.1)  # 平移
            #                         ),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            # transforms.RandomErasing(p=0.5,
            #                          scale=(0.02, 0.1),  # block的面积
            #                          ratio=(0.3, 3.3),  # block的长宽比
            #                          value="random"  # 设置成字符串会随机填充
            #                          ),
        ])

    def __getitem__(self, idx):
        images = self.image_list[idx]
        images = torch.from_numpy(images)
        # images = images / 255.0#self.__train_aug()(images)
        return images


trainX = np.load(root  + 'trainX.npy')
# trainX_mean = np.mean(trainX, axis=0)
# trainX = trainX - trainX_mean
trainX_preprocessed = preprocess(trainX)
img_dataset = Image_Dataset(trainX_preprocessed)

