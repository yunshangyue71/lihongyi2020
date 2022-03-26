import re
import torch
from glob import glob
from PIL import Image
import torchvision.transforms as transforms

class MyDataset(torch.utils.data.Dataset):

    def __init__(self, folderName, transform=None):
        self.transform = transform
        self.data = []
        self.label = []
        self.img_pathes =sorted(glob(folderName + '/*.jpg'))
        for img_path in self.img_pathes:
            try:
                # Get classIdx by parsing image path
                class_idx = int(re.findall(re.compile(r'\d+'), img_path)[1])
            except:
                # if inference mode (there's no answer), class_idx default 0
                class_idx = 0

            image = Image.open(img_path)
            # Get File Descriptor
            image_fp = image.fp
            image.load()
            # Close File Descriptor (or it'll reach OPEN_MAX)
            image_fp.close()

            self.data.append(image)
            self.label.append(class_idx)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        image = self.data[idx]
        if self.transform:
            image = self.transform(image)
        return image, self.label[idx], self.img_pathes[idx]


# 李宏毅提供的网络的数据增强的形式，
trainTransform = transforms.Compose([
    transforms.RandomCrop(256, pad_if_needed=True, padding_mode='symmetric'),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
])

# trainTransform = transforms.Compose([
#     transforms.RandomResizedCrop((256, 256)),
#     transforms.RandomHorizontalFlip(),

#      transforms.ColorJitter(brightness=0.2),
#             transforms.ColorJitter(contrast=0.2),
#             transforms.ColorJitter(saturation=0.2),
#             transforms.ColorJitter(hue=0.2),
#             transforms.GaussianBlur(3),
#     transforms.RandomAffine(degrees=30,  # 旋转
#                                     scale=(0.8, 1),
#                                     shear=(20, 20),  # -20 20
#                                     translate=(0.1, 0.1)  # 平移
#                                     ),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# ])
testTransform = transforms.Compose([
    transforms.CenterCrop(256),
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
