import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

# from inference import inference, predict
from dataset import preprocess, Image_Dataset
from model import AE
from config import *

root = "/media/q/data/lihongyi2020data/hw9/"

trainX = np.load(root  + 'trainX.npy')
trainX_preprocessed = preprocess(trainX)
img_dataset = Image_Dataset(trainX_preprocessed)
model = AE().cuda()
model.load_state_dict(torch.load(root + 'ckpt/' + func + 'last_checkpoint.pth'))
model.eval()


class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for i in range(3):
            tensor[i] = tensor[i] * self.std[i] + self.mean[i]
        # for t, m, s in zip(tensor, self.mean, self.std):
        #     t = t * s + m
            # The normalize code -> t.sub_(m).div_(s)
        return tensor


# unorm = UnNormalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
# unorm(tensor)

# 畫出原圖
plt.figure(figsize=(10, 4))
indexes = [1, 2, 3, 6, 7, 9]
imgs = trainX[indexes,]
for i, img in enumerate(imgs):
    plt.subplot(2, 6, i + 1, xticks=[], yticks=[])
    plt.imshow(img)



# # 畫出 reconstruct 的圖
inp = trainX_preprocessed[indexes,]
inp_ = []
for  i in range(inp.shape[0]):
    a = inp[i].astype(np.uint8)
    a = transforms.ToPILImage()(a)
    a = transforms.ToTensor()(a)
    # a = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(a)
    a = torch.unsqueeze(a, 0)
    inp_.append(a)
inp_ = torch.vstack(inp_)
inp_ = inp_.cuda()


latents, recs = model(inp_)
unorm = UnNormalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
# unorm = UnNormalize(mean=(0.406, 0.456, 0.485), std=(0.225, 0.224,0.229))

# for i in range(recs.shape[0]):
#     recs[i] = unorm(recs[i])
# recs = recs * 255.0
# recs = ((recs + 1) / 2).cpu().detach().numpy()
recs = recs.cpu().detach().numpy()
recs = recs.transpose(0, 2, 3, 1)

for i, img in enumerate(recs):
    plt.subplot(2, 6, 6 + i + 1, xticks=[], yticks=[])
    plt.imshow(img)
plt.savefig("./image_saved/" + func + "p2.png")
plt.tight_layout()
plt.show()
