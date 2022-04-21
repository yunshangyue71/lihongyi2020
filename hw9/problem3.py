import os
import glob
import torch
import numpy as np
from matplotlib.pylab import plt

from dataset import Image_Dataset, DataLoader, preprocess
from model import AE
from inference import inference, predict
from function import cal_acc
from config import *

root = "/media/q/data/lihongyi2020data/hw9/"

model = AE().cuda()
checkpoints_list = sorted(glob.glob(root + 'ckpt/'+ func + 'checkpoint_*.pth'), key= lambda x: int(os.path.splitext(os.path.basename(x))[0].split('_')[-1]))
model.load_state_dict(torch.load(root + 'ckpt/'+ func + 'last_checkpoint.pth'))
model.eval()

# load data
valX = np.load(root + 'valX.npy')
valY = np.load(root + 'valY.npy')

trainX = np.load(root  + 'trainX.npy')
trainX_preprocessed = preprocess(trainX)
dataset = Image_Dataset(trainX_preprocessed)
dataloader = DataLoader(dataset, batch_size=64, shuffle=False)

points = []
with torch.no_grad():
    for i, checkpoint in enumerate(checkpoints_list):
        print('[{}/{}] {}'.format(i+1, len(checkpoints_list), checkpoint))
        model.load_state_dict(torch.load(checkpoint))
        model.eval()
        err = 0
        n = 0
        for x in dataloader:
            x = x.cuda()
            _, rec = model(x)
            err += torch.nn.MSELoss(reduction='sum')(x, rec).item()
            n += x.flatten().size(0)
        print('Reconstruction error (MSE):', err/n)
        latents = inference(X=valX, model=model)
        pred, X_embedded = predict(latents)
        acc = cal_acc(valY, pred)
        print('Accuracy:', acc)
        points.append((err/n, acc))

ps = list(zip(*points))
plt.figure(figsize=(6,6))
plt.subplot(211, title='Reconstruction error (MSE)').plot(ps[0])
plt.subplot(212, title='Accuracy (val)').plot(ps[1])
plt.savefig("./image_saved/"+ func + "p3.png")
plt.show()