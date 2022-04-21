import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import glob

from model import *
from inference import inference, predict
from function import cal_acc
from plot import *
from config import *

root = "/media/q/data/lihongyi2020data/hw9/"

valX = np.load(root + 'valX.npy')
valY = np.load(root + 'valY.npy')

# ==============================================
#  我們示範 basline model 的作圖，
#  report 請同學另外還要再畫一張 improved model 的圖。
# ==============================================
model = AE().cuda()
# model.load_state_dict(torch.load(root +  'ckpt/' + func +  'last_checkpoint.pth'))
model.load_state_dict(torch.load(root +  'ckpt/' + "8/"+  'checkpoint_80.pth'))
model.eval()
latents = inference(valX, model)
pred_from_latent, emb_from_latent = predict(latents)
acc_latent = cal_acc(valY, pred_from_latent)
print('The clustering accuracy is:', acc_latent)
print('The clustering result:')
plot_scatter(emb_from_latent, valY, savefig="./image_saved/" + func + 'p1.png')


