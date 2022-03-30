# main.py
import os
import torch
import argparse
import numpy as np
from torch import nn
from gensim.models import word2vec
from sklearn.model_selection import train_test_split
import tools
import dataset
import model
from train import training
from test import  testing
import random

use_unlabel_data = 1
use_pretrained_model = 0
label_unlabel = 1
# 通過 torch.cuda.is_available() 的回傳值進行判斷是否有使用 GPU 的環境，如果有的話 device 就設為 "cuda"，沒有的話就設為 "cpu"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 處理好各個 data 的路徑
cwp = os.getcwd()
root = cwp + "/../"
train_with_label = root + 'data/training_label.txt'
train_no_label = root + 'data/training_nolabel.txt'
testing_data = root + 'data/testing_data.txt'

w2v_path = root +  'ckpt/w2v_all.model' # 處理 word to vec model 的路徑

# 定義句子長度、要不要固定 embedding、batch 大小、要訓練幾個 epoch、learning rate 的值、model 的資料夾路徑
sen_len = 20
fix_embedding = True # fix embedding during training
batch_size = 256
epoch = 5
lr = 0.001
model_dir = root + "ckpt" # model directory for checkpoint model

print("loading data ...") # 把 'training_label.txt' 跟 'training_nolabel.txt' 讀進來
train_x, y = tools.load_training_data(train_with_label)

# 對 input 跟 labels 做預處理
preprocess = dataset.Preprocess(train_x, sen_len, w2v_path=w2v_path)
embedding = preprocess.make_embedding(load=True)
train_x = preprocess.sentence_word2idx()
y = preprocess.labels_to_tensor(y)

if use_unlabel_data == 1:
    train_x_no_label = tools.load_training_data(train_no_label)
    preprocess = dataset.Preprocess(train_x_no_label, sen_len, w2v_path=w2v_path)
    embedding = preprocess.make_embedding(load=True)
    train_x_no_label = preprocess.sentence_word2idx()

    # unlabel_y = []
    # for label in range(train_x_no_label.shape[0]):
    #     unlabel_y.append(-1)
    # np.savetxt(root + "data/unlabel_label.txt", unlabel_y, fmt='%f')
    # raise
    unlabel_y = np.loadtxt(root + "data/unlabel_label.txt")
    unlabel_y = torch.LongTensor(unlabel_y)
    indexx = torch.where(unlabel_y>=0)
    # unlabel 的数据 太多了，只选取180000试试，否则没作用
    train_x_no_label = train_x_no_label[indexx]
    unlabel_y = unlabel_y[indexx]

    # unlabel_dataset = dataset.TwitterDataset(X=train_x_no_label, y=None)


# 製作一個 model 的對象
if use_pretrained_model==0:
    model = model.LSTM_Net(embedding, embedding_dim=250, hidden_dim=150, num_layers=1, dropout=0.5, fix_embedding=fix_embedding)
else:
    model = torch.load(root + "ckpt" + "/ckpt.model")
model = model.to(device) # device為 "cuda"，model 使用 GPU 來訓練（餵進去的 inputs 也需要是 cuda tensor）


# 把 data 分為 training data 跟 validation data（將一部份 training data 拿去當作 validation data）
X_train, X_val, y_train, y_val = train_x[:180000], train_x[180000:], y[:180000], y[180000:]
if use_unlabel_data == 1:
    sample_index = random.sample(range(0, train_x_no_label.shape[0]), train_x_no_label.shape[0])
    X_train = torch.vstack((X_train, train_x_no_label))
    y_train = torch.hstack((y_train, unlabel_y))
    # X_train = torch.vstack((X_train, unlabel_dataset[sample_index]))
    # y_train = torch.hstack((y_train, unlabel_y[sample_index]))
    # X_train = unlabel_dataset
    # y_train = unlabel_y

# 把 data 做成 dataset 供 dataloader 取用
train_dataset = dataset.TwitterDataset(X=X_train, y=y_train)
val_dataset = dataset.TwitterDataset(X=X_val, y=y_val)

# 把 data 轉成 batch of tensors
train_loader = torch.utils.data.DataLoader(dataset = train_dataset,
                                            batch_size = batch_size,
                                            shuffle = True,
                                            num_workers = 8)

val_loader = torch.utils.data.DataLoader(dataset = val_dataset,
                                            batch_size = batch_size,
                                            shuffle = False,
                                            num_workers = 8)

# 開始訓練
training(batch_size, epoch, lr, model_dir, train_loader, val_loader, model, device)






# label unlabel
if label_unlabel:
    train_x_no_label = tools.load_training_data(train_no_label)
    preprocess = dataset.Preprocess(train_x_no_label, sen_len, w2v_path=w2v_path)
    embedding = preprocess.make_embedding(load=True)
    train_x_no_label = preprocess.sentence_word2idx()
    unlabel_dataset = dataset.TwitterDataset(X=train_x_no_label, y=None)

    unlabel_loader = torch.utils.data.DataLoader(dataset = unlabel_dataset,
                                              batch_size = batch_size,
                                              shuffle=False,
                                              num_workers = 8)



    unlabel_label = testing(batch_size, test_loader=unlabel_loader, model = model, device="cuda")
    unlabel_y = np.loadtxt(root + "data/unlabel_label.txt")
    for i in range(len(unlabel_y)):
        if unlabel_y[i] == -1:
            if unlabel_label[i] > 0.8:
                unlabel_y[i] = 1
            if unlabel_label[i] < 0.2:
                unlabel_y[i] = 0
    np.savetxt(root + "data/unlabel_label.txt", unlabel_y, fmt='%f')