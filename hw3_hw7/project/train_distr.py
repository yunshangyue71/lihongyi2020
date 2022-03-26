import sys
sys.path.append("./")
import torch
import os
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.models as models
from dataset.dataset_dist import *
from model.base_hw7 import *
from torchsummary import summary
import pickle
import numpy as np
import time

def loss_fn_kd(outputs, labels, teacher_outputs, T=20, alpha=0.5):
    # 一般的Cross Entropy
    hard_loss = F.cross_entropy(outputs, labels) * (1. - alpha)
    # 讓logits的log_softmax對目標機率(teacher的logits/T後softmax)做KL Divergence。
    soft_loss = nn.KLDivLoss(reduction='batchmean')(F.log_softmax(outputs/T, dim=1),
                             F.softmax(teacher_outputs/T, dim=1)) * (alpha * T * T)
    return hard_loss + soft_loss

def get_dataloader(mode='training', batch_size=32):

    assert mode in ['training', 'testing', 'validation']

    dataset = MyDataset(
        f'data/food-11/{mode}',
        transform=trainTransform if mode == 'training' else testTransform)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(mode == 'training'),
        num_workers=8, pin_memory=True)

    return dataloader


# --------------将soft label 保存下来，就不用每次运行了
# batch_size = 1
# lr = 1e-3
# epochs = 200
# print_model_size = 0

# teacher_net = models.resnet18(pretrained=False, num_classes=11).cuda()
# if print_model_size:
#     summary(model=teacher_net, batch_size=1, input_size=(3,256,256))
        
# teacher_net.load_state_dict(torch.load(f'ckpt/teacher_resnet18.bin'))
# train_dataloader = get_dataloader('training', batch_size=batch_size)
# valid_dataloader = get_dataloader('validation', batch_size=batch_size)

# path_soft_label = {}
# for now_step, batch_data in enumerate(valid_dataloader):
#     # 清空 optimizer
    
#     # 處理 input
#     inputs, hard_labels,  path = batch_data
#     inputs = inputs.cuda()
#     hard_labels = torch.LongTensor(hard_labels).cuda()
#     # 因為Teacher沒有要backprop，所以我們使用torch.no_grad
#     # 告訴torch不要暫存中間值(去做backprop)以浪費記憶體空間。
#     with torch.no_grad():
#         soft_labels = teacher_net(inputs)
#         # print(path)
#         # print(soft_labels)
#         # raise
#     path_soft_label[path[0]] = soft_labels.cpu().numpy()

# file = open("data/validation.pkl", "wb")
# pickle.dump(path_soft_label, file)
# file.close()




# ---------------------knowlage distilation-----------------
batch_size = 256
lr = 1e-3
epochs = 200
load_model_flag = 1
print_model_size = 0

train_dataloader = get_dataloader('training', batch_size=batch_size)
valid_dataloader = get_dataloader('validation', batch_size=batch_size)

student_net = StudentNet().cuda()
student_net = StudentNetOffice().cuda()

if print_model_size:
    summary(model=student_net, batch_size=1, input_size=(3,256,256))
file = open("data/training.pkl", "rb")
train_soft_label = pickle.load(file)
file2 = open("data/validation.pkl", "rb")
valid_soft_label = pickle.load(file2)

# teacher_net.load_state_dict(torch.load(f'ckpt/teacher_resnet18.bin'))
optimizer = optim.AdamW(student_net.parameters(), lr=lr)
def run_epoch(dataloader, update=True, alpha=0.5):
    total_num, total_hit, total_loss = 0, 0, 0
    for now_step, batch_data in enumerate(dataloader):
        optimizer.zero_grad()
        inputs, hard_labels, pathes = batch_data
        inputs = inputs.cuda()
        hard_labels = torch.LongTensor(hard_labels).cuda()
        # print(pathes.shape)
        # with torch.no_grad():
            # soft_labels = teacher_net(inputs)
        if update:
            soft_labels = []
            for path in pathes:
                soft_labels.append(train_soft_label[path])
            soft_labels = torch.tensor(np.vstack(soft_labels)).cuda()
        else:
            soft_labels = []
            for path in pathes:
                soft_labels.append(valid_soft_label[path])
            soft_labels = torch.tensor(np.vstack(soft_labels)).cuda()
        # print(soft_labels)
        # print(hard_labels)
        # raise
        if update:
            logits = student_net(inputs)
            # 使用我們之前所寫的融合soft label&hard label的loss。
            # T=20是原始論文的參數設定。
            loss = loss_fn_kd(logits, hard_labels, soft_labels, 20, alpha)
            loss.backward()
            optimizer.step()    
        else:
            # 只是算validation acc的話，就開no_grad節省空間。
            with torch.no_grad():
                logits = student_net(inputs)
                loss = loss_fn_kd(logits, hard_labels, soft_labels, 20, alpha)
            
        total_hit += torch.sum(torch.argmax(logits, dim=1) == hard_labels).item()
        total_num += len(inputs)

        total_loss += loss.item() * len(inputs)
    return total_loss / total_num, total_hit / total_num

# TeacherNet永遠都是Eval mode.
# teacher_net.eval()
if load_model_flag == 1:
    student_net.load_state_dict(torch.load("ckpt/student_office_trained_by_me_model.bin"))
now_best_acc = 0
for epoch in range(epochs):
    epoch_start_time = time.time()
    student_net.train()
    train_loss, train_acc = run_epoch(train_dataloader, update=True)
    student_net.eval()
    valid_loss, valid_acc = run_epoch(valid_dataloader, update=False)

    # 存下最好的model。
    if valid_acc > now_best_acc:
        now_best_acc = valid_acc
        torch.save(student_net.state_dict(), 'ckpt/student_office_trained_by_me_model.bin')
    epoch_end_time = time.time()
    print('epoch {:>3d}:  {:2.2f} sec(s) train loss: {:6.4f}, acc {:6.4f} valid loss: {:6.4f}, acc {:6.4f} lr {: 0.7f}'.format(
        epoch, epoch_end_time - epoch_start_time, train_loss, train_acc, valid_loss, valid_acc, lr))
