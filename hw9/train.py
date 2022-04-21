from function import *
from model import *
from dataset import *

import torch
from torch import optim
from config import *

same_seeds(0)

model = AE().cuda()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-5)

model.train()
n_epoch = 100

# 準備 dataloader, model, loss criterion 和 optimizer
img_dataloader = DataLoader(img_dataset, batch_size=64, shuffle=True)

# 主要的訓練過程
for epoch in range(n_epoch):
    epoch_loss = 0
    for data in img_dataloader:
        img = data
        img = img.cuda()

        output1, output = model(img)
        loss = criterion(output, img)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    if (epoch + 1) % 10 == 0:
        torch.save(model.state_dict(), '/media/q/data/lihongyi2020data/hw9/ckpt/'+ func +'checkpoint_{}.pth'.format(epoch + 1))
    print('epoch [{}/{}], loss:{:.5f}'.format(epoch + 1, n_epoch, epoch_loss))

# 訓練完成後儲存 model
torch.save(model.state_dict(), '/media/q/data/lihongyi2020data/hw9/ckpt/'+ func +'last_checkpoint.pth')

