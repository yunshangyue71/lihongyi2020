import os , sys
sys.path.append("./")
import numpy as np
from torch.optim import lr_scheduler

from torch.utils.data import DataLoader
import time
from model.base import Classifier
from project.dataset.dataset import ImgDataset
from model.ResNet import *
from model.base_hw7 import *
from project.loss import loss, focal_loss
from torchsummary import summary

class Train():
    def __init__(self):
        self.batch_size = 160
        self.num_epoch = 500
        self.wh = (256, 256)
        self.wh_v = (256, 256) #(224,224)
        self.lr = 1e-6
        self.print_model_size = 0

    def get_model(self):
        # model = Classifier().cuda()
        # model = ResNet(ResnetBasic, [2, 2, 2, 2]).cuda()
        model = StudentNet().cuda()
        return model

    def get_optimizer(self, model):
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)  # optimizer 使用 Adam
        return optimizer

    def save(self, model, path):
        # 模型保存
        # torch.save(model, 'model.pkl')
        torch.save(model, path)

    def load(self, path):
        # 模型加载
        model = torch.load('model.pkl')
        model = torch.load(path)
        return model

    def save2(self, model, path):
        # 模型参数保存
        torch.save(model.state_dict(), path)
        # torch.save(model.state_dict(), 'model_param.pkl')
        # 模型参数加载
        # model = ModelClass(...)
    def load2(self, model, path):
        # model.load_state_dict(torch.load('model_param.pkl'))
        model.load_state_dict(torch.load(path))
        return model

    def get_data(self):
        root = "/root/autodl-tmp/" #"/media/q/data/lihongyi_2020/hw3/"
        workspace_dir = root + 'data/food-11/'

        train_data = ImgDataset(workspace_dir + "training/", wh=self.wh, mode="train")
        val_data = ImgDataset(workspace_dir + "validation/",wh= self.wh_v,  mode = "val")
        train_loader = DataLoader(train_data,  batch_size=self.batch_size, shuffle=True, num_workers=8, pin_memory=True)
        val_loader = DataLoader(val_data,  batch_size=self.batch_size, shuffle=False, num_workers=8, pin_memory=True)

        return train_loader, val_loader, train_data.__len__(), val_data.__len__()

    def train(self):
        model = self.get_model()
        optimizer = self.get_optimizer(model)

        train_loader, val_loader, train_num, val_num = self.get_data()

        model = self.load2(model,  "ckpt/model_base_deepwise_4M.pkl")
        if self.print_model_size:
            summary(model=model, batch_size=1, input_size=(3, self.wh[0], self.wh[1]))
            return 0
        for epoch in range(self.num_epoch):
            epoch_start_time = time.time()
            train_acc = 0.0
            train_loss = 0.0
            val_acc = 0.0
            val_loss = 0.0
            val_cls_acc = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            confuse = np.zeros((11, 11))
            val_cls_num = [362, 144, 500, 327, 326, 449, 147, 96, 347, 500, 232]

            model.train()  # 確保 model 是在 train model (開啟 Dropout 等...)
            for i, data in enumerate(train_loader):
                optimizer.zero_grad()  # 用 optimizer 將 model 參數的 gradient 歸零
                train_pred = model(data[0].cuda())  # 利用 model 得到預測的機率分佈 這邊實際上就是去呼叫 model 的 forward 函數

                # cal loss
                cross_loss = loss.cross_entropy(train_pred, data[1].cuda())  # 計算 loss （注意 prediction 跟 label 必須同時在 CPU 或是 GPU 上）
                # fl = focal_loss.floss(train_pred, data[1].cuda())
                l2_loss = loss.L2(model) * 1e-6
                # batch_loss =  fl
                batch_loss = cross_loss + l2_loss
                batch_loss.backward()  # 利用 back propagation 算出每個參數的 gradient
                optimizer.step()  # 以 optimizer 用 gradient 更新參數值

                # cal acc
                train_acc += np.sum(np.argmax(train_pred.cpu().data.numpy(), axis=1) == data[1].numpy())
                train_loss += batch_loss.item()
            if epoch+1 > 0 and (epoch+1) % 10 == 0:
                self.save2(model, "ckpt/model_base_deepwise_4M.pkl")
            model.eval()
            with torch.no_grad():
                for i, data in enumerate(val_loader):
                    val_pred = model(data[0].cuda())
                    cross_loss = loss.cross_entropy(val_pred, data[1].cuda())
                    # fl = focal_loss.floss(val_pred, data[1].cuda())
                    l2_loss = loss.L2(model) * 3e-5
                    batch_loss = cross_loss + l2_loss
                    val_tf = np.argmax(val_pred.cpu().data.numpy(), axis=1) == data[1].numpy()
                    val_confuse = np.argmax(val_pred.cpu().data.numpy(), axis=1)
                    ls = list(data[1].numpy())
                    for id in range(len(ls)):
                        cls = ls[id]
                        pre_cls = val_confuse[id]
                        confuse[cls][pre_cls]+=1
                        if val_tf[id] == True:
                            val_cls_acc[cls] += 1

                    val_acc += np.sum(val_tf)
                    val_loss += batch_loss.item()

                # 將結果 print 出來
                print('[%03d/%03d] %2.2f sec(s) Train Acc: %3.6f Loss: %3.6f | Val Acc: %3.6f loss: %3.6f' % \
                      (epoch + 1, self.num_epoch, time.time() - epoch_start_time,
                       train_acc / train_num,  train_loss / train_num, val_acc / val_num, val_loss / val_num),
                      optimizer.state_dict()['param_groups'][0]['lr'])
      

if __name__ == '__main__':
    print(os.getcwd())
    trainer = Train()
    trainer.train()

