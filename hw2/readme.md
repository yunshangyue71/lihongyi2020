# 介绍
data中的dataset_lihongyi 是官方提供的数据集  
因为test dataset 没有提供，因此自己在官网找到了原始数据集用于自己的测试评估
# baseline 原始数据, 将离散的用数字表示，去掉nan的样本
这个效果在0.77, 但是训练的时候，验证集跳动很大  
epoch: 60, train_loss: 5.5122, train_acc: 0.7008, val_loss: 4.1545, val_acc: 0.7745, lr: 0.001000  
epoch: 80, train_loss: 5.3903, train_acc: 0.7074, val_loss: 4.1575, val_acc: 0.7743, lr: 0.001000  
epoch: 100, train_loss: 5.4438, train_acc: 0.7045, val_loss: 5.0055, val_acc: 0.7283, lr: 0.001000  
epoch: 120, train_loss: 5.5379, train_acc: 0.6994, val_loss: 13.8880, val_acc: 0.2461, lr: 0.001000  
epoch: 140, train_loss: 5.4623, train_acc: 0.7035, val_loss: 6.0335, val_acc: 0.6725, lr: 0.001000  
epoch: 160, train_loss: 5.4805, train_acc: 0.7025, val_loss: 4.1545, val_acc: 0.7745, lr: 0.001000  
epoch: 180, train_loss: 5.4180, train_acc: 0.7059, val_loss: 4.1575, val_acc: 0.7743, lr: 0.001000  
epoch: 200, train_loss: 5.5142, train_acc: 0.7007, val_loss: 4.1545, val_acc: 0.7745, lr: 0.001000

# a baseline
0.7765964460595389 4.115245449797192
效果相差不多
# b a + normalization
没有了跳动, 有些feature的数值比较大， 梯度下降的系数对于这些特征改变一点，对最终的结果影响很大，所以会跳动  
epoch: 20, train_loss: 0.3753, train_acc: 0.7610, val_loss: 0.3726, val_acc: 0.7603, lr: 0.010000  
epoch: 40, train_loss: 0.3753, train_acc: 0.7608, val_loss: 0.3726, val_acc: 0.7604, lr: 0.010000  
epoch: 60, train_loss: 0.3753, train_acc: 0.7609, val_loss: 0.3726, val_acc: 0.7605, lr: 0.010000  
epoch: 80, train_loss: 0.3753, train_acc: 0.7609, val_loss: 0.3726, val_acc: 0.7604, lr: 0.010000  
epoch: 100, train_loss: 0.3753, train_acc: 0.7609, val_loss: 0.3726, val_acc: 0.7605, lr: 0.010000  
epoch: 120, train_loss: 0.3753, train_acc: 0.7608, val_loss: 0.3726, val_acc: 0.7605, lr: 0.010000 

# c b +L2 loss
这个影响不大
epoch: 180, train_loss: 0.3756, train_acc: 0.7602, val_loss: 0.3727, val_acc: 0.7598, lr: 0.000100  
epoch: 200, train_loss: 0.3756, train_acc: 0.7602, val_loss: 0.3727, val_acc: 0.7598, lr: 0.000100  

# d c + onehot 
test acc 0.7920285582633819
# e d + onehot 离散性的不加 normalization
test acc 还是0.79

# 总结
normalization还是要做的，做了以后训练更加稳定  
自己用于训练的数据和lihongyi-2020的不太一样，他的数据是有504个特征，（数据是一样的，特征经过重新选择，选择的工作也是官网做的） 
他的数据baseline就可以达到0.88， 说明了特征工程的重要性  
如果使用XGBoost或者RF等方法，可以增加精度