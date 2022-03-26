# 介绍
这里我并不打算训练出一个高精度的模型。我以官方的baseline为基准
训练一个小的模型，来看精度降低了多少， 以及同等大小的情况下，那种方法效果最好。另外一点也是为了减少训练时间
# 网络设计
## a base line的卷积替换为deepwise
6 -> a  
层数不变，feature size 不变
### model size 144->1.73 下降了83倍
3-64-128-256-512-512  
Total params: 454,667  
Trainable params: 454,667  
Non-trainable params: 0  
Input size (MB): 0.75  
Forward/backward pass size (MB): 185.00  
Params size (MB): 1.73  
Estimated Total Size (MB): 187.49  
### 精度由 0.81-> 0.67 下降了14%
[014/500] 27.51 sec(s) Train Acc: 0.674539 Loss: 0.006149 | Val Acc: 0.666472 loss: 0.007061 1e-05  
[015/500] 27.65 sec(s) Train Acc: 0.673120 Loss: 0.006113 | Val Acc: 0.672303 loss: 0.006895 1e-05  
[016/500] 27.88 sec(s) Train Acc: 0.679404 Loss: 0.006095 | Val Acc: 0.670262 loss: 0.006878 1e-05  
[017/500] 27.63 sec(s) Train Acc: 0.676870 Loss: 0.006139 | Val Acc: 0.672012 loss: 0.006941 1e-05  
[018/500] 27.23 sec(s) Train Acc: 0.678695 Loss: 0.006096 | Val Acc: 0.678426 loss: 0.006862 1e-05  
[019/500] 27.58 sec(s) Train Acc: 0.680418 Loss: 0.006099 | Val Acc: 0.666181 loss: 0.007012 1e-05  
[020/500] 27.37 sec(s) Train Acc: 0.674944 Loss: 0.006148 | Val Acc: 0.671720 loss: 0.006911 1e-05  
[021/500] 28.12 sec(s) Train Acc: 0.673627 Loss: 0.006135 | Val Acc: 0.669388 loss: 0.006971 1e-05  

[001/500] 28.84 sec(s) Train Acc: 0.676161 Loss: 0.006146 | Val Acc: 0.668805 loss: 0.006917 1e-06  
[002/500] 29.36 sec(s) Train Acc: 0.679607 Loss: 0.006086 | Val Acc: 0.669096 loss: 0.006892 1e-06  
[003/500] 28.72 sec(s) Train Acc: 0.679607 Loss: 0.006131 | Val Acc: 0.672012 loss: 0.006897 1e-06  
[004/500] 28.79 sec(s) Train Acc: 0.684573 Loss: 0.006018 | Val Acc: 0.672303 loss: 0.006899 1e-06  
[005/500] 28.60 sec(s) Train Acc: 0.680823 Loss: 0.006061 | Val Acc: 0.668222 loss: 0.006924 1e-06  
[006/500] 29.95 sec(s) Train Acc: 0.681228 Loss: 0.006061 | Val Acc: 0.670554 loss: 0.006891 1e-06  
[007/500] 28.73 sec(s) Train Acc: 0.677884 Loss: 0.006132 | Val Acc: 0.670262 loss: 0.006893 1e-06  
[008/500] 28.56 sec(s) Train Acc: 0.670282 Loss: 0.006156 | Val Acc: 0.672886 loss: 0.006872 1e-06  
[009/500] 28.48 sec(s) Train Acc: 0.671093 Loss: 0.006177 | Val Acc: 0.675510 loss: 0.006851 1e-06  
[010/500] 27.98 sec(s) Train Acc: 0.676667 Loss: 0.006114 | Val Acc: 0.666472 loss: 0.006935 1e-06  

### 模型大小
144 -> 4.13, 下降了35倍   
Total params: 1,083,595  
Trainable params: 1,083,595  
Non-trainable params: 0  
Input size (MB): 0.75  
Forward/backward pass size (MB): 168.01  
Params size (MB): 4.13  
Estimated Total Size (MB): 172.89  
### 精度 0.67 - >0.72 5%个点
[278/500] 27.54 sec(s) Train Acc: 0.734137 Loss: 0.004964 | Val Acc: 0.700583 loss: 0.007025 0.0001  
[279/500] 27.88 sec(s) Train Acc: 0.732313 Loss: 0.004984 | Val Acc: 0.716327 loss: 0.006858 0.0001  
[280/500] 27.45 sec(s) Train Acc: 0.744577 Loss: 0.004823 | Val Acc: 0.695627 loss: 0.007157 0.0001  
[281/500] 27.43 sec(s) Train Acc: 0.732212 Loss: 0.004935 | Val Acc: 0.713120 loss: 0.006968 0.0001  
[282/500] 27.45 sec(s) Train Acc: 0.734644 Loss: 0.004977 | Val Acc: 0.687755 loss: 0.007773 0.0001  
[283/500] 27.16 sec(s) Train Acc: 0.746706 Loss: 0.004868 | Val Acc: 0.705539 loss: 0.007105 0.0001  

[016/500] 28.02 sec(s) Train Acc: 0.765153 Loss: 0.004401 | Val Acc: 0.718076 loss: 0.006719 1e-05  
[017/500] 28.62 sec(s) Train Acc: 0.762923 Loss: 0.004508 | Val Acc: 0.722449 loss: 0.006691 1e-05  
[018/500] 27.99 sec(s) Train Acc: 0.758767 Loss: 0.004514 | Val Acc: 0.720117 loss: 0.006724 1e-05  
[019/500] 28.02 sec(s) Train Acc: 0.764139 Loss: 0.004512 | Val Acc: 0.715452 loss: 0.006762 1e-05  
[020/500] 28.51 sec(s) Train Acc: 0.771133 Loss: 0.004433 | Val Acc: 0.720117 loss: 0.006720 1e-05  
[005/500] 27.40 sec(s) Train Acc: 0.762822 Loss: 0.004466 | Val Acc: 0.720700 loss: 0.006723 1e-06  
[006/500] 27.09 sec(s) Train Acc: 0.764748 Loss: 0.004405 | Val Acc: 0.718659 loss: 0.006664 1e-06  
[007/500] 28.13 sec(s) Train Acc: 0.764038 Loss: 0.004455 | Val Acc: 0.721574 loss: 0.006678 1e-06  
[008/500] 27.88 sec(s) Train Acc: 0.766673 Loss: 0.004457 | Val Acc: 0.719242 loss: 0.006710 1e-06  
[009/500] 27.54 sec(s) Train Acc: 0.763430 Loss: 0.004401 | Val Acc: 0.721574 loss: 0.006672 1e-06  
[010/500] 27.80 sec(s) Train Acc: 0.759984 Loss: 0.004501 | Val Acc: 0.720408 loss: 0.006625 1e-06  

# 知识蒸馏
## baseline 替换为deepwise后 用
### teacher net 的结构
  
        Layer (type)               Output Shape         Param #  
 
            Conv2d-1          [1, 64, 128, 128]           9,408  
       BatchNorm2d-2          [1, 64, 128, 128]             128  
              ReLU-3          [1, 64, 128, 128]               0  
         MaxPool2d-4            [1, 64, 64, 64]               0  
            Conv2d-5            [1, 64, 64, 64]          36,864  
       BatchNorm2d-6            [1, 64, 64, 64]             128  
              ReLU-7            [1, 64, 64, 64]               0  
            Conv2d-8            [1, 64, 64, 64]          36,864  
       BatchNorm2d-9            [1, 64, 64, 64]             128  
             ReLU-10            [1, 64, 64, 64]               0  
       BasicBlock-11            [1, 64, 64, 64]               0  
           Conv2d-12            [1, 64, 64, 64]          36,864   
      BatchNorm2d-13            [1, 64, 64, 64]             128  
             ReLU-14            [1, 64, 64, 64]               0  
           Conv2d-15            [1, 64, 64, 64]          36,864  
      BatchNorm2d-16            [1, 64, 64, 64]             128  
             ReLU-17            [1, 64, 64, 64]               0  
       BasicBlock-18            [1, 64, 64, 64]               0  
           Conv2d-19           [1, 128, 32, 32]          73,728   
      BatchNorm2d-20           [1, 128, 32, 32]             256  
             ReLU-21           [1, 128, 32, 32]               0  
           Conv2d-22           [1, 128, 32, 32]         147,456  
      BatchNorm2d-23           [1, 128, 32, 32]             256  
           Conv2d-24           [1, 128, 32, 32]           8,192  
      BatchNorm2d-25           [1, 128, 32, 32]             256  
             ReLU-26           [1, 128, 32, 32]               0  
       BasicBlock-27           [1, 128, 32, 32]               0  
           Conv2d-28           [1, 128, 32, 32]         147,456  
      BatchNorm2d-29           [1, 128, 32, 32]             256  
             ReLU-30           [1, 128, 32, 32]               0  
           Conv2d-31           [1, 128, 32, 32]         147,456  
      BatchNorm2d-32           [1, 128, 32, 32]             256  
             ReLU-33           [1, 128, 32, 32]               0  
       BasicBlock-34           [1, 128, 32, 32]               0   
           Conv2d-35           [1, 256, 16, 16]         294,912  
      BatchNorm2d-36           [1, 256, 16, 16]             512  
             ReLU-37           [1, 256, 16, 16]               0  
           Conv2d-38           [1, 256, 16, 16]         589,824  
      BatchNorm2d-39           [1, 256, 16, 16]             512  
           Conv2d-40           [1, 256, 16, 16]          32,768  
      BatchNorm2d-41           [1, 256, 16, 16]             512  
             ReLU-42           [1, 256, 16, 16]               0  
       BasicBlock-43           [1, 256, 16, 16]               0  
           Conv2d-44           [1, 256, 16, 16]         589,824  
      BatchNorm2d-45           [1, 256, 16, 16]             512  
             ReLU-46           [1, 256, 16, 16]               0  
           Conv2d-47           [1, 256, 16, 16]         589,824  
      BatchNorm2d-48           [1, 256, 16, 16]             512  
             ReLU-49           [1, 256, 16, 16]               0  
       BasicBlock-50           [1, 256, 16, 16]               0  
           Conv2d-51             [1, 512, 8, 8]       1,179,648  
      BatchNorm2d-52             [1, 512, 8, 8]           1,024  
             ReLU-53             [1, 512, 8, 8]               0  
           Conv2d-54             [1, 512, 8, 8]       2,359,296  
      BatchNorm2d-55             [1, 512, 8, 8]           1,024  
           Conv2d-56             [1, 512, 8, 8]         131,072  
      BatchNorm2d-57             [1, 512, 8, 8]           1,024  
             ReLU-58             [1, 512, 8, 8]               0  
       BasicBlock-59             [1, 512, 8, 8]               0  
           Conv2d-60             [1, 512, 8, 8]       2,359,296  
      BatchNorm2d-61             [1, 512, 8, 8]           1,024  
             ReLU-62             [1, 512, 8, 8]               0  
           Conv2d-63             [1, 512, 8, 8]       2,359,296  
      BatchNorm2d-64             [1, 512, 8, 8]           1,024  
             ReLU-65             [1, 512, 8, 8]               0  
       BasicBlock-66             [1, 512, 8, 8]               0  
AdaptiveAvgPool2d-67             [1, 512, 1, 1]               0   
           Linear-68                    [1, 11]           5,643  
  
Total params: 11,182,155  
Trainable params: 11,182,155  
Non-trainable params: 0  
  
Input size (MB): 0.75  
Forward/backward pass size (MB): 82.00  
Params size (MB): 42.66  
Estimated Total Size (MB): 125.41   
### a 小模型蒸馏的效果
hard label的效果是72%  
蒸馏的效果是 75% 提升了3个点
官方提供的模型深度多了一些，它的精度更高，可以到0.81 自己没有进行训练  
epoch 125:  20.58 sec(s) train loss: 0.5906, acc 0.9000 valid loss: 0.7740, acc 0.7297   1e-3
epoch 126:  20.46 sec(s) train loss: 0.5881, acc 0.9045 valid loss: 0.7814, acc 0.7187  
epoch 127:  20.55 sec(s) train loss: 0.5909, acc 0.8996 valid loss: 0.8156, acc 0.6915  
epoch 128:  20.24 sec(s) train loss: 0.5836, acc 0.9077 valid loss: 0.7610, acc 0.7420  
epoch 129:  20.30 sec(s) train loss: 0.5828, acc 0.9058 valid loss: 0.7935, acc 0.7125  
epoch 130:  20.11 sec(s) train loss: 0.5843, acc 0.9061 valid loss: 0.7748, acc 0.7222  

epoch   5:  21.29 sec(s) train loss: 0.5969, acc 0.8923 valid loss: 0.7399, acc 0.7542  1e-4  
epoch   6:  22.00 sec(s) train loss: 0.5974, acc 0.8913 valid loss: 0.7410, acc 0.7557  
epoch   7:  21.30 sec(s) train loss: 0.5941, acc 0.8941 valid loss: 0.7379, acc 0.7536  
epoch   8:  21.70 sec(s) train loss: 0.5948, acc 0.8916 valid loss: 0.7367, acc 0.7589  
epoch   9:  21.74 sec(s) train loss: 0.5934, acc 0.8927 valid loss: 0.7411, acc 0.7545  
epoch  10:  21.52 sec(s) train loss: 0.5926, acc 0.8959 valid loss: 0.7388, acc 0.7534  

epoch   5:  21.66 sec(s) train loss: 0.5942, acc 0.8958 valid loss: 0.7354, acc 0.7571  1e-6  
epoch   6:  21.21 sec(s) train loss: 0.5961, acc 0.8905 valid loss: 0.7374, acc 0.7551  
epoch   7:  21.13 sec(s) train loss: 0.5954, acc 0.8917 valid loss: 0.7373, acc 0.7589  
epoch   8:  21.32 sec(s) train loss: 0.5910, acc 0.8923 valid loss: 0.7359, acc 0.7583  
epoch   9:  21.53 sec(s) train loss: 0.5941, acc 0.8915 valid loss: 0.7381, acc 0.7554  
epoch  10:  21.21 sec(s) train loss: 0.5958, acc 0.8891 valid loss: 0.7354, acc 0.7603  

## 官网模型进行蒸馏
### 模型大小

        Layer (type)               Output Shape         Param #

            Conv2d-1          [1, 16, 256, 256]             448
       BatchNorm2d-2          [1, 16, 256, 256]              32
             ReLU6-3          [1, 16, 256, 256]               0
         MaxPool2d-4          [1, 16, 128, 128]               0
            Conv2d-5          [1, 16, 128, 128]             160
       BatchNorm2d-6          [1, 16, 128, 128]              32
             ReLU6-7          [1, 16, 128, 128]               0
            Conv2d-8          [1, 32, 128, 128]             544
         MaxPool2d-9            [1, 32, 64, 64]               0
           Conv2d-10            [1, 32, 64, 64]             320
      BatchNorm2d-11            [1, 32, 64, 64]              64
            ReLU6-12            [1, 32, 64, 64]               0
           Conv2d-13            [1, 64, 64, 64]           2,112
        MaxPool2d-14            [1, 64, 32, 32]               0
           Conv2d-15            [1, 64, 32, 32]             640
      BatchNorm2d-16            [1, 64, 32, 32]             128
            ReLU6-17            [1, 64, 32, 32]               0
           Conv2d-18           [1, 128, 32, 32]           8,320
        MaxPool2d-19           [1, 128, 16, 16]               0
           Conv2d-20           [1, 128, 16, 16]           1,280
      BatchNorm2d-21           [1, 128, 16, 16]             256
            ReLU6-22           [1, 128, 16, 16]               0
           Conv2d-23           [1, 256, 16, 16]          33,024
           Conv2d-24           [1, 256, 16, 16]           2,560
      BatchNorm2d-25           [1, 256, 16, 16]             512
            ReLU6-26           [1, 256, 16, 16]               0
           Conv2d-27           [1, 256, 16, 16]          65,792
           Conv2d-28           [1, 256, 16, 16]           2,560
      BatchNorm2d-29           [1, 256, 16, 16]             512
            ReLU6-30           [1, 256, 16, 16]               0
           Conv2d-31           [1, 256, 16, 16]          65,792
           Conv2d-32           [1, 256, 16, 16]           2,560
      BatchNorm2d-33           [1, 256, 16, 16]             512
            ReLU6-34           [1, 256, 16, 16]               0
           Conv2d-35           [1, 256, 16, 16]          65,792
AdaptiveAvgPool2d-36             [1, 256, 1, 1]               0
           Linear-37                    [1, 11]           2,827

Total params: 256,779  
Trainable params: 256,779  
Non-trainable params: 0  

Input size (MB): 0.75  
Forward/backward pass size (MB): 52.50  
Params size (MB): 0.98  
Estimated Total Size (MB): 54.23  

### 效果
4.5M 变为1M 降低了4个点， 在牺牲一点性能的前提下，可以极大的降低模型大小  
epoch 125:  14.66 sec(s) train loss: 0.6782, acc 0.8123 valid loss: 0.8317, acc 0.6781  1e-3  
epoch 126:  14.70 sec(s) train loss: 0.6787, acc 0.8166 valid loss: 0.8581, acc 0.6703     
epoch 127:  14.07 sec(s) train loss: 0.6755, acc 0.8169 valid loss: 0.8163, acc 0.7003  
epoch 128:  14.16 sec(s) train loss: 0.6723, acc 0.8200 valid loss: 0.8297, acc 0.6980  
epoch 129:  14.18 sec(s) train loss: 0.6758, acc 0.8165 valid loss: 0.8250, acc 0.6933  
epoch 130:  14.49 sec(s) train loss: 0.6778, acc 0.8140 valid loss: 0.8312, acc 0.6810  

epoch  14:  13.42 sec(s) train loss: 0.6211, acc 0.8731 valid loss: 0.8059, acc 0.7067 lr  0.0000010  
epoch  15:  14.23 sec(s) train loss: 0.6248, acc 0.8706 valid loss: 0.8048, acc 0.7085 lr  0.0000010  
epoch  16:  13.88 sec(s) train loss: 0.6224, acc 0.8750 valid loss: 0.8051, acc 0.7055 lr  0.0000010  
epoch  17:  13.31 sec(s) train loss: 0.6232, acc 0.8761 valid loss: 0.8047, acc 0.7085 lr  0.0000010  
epoch  18:  13.50 sec(s) train loss: 0.6208, acc 0.8777 valid loss: 0.8044, acc 0.7076 lr  0.0000010  
epoch  19:  13.71 sec(s) train loss: 0.6217, acc 0.8707 valid loss: 0.8056, acc 0.7082 lr  0.0000010  
epoch  20:  13.64 sec(s) train loss: 0.6245, acc 0.8696 valid loss: 0.8055, acc 0.7029 lr  0.0000010  
epoch  21:  13.74 sec(s) train loss: 0.6220, acc 0.8715 valid loss: 0.8044, acc 0.7093 lr  0.0000010  
epoch  22:  13.77 sec(s) train loss: 0.6236, acc 0.8746 valid loss: 0.8053, acc 0.7073 lr  0.0000010  

###  用更丰富的图片增强的格式 效果会变坏
teacher net 和 student net之间应该有同样的数据增强的格式， teacher net 是怎么训练的， student 就应该怎么训练  
另外自己做错了一点， 不能用teacher-net 预测一遍soft- label后然后固定下来，因为有数据增强，这样是不对的  
epoch  32:  24.77 sec(s) train loss: 0.7820, acc 0.6988 valid loss: 0.8631, acc 0.6580 lr  0.0001000  
epoch  33:  24.92 sec(s) train loss: 0.7717, acc 0.7063 valid loss: 0.8502, acc 0.6700 lr  0.0001000  
epoch  34:  24.85 sec(s) train loss: 0.7794, acc 0.7026 valid loss: 0.8454, acc 0.6711 lr  0.0001000  
epoch  35:  25.50 sec(s) train loss: 0.7788, acc 0.7027 valid loss: 0.8502, acc 0.6746 lr  0.0001000  
epoch  36:  24.84 sec(s) train loss: 0.7757, acc 0.7021 valid loss: 0.8480, acc 0.6682 lr  0.0001000  

## 总结
1. 可以牺牲一些精度，使得模型大幅度的减少size， 42M的resnet18 88%， 自己设计的小的模型1M 80%
2. 自己犯的错误
   在训练的时候，用teacher net 预测了soft label 然后保存下来了， 这样不好，因为训练的过程中，每次都是会进行数据增强的,   每次数据增强的图片都是不一样的，所以每次都要进行teacher-net的预测 
## TODO
1. 损失加上 L2 损失
2. 训练的时候获取soft-label
   
# 模型压缩
这里没什么好讲的
双精度 64 位  
单精度 32 位
半精度 16 位

精度降低到半精度比较折中
# 模型剪枝
作者的思路是开始模型某一层有8个channel， 第一次剪枝将8个channel选择出7个，并将这七个channel的参数赋值到新的model   
新的model和旧的model的模型是不一样的，需要自己手动的赋值  
这个不好训、用的少