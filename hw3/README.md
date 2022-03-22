
# 1 官方baseline
过拟合非常严重  
[001/500] 11.87 sec(s) Train Acc: 0.351814 Loss: 0.014645 | Val Acc: 0.346356 loss: 0.003551 0.0001  
[002/500] 11.94 sec(s) Train Acc: 0.496959 Loss: 0.011545 | Val Acc: 0.460933 loss: 0.002895 0.0001  
[003/500] 11.74 sec(s) Train Acc: 0.570748 Loss: 0.009883 | Val Acc: 0.523907 loss: 0.002585 0.0001  
[004/500] 12.13 sec(s) Train Acc: 0.633691 Loss: 0.008365 | Val Acc: 0.452187 loss: 0.003058 0.0001  
[005/500] 11.97 sec(s) Train Acc: 0.685587 Loss: 0.007294 | Val Acc: 0.508163 loss: 0.003289 0.0001  
[006/500] 11.95 sec(s) Train Acc: 0.737381 Loss: 0.006134 | Val Acc: 0.568805 loss: 0.002616 0.0001  
[007/500] 12.01 sec(s) Train Acc: 0.798804 Loss: 0.004673 | Val Acc: 0.553644 loss: 0.002763 0.0001  
[008/500] 11.76 sec(s) Train Acc: 0.869653 Loss: 0.003320 | Val Acc: 0.501749 loss: 0.003856 0.0001  
[009/500] 11.82 sec(s) Train Acc: 0.911210 Loss: 0.002378 | Val Acc: 0.588047 loss: 0.002991 0.0001  
[010/500] 11.93 sec(s) Train Acc: 0.939185 Loss: 0.001817 | Val Acc: 0.569096 loss: 0.003217 0.0001  
[011/500] 12.21 sec(s) Train Acc: 0.985911 Loss: 0.000611 | Val Acc: 0.605539 loss: 0.003026 0.0001  
[012/500] 12.57 sec(s) Train Acc: 0.998074 Loss: 0.000227 | Val Acc: 0.573469 loss: 0.003620 0.0001  
[013/500] 12.06 sec(s) Train Acc: 0.968984 Loss: 0.000933 | Val Acc: 0.601749 loss: 0.003163 0.0001  
[014/500] 11.86 sec(s) Train Acc: 0.994324 Loss: 0.000292 | Val Acc: 0.615452 loss: 0.003281 0.0001  
[015/500] 12.25 sec(s) Train Acc: 1.000000 Loss: 0.000059 | Val Acc: 0.642566 loss: 0.003086 0.0001  
[016/500] 11.99 sec(s) Train Acc: 1.000000 Loss: 0.000030 | Val Acc: 0.591837 loss: 0.003836 0.0001  
# 2 数据增强的影响
1->2 提升了些7个点  
[008/500] 18.53 sec(s) Train Acc: 0.988851 Loss: 0.000332 | Val Acc: 0.669679 loss: 0.003793 1e-05  
[009/500] 18.33 sec(s) Train Acc: 0.991182 Loss: 0.000314 | Val Acc: 0.667347 loss: 0.003853 1e-05  
[010/500] 17.20 sec(s) Train Acc: 0.987634 Loss: 0.000368 | Val Acc: 0.666181 loss: 0.003821 1e-05  
# 3 normalize的影响
2->3 差别不大  
[130/500] 16.46 sec(s) Train Acc: 0.991790 Loss: 0.000227 | Val Acc: 0.672595 loss: 0.004205 1e-05  
[131/500] 16.20 sec(s) Train Acc: 0.995439 Loss: 0.000161 | Val Acc: 0.670845 loss: 0.004200 1e-05  
[132/500] 16.77 sec(s) Train Acc: 0.997365 Loss: 0.000123 | Val Acc: 0.675510 loss: 0.004201 1e-05  
[133/500] 16.29 sec(s) Train Acc: 0.997061 Loss: 0.000134 | Val Acc: 0.677259 loss: 0.004294 1e-05  
[134/500] 16.77 sec(s) Train Acc: 0.996047 Loss: 0.000128 | Val Acc: 0.678426 loss: 0.004164 1e-05  
[135/500] 16.49 sec(s) Train Acc: 0.995743 Loss: 0.000149 | Val Acc: 0.675219 loss: 0.004244 1e-05  
[136/500] 16.41 sec(s) Train Acc: 0.997770 Loss: 0.000121 | Val Acc: 0.678717 loss: 0.004216 1e-05  
# 4 正则项的影响
3->4  
## L1的影响
## L2的影响
影响不大  
[047/500] 20.42 sec(s) Train Acc: 0.991283 Loss: 0.000309 | Val Acc: 0.671137 loss: 0.003776 1e-05  
[048/500] 20.41 sec(s) Train Acc: 0.990878 Loss: 0.000372 | Val Acc: 0.672886 loss: 0.003744 1e-05  
[049/500] 16.45 sec(s) Train Acc: 0.989966 Loss: 0.000362 | Val Acc: 0.672303 loss: 0.003726 1e-05  
[050/500] 18.34 sec(s) Train Acc: 0.991080 Loss: 0.000311 | Val Acc: 0.672886 loss: 0.003819 1e-05  
[051/500] 18.15 sec(s) Train Acc: 0.992905 Loss: 0.000300 | Val Acc: 0.671137 loss: 0.003767 1e-05  
[052/500] 19.79 sec(s) Train Acc: 0.991891 Loss: 0.000296 | Val Acc: 0.668222 loss: 0.003672 1e-05  
[022/500] 16.41 sec(s) Train Acc: 0.996250 Loss: 0.000154 | Val Acc: 0.675802 loss: 0.004342 1e-05  
[023/500] 18.99 sec(s) Train Acc: 0.995844 Loss: 0.000154 | Val Acc: 0.680175 loss: 0.004411 1e-05  
[024/500] 17.16 sec(s) Train Acc: 0.997466 Loss: 0.000131 | Val Acc: 0.674344 loss: 0.004441 1e-05  
[025/500] 16.99 sec(s) Train Acc: 0.996452 Loss: 0.000138 | Val Acc: 0.676968 loss: 0.004493 1e-05  
# 5 丰富数据增强
4->5 提高了10个点，之前自己增强的不够  
[010/500] 15.36 sec(s) Train Acc: 0.925603 Loss: 0.001862 | Val Acc: 0.779009 loss: 0.003156 1e-06  
[011/500] 14.87 sec(s) Train Acc: 0.926414 Loss: 0.001891 | Val Acc: 0.777259 loss: 0.003206 1e-06  
[012/500] 14.77 sec(s) Train Acc: 0.924083 Loss: 0.001913 | Val Acc: 0.779300 loss: 0.003180 1e-06  
[013/500] 14.99 sec(s) Train Acc: 0.922562 Loss: 0.001928 | Val Acc: 0.781341 loss: 0.003203 1e-06  
[014/500] 14.88 sec(s) Train Acc: 0.926921 Loss: 0.001893 | Val Acc: 0.781341 loss: 0.003205 1e-06  
[015/500] 15.72 sec(s) Train Acc: 0.927833 Loss: 0.001815 | Val Acc: 0.779883 loss: 0.003185 1e-06  
[016/500] 15.07 sec(s) Train Acc: 0.923170 Loss: 0.001975 | Val Acc: 0.780466 loss: 0.003163 1e-06  
[017/500] 14.77 sec(s) Train Acc: 0.929961 Loss: 0.001931 | Val Acc: 0.783090 loss: 0.003205 1e-06  
[018/500] 15.04 sec(s) Train Acc: 0.923576 Loss: 0.001946 | Val Acc: 0.780466 loss: 0.003214 1e-06  
[019/500] 14.81 sec(s) Train Acc: 0.925704 Loss: 0.001924 | Val Acc: 0.783382 loss: 0.003146 1e-06  
[020/500] 14.95 sec(s) Train Acc: 0.924387 Loss: 0.001886 | Val Acc: 0.782216 loss: 0.003174 1e-06  
[021/500] 15.32 sec(s) Train Acc: 0.924184 Loss: 0.001935 | Val Acc: 0.779883 loss: 0.003202 1e-06  
# 6 分辨率128*128-256*256， 提升了4个点
5->6 分辨率提升了4个点    
[001/500] 35.02 sec(s) Train Acc: 0.883033 Loss: 0.002983 | Val Acc: 0.809329 loss: 0.005017 1e-07  
[002/500] 34.40 sec(s) Train Acc: 0.876242 Loss: 0.003121 | Val Acc: 0.809621 loss: 0.005041 1e-07  
[003/500] 34.35 sec(s) Train Acc: 0.870262 Loss: 0.003220 | Val Acc: 0.807289 loss: 0.005032 1e-07  
[004/500] 34.75 sec(s) Train Acc: 0.879181 Loss: 0.003154 | Val Acc: 0.807289 loss: 0.005061 1e-07  
[005/500] 34.43 sec(s) Train Acc: 0.883641 Loss: 0.002959 | Val Acc: 0.810496 loss: 0.005039 1e-07  
[006/500] 34.52 sec(s) Train Acc: 0.877154 Loss: 0.003086 | Val Acc: 0.809621 loss: 0.005049 1e-07  
[007/500] 34.38 sec(s) Train Acc: 0.877053 Loss: 0.003196 | Val Acc: 0.808455 loss: 0.005048 1e-07  
[008/500] 34.40 sec(s) Train Acc: 0.877863 Loss: 0.003068 | Val Acc: 0.808455 loss: 0.005036 1e-07  
[009/500] 34.62 sec(s) Train Acc: 0.879485 Loss: 0.003033 | Val Acc: 0.808455 loss: 0.005039 1e-07  
[010/500] 34.91 sec(s) Train Acc: 0.873100 Loss: 0.003208 | Val Acc: 0.811370 loss: 0.005048 1e-07  
# 7 focol loss 的影响
5->7 因为时间关系，所以不在大的分辨率上进行训练，没提升  
[080/500] 15.92 sec(s) Train Acc: 0.921751 Loss: 0.000792 | Val Acc: 0.778134 loss: 0.004735 1e-06  
[081/500] 16.14 sec(s) Train Acc: 0.914251 Loss: 0.000874 | Val Acc: 0.778717 loss: 0.004744 1e-06  
[082/500] 15.61 sec(s) Train Acc: 0.918609 Loss: 0.000868 | Val Acc: 0.780466 loss: 0.004730 1e-06  
[083/500] 15.58 sec(s) Train Acc: 0.916785 Loss: 0.000862 | Val Acc: 0.781050 loss: 0.004713 1e-06  
[084/500] 16.17 sec(s) Train Acc: 0.918812 Loss: 0.000842 | Val Acc: 0.782216 loss: 0.004718 1e-06  
[085/500] 15.74 sec(s) Train Acc: 0.922866 Loss: 0.000827 | Val Acc: 0.774344 loss: 0.004764 1e-06  
[086/500] 15.39 sec(s) Train Acc: 0.916785 Loss: 0.000869 | Val Acc: 0.776093 loss: 0.004776 1e-06  
[087/500] 15.60 sec(s) Train Acc: 0.920028 Loss: 0.000833 | Val Acc: 0.780175 loss: 0.004697 1e-06  
[088/500] 15.75 sec(s) Train Acc: 0.921549 Loss: 0.000821 | Val Acc: 0.776676 loss: 0.004754 1e-06  
[089/500] 15.11 sec(s) Train Acc: 0.920941 Loss: 0.000859 | Val Acc: 0.780175 loss: 0.004701 1e-06  
[090/500] 15.37 sec(s) Train Acc: 0.921346 Loss: 0.000809 | Val Acc: 0.778426 loss: 0.004743 1e-06  
[091/500] 15.52 sec(s) Train Acc: 0.915974 Loss: 0.000805 | Val Acc: 0.777843 loss: 0.004759 1e-06  
[010/500] 16.06 sec(s) Train Acc: 0.919218 Loss: 0.000318 | Val Acc: 0.779883 loss: 0.002336 1e-07  
[011/500] 16.15 sec(s) Train Acc: 0.916988 Loss: 0.000310 | Val Acc: 0.779300 loss: 0.002334 1e-07  
[012/500] 15.79 sec(s) Train Acc: 0.920637 Loss: 0.000305 | Val Acc: 0.777843 loss: 0.002337 1e-07  
[013/500] 15.36 sec(s) Train Acc: 0.917190 Loss: 0.000303 | Val Acc: 0.780175 loss: 0.002337 1e-07  
[014/500] 15.61 sec(s) Train Acc: 0.918204 Loss: 0.000306 | Val Acc: 0.779300 loss: 0.002328 1e-07  
[015/500] 15.30 sec(s) Train Acc: 0.918609 Loss: 0.000298 | Val Acc: 0.781341 loss: 0.002334 1e-07  
# 8 ResNet18
6->8  
resnet18, 5个block--> flatten -->512fc-->256fc-->128fc-->11  
学习率：[1e-4, 5e-5, 1e-5, 1e-6]  
优化器：Adam  
整体训练的epoch: 500  
在验证集上的精度:0.87  
在测试集上的精度：0.86
训练到最后还没有过拟合，因此可以不加dropout，表明可以继续使用更加复杂的网络  
[001/500] 170.82 sec(s) Train Acc: 0.897121 Loss: 0.021312 | Val Acc: 0.869096 loss: 0.068980 1e-06  
[002/500] 171.21 sec(s) Train Acc: 0.896006 Loss: 0.021516 | Val Acc: 0.868513 loss: 0.069546 1e-06  
[003/500] 171.22 sec(s) Train Acc: 0.897730 Loss: 0.021163 | Val Acc: 0.871429 loss: 0.068692 1e-06  
[004/500] 171.25 sec(s) Train Acc: 0.892966 Loss: 0.021292 | Val Acc: 0.867055 loss: 0.068964 1e-06  
[005/500] 170.90 sec(s) Train Acc: 0.898540 Loss: 0.020733 | Val Acc: 0.870845 loss: 0.068588 1e-06  
[006/500] 171.15 sec(s) Train Acc: 0.897324 Loss: 0.020895 | Val Acc: 0.869679 loss: 0.068436 1e-06  
[007/500] 170.92 sec(s) Train Acc: 0.898642 Loss: 0.020886 | Val Acc: 0.868222 loss: 0.069094 1e-06  
[008/500] 170.87 sec(s) Train Acc: 0.900669 Loss: 0.020745 | Val Acc: 0.868805 loss: 0.068614 1e-06  
[009/500] 170.98 sec(s) Train Acc: 0.897426 Loss: 0.020868 | Val Acc: 0.869096 loss: 0.068787 1e-06  
[010/500] 171.24 sec(s) Train Acc: 0.897730 Loss: 0.020684 | Val Acc: 0.865015 loss: 0.069337 1e-06  
[011/500] 171.17 sec(s) Train Acc: 0.898439 Loss: 0.020600 | Val Acc: 0.866764 loss: 0.068974 1e-06  
# FC参数去掉后的效果
## 去掉之前
Total params: 11,597,131  
Trainable params: 11,597,131  
Non-trainable params: 0  
Input size (MB): 0.75  
Forward/backward pass size (MB): 872.01  
Params size (MB): 44.24  
Estimated Total Size (MB): 917.00  
## 去掉之后
Total params: 11,174,475  
Trainable params: 11,174,475  
Non-trainable params: 0  
Input size (MB): 0.75  
Forward/backward pass size (MB): 872.00  
Params size (MB): 42.63  
Estimated Total Size (MB): 915.38  

# TODO
1. 增加模型的复杂程度进而过拟合， resenet50， VGG16等
2. dropout的影响减少过拟合(CNN 后添加+DNN后添加, DNN后添加)
始终没有过拟合，因此没有继续优化
3. 可以用resnet34 50 直至过拟合以后再添加dropout，因为需要训练的时间比较久，就不做了
4.  验证集和加入训练微调

# bad case ：resnet 18的效果
resnet18, 5个block 前三个不加dropout，后面都加上dropout(0.5)  
-512, 256, 128, 11
学习率：[1e-4, 5e-5, 1e-5, 1e-6]  
优化器：Adam  
整体训练的epoch: 320  
在验证集上的精度:0.8  这个是增强的效果不够好
[001/500] 141.79 sec(s) Train Acc: 0.804379 Loss: 0.004304 | Val Acc: 0.795918 loss: 0.010099 1e-07    
[002/500] 141.92 sec(s) Train Acc: 0.807419 Loss: 0.004316 | Val Acc: 0.791837 loss: 0.010137 1e-07  
[003/500] 141.78 sec(s) Train Acc: 0.806811 Loss: 0.004323 | Val Acc: 0.787755 loss: 0.010112 1e-07  
[004/500] 141.90 sec(s) Train Acc: 0.814008 Loss: 0.004178 | Val Acc: 0.784548 loss: 0.010076 1e-07  

