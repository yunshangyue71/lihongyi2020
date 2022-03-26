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
Total params: 37,999,627  
Trainable params: 37,999,627  
Non-trainable params: 0  
Input size (MB): 0.75  
Forward/backward pass size (MB): 198.27  
Params size (MB): 144.96  
Estimated Total Size (MB): 343.98  
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
# 第二次训练
6-8
[001/500] 194.99 sec(s) Train Acc: 0.831036 Loss: 0.008322 | Val Acc: 0.802624 loss: 0.015657 0.0001  
[002/500] 195.79 sec(s) Train Acc: 0.838232 Loss: 0.007835 | Val Acc: 0.791254 loss: 0.017806 0.0001  
[003/500] 196.32 sec(s) Train Acc: 0.837827 Loss: 0.007847 | Val Acc: 0.773178 loss: 0.017710 0.0001  
[004/500] 195.97 sec(s) Train Acc: 0.832556 Loss: 0.008185 | Val Acc: 0.730029 loss: 0.019464 0.0001  
[005/500] 195.73 sec(s) Train Acc: 0.839449 Loss: 0.007800 | Val Acc: 0.774927 loss: 0.020358 0.0001  
[006/500] 195.21 sec(s) Train Acc: 0.839651 Loss: 0.007803 | Val Acc: 0.653936 loss: 0.029801 0.0001  
[007/500] 195.39 sec(s) Train Acc: 0.845530 Loss: 0.007758 | Val Acc: 0.805248 loss: 0.015648 0.0001  
[008/500] 195.56 sec(s) Train Acc: 0.840259 Loss: 0.007874 | Val Acc: 0.763265 loss: 0.020077 0.0001  
[009/500] 194.97 sec(s) Train Acc: 0.840259 Loss: 0.007806 | Val Acc: 0.800292 loss: 0.015857 0.0001  
[010/500] 195.21 sec(s) Train Acc: 0.839651 Loss: 0.007814 | Val Acc: 0.740816 loss: 0.019740 0.0001  
[011/500] 195.17 sec(s) Train Acc: 0.847152 Loss: 0.007532 | Val Acc: 0.793586 loss: 0.016511 0.0001  
[012/500] 195.18 sec(s) Train Acc: 0.843402 Loss: 0.007588 | Val Acc: 0.755977 loss: 0.018627 0.0001  
[013/500] 195.16 sec(s) Train Acc: 0.842489 Loss: 0.007558 | Val Acc: 0.815743 loss: 0.015609 0.0001
[014/500] 194.99 sec(s) Train Acc: 0.842287 Loss: 0.007875 | Val Acc: 0.814577 loss: 0.016771 0.0001  
[015/500] 195.75 sec(s) Train Acc: 0.840665 Loss: 0.007843 | Val Acc: 0.806122 loss: 0.015847 0.0001    
[016/500] 195.76 sec(s) Train Acc: 0.846746 Loss: 0.007371 | Val Acc: 0.775802 loss: 0.018247 0.0001  
[017/500] 195.54 sec(s) Train Acc: 0.850902 Loss: 0.007304 | Val Acc: 0.734985 loss: 0.021312 0.0001  
[018/500] 195.31 sec(s) Train Acc: 0.848875 Loss: 0.007496 | Val Acc: 0.779009 loss: 0.018160 0.0001  
[019/500] 195.24 sec(s) Train Acc: 0.839753 Loss: 0.007831 | Val Acc: 0.797376 loss: 0.017128 0.0001  
[020/500] 195.39 sec(s) Train Acc: 0.842388 Loss: 0.007631 | Val Acc: 0.797376 loss: 0.016886 0.0001  

[001/500] 194.81 sec(s) Train Acc: 0.875633 Loss: 0.006332 | Val Acc: 0.838776 loss: 0.013647 1e-05  
[002/500] 195.98 sec(s) Train Acc: 0.884553 Loss: 0.005817 | Val Acc: 0.853061 loss: 0.012949 1e-05  
[003/500] 197.32 sec(s) Train Acc: 0.884148 Loss: 0.005735 | Val Acc: 0.848688 loss: 0.013182 1e-05  
[004/500] 196.72 sec(s) Train Acc: 0.884958 Loss: 0.005690 | Val Acc: 0.854227 loss: 0.013197 1e-05  
[005/500] 195.49 sec(s) Train Acc: 0.890837 Loss: 0.005537 | Val Acc: 0.861516 loss: 0.012705 1e-05  
[006/500] 195.07 sec(s) Train Acc: 0.894283 Loss: 0.005350 | Val Acc: 0.858017 loss: 0.012697 1e-05  
[007/500] 195.09 sec(s) Train Acc: 0.896006 Loss: 0.005259 | Val Acc: 0.852770 loss: 0.013002 1e-05  
[008/500] 195.24 sec(s) Train Acc: 0.894892 Loss: 0.005376 | Val Acc: 0.854227 loss: 0.012758 1e-05  
[009/500] 195.01 sec(s) Train Acc: 0.895398 Loss: 0.005416 | Val Acc: 0.857434 loss: 0.012802 1e-05  
[010/500] 195.37 sec(s) Train Acc: 0.898338 Loss: 0.005217 | Val Acc: 0.858601 loss: 0.012801 1e-05  
[011/500] 195.41 sec(s) Train Acc: 0.892966 Loss: 0.005361 | Val Acc: 0.860058 loss: 0.012833 1e-05  
[012/500] 195.38 sec(s) Train Acc: 0.894892 Loss: 0.005272 | Val Acc: 0.854810 loss: 0.013183 1e-05  
[013/500] 195.52 sec(s) Train Acc: 0.892864 Loss: 0.005312 | Val Acc: 0.857143 loss: 0.013129 1e-05  
[014/500] 195.61 sec(s) Train Acc: 0.904622 Loss: 0.004929 | Val Acc: 0.859767 loss: 0.013003 1e-05  
[015/500] 195.22 sec(s) Train Acc: 0.898135 Loss: 0.005097 | Val Acc: 0.857143 loss: 0.013145 1e-05  
[016/500] 195.83 sec(s) Train Acc: 0.900669 Loss: 0.005102 | Val Acc: 0.855685 loss: 0.013334 1e-05  
[017/500] 195.09 sec(s) Train Acc: 0.891648 Loss: 0.005326 | Val Acc: 0.851020 loss: 0.013371 1e-05  
[018/500] 195.17 sec(s) Train Acc: 0.897527 Loss: 0.005154 | Val Acc: 0.860641 loss: 0.013106 1e-05  
[019/500] 195.30 sec(s) Train Acc: 0.896311 Loss: 0.005231 | Val Acc: 0.855394 loss: 0.013135 1e-05  
[020/500] 195.41 sec(s) Train Acc: 0.896108 Loss: 0.005251 | Val Acc: 0.850437 loss: 0.013439 1e-05  
[021/500] 195.17 sec(s) Train Acc: 0.896311 Loss: 0.005259 | Val Acc: 0.854227 loss: 0.013465 1e-05  
[022/500] 195.24 sec(s) Train Acc: 0.896716 Loss: 0.005146 | Val Acc: 0.857726 loss: 0.013224 1e-05  
[023/500] 195.01 sec(s) Train Acc: 0.897426 Loss: 0.005010 | Val Acc: 0.858601 loss: 0.013354 1e-05  

[001/500] 194.99 sec(s) Train Acc: 0.900061 Loss: 0.004851 | Val Acc: 0.855685 loss: 0.013255 1e-06
[002/500] 195.43 sec(s) Train Acc: 0.901378 Loss: 0.005005 | Val Acc: 0.856560 loss: 0.013266 1e-06
[003/500] 195.47 sec(s) Train Acc: 0.899757 Loss: 0.005042 | Val Acc: 0.851020 loss: 0.013261 1e-06
[004/500] 195.57 sec(s) Train Acc: 0.905331 Loss: 0.004816 | Val Acc: 0.856560 loss: 0.013150 1e-06
[005/500] 195.58 sec(s) Train Acc: 0.901987 Loss: 0.004899 | Val Acc: 0.856560 loss: 0.013174 1e-06
[006/500] 195.65 sec(s) Train Acc: 0.900872 Loss: 0.004996 | Val Acc: 0.857726 loss: 0.013110 1e-06
[007/500] 195.31 sec(s) Train Acc: 0.901074 Loss: 0.005005 | Val Acc: 0.853936 loss: 0.013429 1e-06
[008/500] 195.69 sec(s) Train Acc: 0.906142 Loss: 0.004820 | Val Acc: 0.855685 loss: 0.013212 1e-06
[009/500] 195.42 sec(s) Train Acc: 0.905129 Loss: 0.004838 | Val Acc: 0.855394 loss: 0.013086 1e-06
[010/500] 195.83 sec(s) Train Acc: 0.904014 Loss: 0.004848 | Val Acc: 0.860641 loss: 0.013168 1e-06
[011/500] 195.41 sec(s) Train Acc: 0.905230 Loss: 0.004842 | Val Acc: 0.858309 loss: 0.013064 1e-06
[012/500] 195.58 sec(s) Train Acc: 0.907055 Loss: 0.004711 | Val Acc: 0.858892 loss: 0.013130 1e-06
[013/500] 195.45 sec(s) Train Acc: 0.900872 Loss: 0.005037 | Val Acc: 0.858892 loss: 0.013244 1e-06
[014/500] 196.03 sec(s) Train Acc: 0.903000 Loss: 0.004949 | Val Acc: 0.859767 loss: 0.013048 1e-06
[015/500] 195.46 sec(s) Train Acc: 0.898439 Loss: 0.004993 | Val Acc: 0.862099 loss: 0.013040 1e-06
[016/500] 195.44 sec(s) Train Acc: 0.900162 Loss: 0.004906 | Val Acc: 0.861516 loss: 0.013128 1e-06
[017/500] 195.82 sec(s) Train Acc: 0.906649 Loss: 0.004872 | Val Acc: 0.858017 loss: 0.013087 1e-06
[018/500] 195.50 sec(s) Train Acc: 0.905838 Loss: 0.004730 | Val Acc: 0.860058 loss: 0.013144 1e-06
[019/500] 195.71 sec(s) Train Acc: 0.901885 Loss: 0.004945 | Val Acc: 0.862099 loss: 0.012931 1e-06
[020/500] 195.62 sec(s) Train Acc: 0.904825 Loss: 0.004814 | Val Acc: 0.859184 loss: 0.013048 1e-06

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
4.  验证集和加入训练微调,中间没有使用save_best 感觉会掉一些点

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

