import torch
import torch.nn as nn
import torch.nn.functional as F


class ResnetBasic(nn.Module):
    """Basic Block for resnet 18 and resnet 34
        x--CnnBNRelu--CnnBN   + ReLU ---out
          |------------------|
    """
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        #residual function
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * ResnetBasic.expansion, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels * ResnetBasic.expansion)
        )

        #shortcut
        self.shortcut = nn.Sequential()

        #the shortcut output dimension is not the same with residual function
        #use 1*1 convolution to match the dimension
        if stride != 1 or in_channels != ResnetBasic.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * ResnetBasic.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * ResnetBasic.expansion)
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))

class ResnetBasicSlim(nn.Module):
    '''
    减少网络的计算, 常用于layer大于50的网络
    x--1*1CnnBNRelu--CnnBN   --1*1CnnBN  + ReLU ---out
      |--------------------------------|
    '''
    expansion = 4
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, stride=stride, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * ResnetBasicSlim.expansion, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels * ResnetBasicSlim.expansion),
        )

        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != out_channels * ResnetBasicSlim.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * ResnetBasicSlim.expansion, stride=stride, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels * ResnetBasicSlim.expansion)
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))

class ResNet(nn.Module):
    def __init__(self, block, num_block, num_classes=11):
        super().__init__()

        self.in_channels = 64

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))
        #we use a different inputsize than the original paper
        #so conv2_x's stride is 1
        self.conv2_x = self._make_layer(block, 64, num_block[0], 1)
        self.conv3_x = self._make_layer(block, 128, num_block[1], 2)
        self.conv4_x = self._make_layer(block, 256, num_block[2], 2)
        self.conv5_x = self._make_layer(block, 512, num_block[3], 2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(512 * block.expansion, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, num_classes)
        self.init_weight()

    def _make_layer(self, block, out_channels, num_blocks, stride):
        """make resnet layers(by layer i didnt mean this 'layer' was the
        same as a neuron netowork layer, ex. conv layer), one layer may
        contain more than one residual block
        Args:
            block: block type, basic block or bottle neck block
            out_channels: output depth channel number of this layer
            num_blocks: how many blocks per layer
            stride: the stride of the first block of this layer
        Return:
            return a resnet layer
        """

        # we have num_block blocks per layer, the first block
        # could be 1 or 2, other blocks would always be 1
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        output = self.conv1(x)
        output = self.conv2_x(output)
        # output = nn.Dropout(0.5)(output)
        output = self.conv3_x(output)
        # output = nn.Dropout(0.5)(output)
        output = self.conv4_x(output)
        # output = nn.Dropout(0.5)(output)
        output = self.conv5_x(output)
        # output = nn.Dropout(0.5)(output)
        output = self.avg_pool(output)
        # output = nn.Dropout(0.5)(output)
        output = output.view(output.size(0), -1)
        output = self.fc1(output)
        # output = nn.Dropout(0.5)(output)
        output = self.fc2(output)
        # output = nn.Dropout(0.5)(output)
        output = self.fc3(output)
        # output = nn.Dropout(0.5)(output)
        output = self.fc4(output)
        return output

    def init_weight(self):
        for m in self.children():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.01)
                nn.init.constant_(m.bias, 0)
            # 也可以判断是否为conv2d，使用相应的初始化方式
            elif isinstance(m, nn.Conv2d):
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.normal_(m.weight, std=0.001)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                # def init_weights(m):

def resnet18():
    return ResNet(ResnetBasic, [2, 2, 2, 2])
def resnet34():
    return ResNet(ResnetBasic, [3,4,6,3])
def resnet50():
    return ResNet(ResnetBasicSlim, [3,4,6,3])
def resnet101():
    return ResNet(ResnetBasicSlim, [3,4,23,3])
def resnet152():
    return ResNet(ResnetBasicSlim, [3,8,36,3])