import numpy as np
import pandas as pd
import os
import math
import csv
import matplotlib.pyplot as plt
from dataset import *

root = "../"
def plot():
    plt.figure(figsize=(10, 5))  # 设置画布的尺寸
    plt.title('Loss-Iteration Line Chart', fontsize=20)  # 标题，并设定字号大小
    plt.xlabel(u'x-iteration times', fontsize=14)  # 设置x轴，并设定字号大小
    plt.ylabel(u'y-loss', fontsize=14)  # 设置y轴，并设定字号大小
    plt.xlim((0, 1000))
    plt.ylim((5, 15))
    # color：颜色，linewidth：线宽，linestyle：线条类型，label：图例，marker：数据点的类型
    plt.plot(model_data['t'], model_data['loss1'], color="darkblue", linewidth=1, linestyle='--',
             label='learning_rate=0.5', marker='+')
    plt.plot(model_data['t'], model_data['loss2'], color="deeppink", linewidth=1, linestyle=':',
             label='learning_rate=2', marker='o')
    plt.plot(model_data['t'], model_data['loss3'], color="goldenrod", linewidth=1, linestyle='-',
             label='learning_rate=50', marker='*')
    plt.plot(model_data['t'], model_data['loss4'], color="green", linewidth=1, linestyle='-', label='learning_rate=100',
             marker='*')

    plt.legend(loc=1)  # 图例展示位置，数字代表第几象限
    plt.show()  # 显示图像

def train( x, y, xv, yv):
    learning_rate = 0.5 #e-1  # [0.5, 2, 50, 100]
    iter_time = 20000

    x = np.concatenate((np.ones([np.shape(x)[0], 1]), x), axis=1).astype(float)
    xv = np.concatenate((np.ones([np.shape(xv)[0], 1]), xv), axis=1).astype(float)

    dim = x.shape[1]  # 连续9个小时的，再加上一个bias
    w = np.zeros([dim, 1])

    adagrad = np.zeros([dim, 1])
    eps = 0.0000000001
    for t in range(iter_time):
        # np.dot矩阵点积, power:x^y
        train_loss = np.sqrt(np.sum(np.power(np.dot(x, w) - y, 2)) / y.shape[0])  # rmse标准差
        # 每进行100次迭代, 输出loss值

        # transpose矩阵转置
        gradient = 2 * np.dot(x.transpose(), np.dot(x, w) - y)  # dim*1
        adagrad += gradient ** 2
        w = w - learning_rate * gradient / np.sqrt(adagrad + eps)
        if t % 1000== 0 and t > 18000:
            val_loss = np.sqrt(np.sum(np.power(np.dot(xv, w) - yv, 2)) / yv.shape[0])
            print(t, "train_loss: %2.4f, val_loss: %2.4f  "% (train_loss, val_loss))
    cc = w[-1]
    w = w.reshape(-1)[:-1].reshape(18, 9)
    w = np.abs(w)
    for i in range(w.shape[0]):
        print(i, np.sum(w[i]), "\n", w[i])
    print(cc)

    # print("./*100")
    # print(w.reshape(-1)[:-1].reshape(9, 18).T, w[-1])
    # for i in range(w.shape[0]):
    #     print(w[i])
    # print(cc)
    np.save(root + 'model/weight.npy', w)
    # print("done")



def test(test_x):
    w = np.load(root + 'model/weight.npy')
    ans_y = np.dot(test_x, w)
    with open(root + 'data/submit.csv', mode='w', newline='') as submit_file:
        csv_writer = csv.writer(submit_file)
        header = ['id', 'value']
        print(header)
        csv_writer.writerow(header)
        for i in range(240):
            row = ['id_' + str(i), ans_y[i][0]]
            csv_writer.writerow(row)
            print(row)





if __name__ == '__main__':
    dataset = Dataset()

    #测试的时候，可以利用的信息，是前面9个小时所有污染物的指标，所以训练的时候模拟的是前9个小时，所有污染物的指标
    raw_data = dataset.get_rawdata()
    x,y = data = dataset.prepro_data(raw_data)
    x, mean, std = dataset.normalize(x)
    x_train_set,y_train_set,x_validation,y_validation = dataset.split_data(x, y)

    train(x_train_set, y_train_set, x_validation, y_validation)

    # 仅仅利用前9个小时的pm2.5
    # for m in range(18):
    #     print(m)
    #     x, y = data = dataset.prepro_data_pm(raw_data, m)
    #
    #     # #自己的方法
    #     # raw_data = dataset.get_rawdata_me()
    #     # x, y = data = dataset.prepare_data_me(raw_data)
    #     #
    #     # # raw_data = dataset.get_rawdata()
    #     # # x, y = data = dataset.prepro_data(raw_data)
    #
    #     x, mean, std = dataset.normalize(x)
    #     x_train_set,y_train_set,x_validation,y_validation = dataset.split_data(x, y)
    #
    #     train(x_train_set, y_train_set, x_validation, y_validation)

    # dataset = DatasetTest(mean, std)
    # raw_data_test = dataset.get_rawdata()
    # data = dataset.prepro_data(raw_data_test)
    #
    # model = Test()
    # model.test(data)


