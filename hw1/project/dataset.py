import numpy as np
import pandas as pd
import os
import math
import csv
import matplotlib.pyplot as plt

class Path(object):
    def __init__(self):
        self.root = "/media/q/data/lihongyi2020/hw1/"
        self.data_root = self.root + "data/"
        self.model_root = self.root + "model/"

class Dataset(Path):
    def __init__(self):
        Path.__init__(self)
    def get_rawdata(self):
        data = pd.read_csv(self.data_root + 'train.csv', encoding='big5')
        no_use = data.iloc[:, :3]
        data = data.iloc[:, 3:]
        data[data == 'NR'] = 0  # 无降雨值设置为0
        # （4320， 24） 18*20*12, 24 ； 18项参数，一个月20天（10天用于测试） 12个月，24个小时
        raw_data = data.to_numpy()
        return raw_data

    def prepro_data(self, raw_data):
        month_data = {}
        for month in range(12):
            sample = np.empty([18, 480])
            for day in range(20):
                sample[:, day * 24: (day + 1) * 24] = raw_data[18 * (20 * month + day): 18 * (20 * month + day + 1), :]
                # 取raw_data的行,对sample的列进行拼接
            month_data[month] = sample
        x = np.empty([12 * 471, 18 * 9], dtype=float)
        y = np.empty([12 * 471, 1], dtype=float)
        for month in range(12):
            for day in range(20):
                for hour in range(24):
                    if day == 19 and hour > 14:  # 最后一天的最后10h是最后一组样本了,再往后无法作样本了
                        continue
                    x[month * 471 + day * 24 + hour, :] = month_data[month][:,
                                                          day * 24 + hour: day * 24 + hour + 9].reshape(1, -1)  # vector dim:18*9 (9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9)
                    y[month * 471 + day * 24 + hour, 0] = month_data[month][9, day * 24 + hour + 9]  # value第10个属性是PM2.5
        return x, y

    def prepro_data_pm(self, raw_data, m):
        month_data = {}
        for month in range(12):
            sample = np.empty([18, 480])
            for day in range(20):
                sample[:, day * 24: (day + 1) * 24] = raw_data[18 * (20 * month + day): 18 * (20 * month + day + 1), :]
                # 取raw_data的行,对sample的列进行拼接
            month_data[month] = sample
        x = np.empty([12 * 471,  9], dtype=float)
        y = np.empty([12 * 471, 1], dtype=float)
        for month in range(12):
            for day in range(20):
                for hour in range(24):
                    if day == 19 and hour > 14:  # 最后一天的最后10h是最后一组样本了,再往后无法作样本了
                        continue
                    x[month * 471 + day * 24 + hour, :] = month_data[month][m, day * 24 + hour: day * 24 + hour + 9].reshape(1, -1)  # vector dim:18*9 (9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9)
                    y[month * 471 + day * 24 + hour, 0] = month_data[month][9, day * 24 + hour + 9]  # value第10个属性是PM2.5
        return x, y


    def normalize(self, x):
        mean_x = np.mean(x, axis=0)  # 18 * 9
        std_x = np.std(x, axis=0)  # 18 * 9 标准差
        for i in range(len(x)):  # 12 * 471
            for j in range(len(x[0])):  # 18 * 9
                if std_x[j] != 0:
                    x[i][j] = (x[i][j] - mean_x[j]) / std_x[j]


        return x, mean_x, std_x


    def split_data(self,x, y):
        x_train_set = x[: math.floor(len(x) * 0.6), :]  # 取80%的样本, floor向下取整. 0~0.8
        y_train_set = y[: math.floor(len(y) * 0.6), :]
        x_validation = x[math.floor(len(x) * 0.6): math.floor(len(x) * 0.8), :]  # 0.8~1
        y_validation = y[math.floor(len(y) * 0.6): math.floor(len(y) * 0.8), :]
        x_test = x[math.floor(len(x) * 0.6): math.floor(len(x) * 0.8), :]
        # print(x_train_set)
        # print(y_train_set)
        # print(x_validation)
        # print(y_validation)
        # print(len(x_train_set))
        # print(len(y_train_set))
        # print(len(x_validation))
        # print(len(y_validation))
        return x_train_set,y_train_set,x_validation,y_validation




class DatasetTest(Path):
    def __init__(self,std_x, mean_x):
        Path.__init__(self)
        self.std_x = std_x
        self.mean_x = mean_x

    def get_rawdata(self):
        testdata = pd.read_csv(self.data_root + 'test.csv', header=None, encoding='big5')
        test_data = testdata.iloc[:, 2:]
        test_data[test_data == 'NR'] = 0
        test_data = test_data.to_numpy()

        return test_data

    def prepro_data(self, test_data):
        test_x = np.empty([240, 18 * 9], dtype=float)
        for i in range(240):
            test_x[i, :] = test_data[18 * i: 18 * (i + 1), :].reshape(1, -1)

        for i in range(len(test_x)):
            for j in range(len(test_x[0])):
                if self.std_x[j] != 0:
                    test_x[i][j] = (test_x[i][j] - self.mean_x[j]) / self.std_x[j]
        test_x = np.concatenate((np.ones([240, 1]), test_x), axis=1).astype(float)

        return test_x