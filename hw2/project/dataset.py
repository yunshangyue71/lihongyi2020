import numpy as np
import pandas as pd
from pandas import  DataFrame as df

import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator

root = "/media/q/data/lihongyi2020/hw2"
np.random.seed(0)
X_train_fpath = root + '/data/adult.data'
Y_train_fpath = root + '/data/Y_train'
X_test_fpath = root + '/data/X_test'
output_fpath = root + '/output_{}.csv'

# read files
feature = {0: "age", 1: "workclass", 2: "final-weight", 3: "education", 4: "education-num", 5: "marital-status",
           6: "occupation", 7: "relationship", 8: "race", 9: "sex", 10: "capital-gain",
           11: "capital-loss", 12: "hours-per-week", 13: "country", 14: "income-level"}
workclass = {
    0:" Private", 1:" Self-emp-not-inc",  2:" Self-emp-inc",  3:" Federal-gov",
    4:" Local-gov",    5:" State-gov", 6:" Without-pay", 7:" Never-worked",
}
education = {
    0:" Preschool",#"(幼儿园)
    1:" 1st-4th",#"(小学1-4年级);
    2:" 5th-6th",#"(小学5-6年级);
    3:" 7th-8th",#(初中1-2年级)
    4:" 9th",#(初三),"
    5:" 10th",#"(高一);
    6:" 11th",#"(高二);
    7:" 12th",#"(高三);
    8:" HS-grad",#"(高中毕业);

    9:" Some-college",#"(大学未毕业);
    10:" Assoc-voc",#"(准职业学位);
    11:" Prof-school",#"(职业学校);
    12:" Assoc-acdm",#"(大学专科);
    13:" Assoc-voc",#"(准职业学位);

    14:" Bachelors",#(学士);
    15:" Masters",#"(硕士);
    16:" Doctorate",#"(博士);


}
marital_status = {
    0:" Married-civ-spouse",#(已婚平民配偶);
    1:" Divorced",#"(离婚);
    2:" Never-married",#(未婚);
    3:" Separated",#(分居);
    4:" Widowed",#(丧偶);
    5:" Married-spouse-absent",#(已婚配偶异地);
    6:" arried-AF-spouse",#(已婚军属)
}
occupation = {
0:" Tech-support",# "(技术支持);
1:" Craft-repair",#(手工艺维修);
2:" Other-service",#(其他职业);
3:" Sales",#(销售);
4:" Exec-managerial",#(执行主管);
5:" Prof-specialty",#(专业技术);
6:" Handlers-cleaners",#(劳工保洁);
7:" Machine-op-inspct",#(机械操作);
8:" Adm-clerical",#(管理文书);
9:" Farming-fishing",#(农业捕捞);
10:" Transport-moving",#(运输);
11: " Priv-house-serv",#(家政服务);
12:" Protective-serv",#(保安);
13:" Armed-Forces",#(军人)
}

relationship ={
0:" Wife",#(妻子);
1:" Own-child",#(孩子);
2:" Husband",#(丈夫);
3:" Not-in-family",#(离家);
4:" Other-relative",#(其他关系);
5:" Unmarried",#(未婚)
}
race = {
0:" White",#(白人);
1:" Asian-Pac-Islander",#(亚裔、太平洋岛裔);
2:" Amer-Indian-Eskimo",#(美洲印第安裔、爱斯基摩裔);
3:" Black",#(非裔);
4:" Other",#(其他)
}
sex = {
0: " Female", #(女);
1: " Male"#(男)
}
country = {

0: " United-States",  #美国);
1 : " Cambodia",  #柬埔寨);
2: " England",  #英国);
3 : " Puerto-Rico",  #波多黎各);
4 : " Canada",  #加拿大);
5 : " Germany",  #德国);
6: " Outlying-US(Guam-USVI-etc)",  #美国海外属地);
7 : " India",  #印度);
8 : " Japan",  #日本);
9 : " Greece",  #希腊);
10 : " South",  #南美);
11: " China",  #中国);
12 : " Cuba",  #古巴);
13 : " Iran",  #伊朗);
14 : " Honduras",  #洪都拉斯);
15 : " Philippines",  #菲律宾);
16 : " Italy",  #意大利);
17 : " Poland",  #波兰);
18 : " Jamaica",  #牙买加)，
19:" Vietnam",  #越南);
20: " Mexico",  #墨西哥);
21 : " Portugal",  #葡萄牙);
22 : " Ireland",  #爱尔兰);
23 : " France",  #法国);
24 : " Dominican-Republic",  #多米尼加共和国);
25 : " Laos",  #老挝);
26 : " Ecuador",  #厄瓜多尔);
27 : " Taiwan",  #台湾);
28 : " Haiti",  #海地);
29 : " Columbia",  #哥伦比亚);
30 : " Hungary",  #匈牙利);
31 : " Guatemala",  #危地马拉);
32 : " Nicaragua",  #尼加拉瓜);
33 : " Scotland",  #苏格兰);
34 : " Thailand",  #泰国);
35 : " Yugoslavia",  #南斯拉夫);
36 : " El-Salvador",  #萨尔瓦多);
37 : " Trinadad&Tobago",  #特立尼达和多巴哥);
38 : " Peru",  #秘鲁);
39 : " Hong",  #香港);
40 : " Holand-Netherlands",  #荷兰)
}
income_level = {
    0:" <=50K",
   1:" >50K"
}
income_level2 = {
    0:" <=50K.",
   1:" >50K."
}
class DataProcess():
    def __init__(self):
        self.train = pd.read_csv(root + "/data/train.csv")
        self.val = pd.read_csv(root + "/data/val.csv")
        self.test = pd.read_csv(root + "/data/test.csv")
        self.mean = {}
        self.std = {}

    def split_save_trainval(self, train_per = 0.8):
        x_train = []
        with open(X_train_fpath) as f:
            lines = f.readlines()
            for line in lines:
                data = line.strip().split(",")
                x_train.append(data)
        X_train = df(x_train)
        X_train.rename(columns=feature, inplace=True)
        df1 = X_train.sample(frac=1)
        row_num, col_num = df1.shape
        trainX = df1[0:int(row_num * train_per)]
        valX = df1[int(row_num * train_per):]
        trainX.to_csv(root + "/data/train.csv")
        valX.to_csv(root + "/data/val.csv")

        test = []
        with open(root + "/data/adult.test") as f:
            lines = f.readlines()
            for line in lines:
                data = line.strip().split(",")
                test.append(data)
        test = df(test)
        test.rename(columns=feature, inplace=True)
        test.to_csv(root + "/data/test.csv")

    def __to_num(self, dataset):
        for key, value in workclass.items():
            dataset.replace(value, int(key), inplace=True)
        for key, value in education.items():
            dataset.replace(value, int(key), inplace=True)
        for key, value in marital_status.items():
            dataset.replace(value, int(key), inplace=True)
        for key, value in occupation.items():
            dataset.replace(value, int(key), inplace=True)
        for key, value in relationship.items():
            dataset.replace(value, int(key), inplace=True)
        for key, value in race.items():
            dataset.replace(value, int(key), inplace=True)
        for key, value in country.items():
            dataset.replace(value, int(key), inplace=True)
        for key, value in sex.items():
            dataset.replace(value, int(key), inplace=True)

        for key, value in income_level.items():
            dataset.replace(value, int(key), inplace=True)
        for key, value in income_level2.items():
            dataset.replace(value, int(key), inplace=True)
        # 非数字的变为nan
        for col in dataset.columns.values:
            dataset[col] = pd.to_numeric(dataset[col], "coerce")
        dataset = dataset.astype("float")
        return dataset
        # print()

    def onehot(self, dataset1, dataset2 ,dataset3):
        feature_names = ["marital-status", "sex","workclass", "education","occupation",
                         "relationship", "race", "country"
        ]
        dataset1_num = dataset1.shape[0]
        dataset2_num = dataset2.shape[0]
        dataset3_num = dataset3.shape[0]
        dataset = pd.concat([dataset1, dataset2, dataset3])

        for feature_name in feature_names:
            temp = pd.get_dummies(dataset[feature_name], prefix=feature_name)
            dataset.drop(labels=feature_name, axis= 1, inplace=True)
            dataset = pd.concat([temp, dataset], axis=1) # label in the last col
        return dataset[:dataset1_num], dataset[dataset1_num: dataset2_num+dataset1_num], dataset[dataset2_num+dataset1_num:]
    def drop_not_num(self, dataset):
        dataset.dropna(axis=0,how='any', inplace=True) # 删除 有 nan的行

    def get_normalization(self):
        for col in self.train.columns.values:
            temp = self.train[col]
            mean = temp.mean()
            std = temp.std()
            self.mean[col] = mean
            self.std[col] = max(std, 1e-8)
    def do_normalization(self, dataset):
        for col in dataset.columns.values:
            if col in [ "income-level"]:
                continue
            if col.split("_")[0] in ["marital-status", "sex", "workclass", "education", "occupation", "relationship", "race", "country"]:
                continue
            dataset[col] = (dataset[col] - self.mean[col]) / self.std[col]
    def run(self):
        self.train = self.__to_num(self.train)
        self.val = self.__to_num(self.val)
        self.test = self.__to_num(self.test)

        self.drop_not_num(self.train)
        self.drop_not_num(self.val)
        self.drop_not_num(self.test)

        self.train, self.val, self.test = self.onehot(self.train, self.val, self.test)


        self.get_normalization()
        self.do_normalization(self.train)
        self.do_normalization(self.val)
        self.do_normalization(self.test)
        return np.array(self.train), np.array(self.val), np.array(self.test)

    def age(self):
        self.X_train["age"] = pd.to_numeric(self.X_train["age"])
        age = self.X_train["age"]
        age_n = np.array(age)
        fig = plt.figure(figsize = (16, 10))

        ax1 = fig.add_subplot(2,2, 1)
        ax1.hist(age_n, bins = 20)
        ax1.set_title("orign")


        # 正则化
        mean = np.nanmean(age_n)
        std  = np.nanstd(age_n)

        nan_index = np.isnan(age_n)
        age_n[nan_index] = mean

        age2 = (age_n - mean)/std
        ax2 = fig.add_subplot(222)
        ax2.hist(age2, bins=20)
        ax2.set_title("orign-normalize")
        plt.show()

    def final_weight(self):
        self.X_train["final-weight"] = pd.to_numeric(self.X_train["final-weight"])
        fw = self.X_train["final-weight"]
        fw_n = np.array(fw)
        fig = plt.figure(figsize=(16, 10))

        ax1 = fig.add_subplot(2, 2, 1)
        ax1.hist(fw_n, bins=20)
        ax1.set_title("orign")

        mean = np.nanmean(fw_n)
        std = np.nanstd(fw_n)

        nan_index = np.isnan(fw_n)
        fw_n[nan_index] = mean

        fw2 = (fw_n - mean) / std
        ax2 = fig.add_subplot(222)
        ax2.hist(fw2, bins=20)
        ax2.set_title("orign-normalize")

        plt.show()
    def education_num(self):
        self.X_train["education-num"] = pd.to_numeric(self.X_train["education-num"])
        fw = self.X_train["education-num"]
        fw_n = np.array(fw)
        fig = plt.figure(figsize=(16, 10))

        ax1 = fig.add_subplot(2, 2, 1)
        ax1.hist(fw_n, bins=20)
        ax1.set_title("orign")

        mean = np.nanmean(fw_n)
        std = np.nanstd(fw_n)

        nan_index = np.isnan(fw_n)
        fw_n[nan_index] = mean

        fw2 = (fw_n - mean) / std
        ax2 = fig.add_subplot(222)
        ax2.hist(fw2, bins=20)
        ax2.set_title("orign-normalize")

        plt.show()
    def capital_gain(self):
        self.X_train["capital-gain"] = pd.to_numeric(self.X_train["capital-gain"])
        fw = self.X_train["capital-gain"]
        fw_n = np.array(fw)
        fig = plt.figure(figsize=(16, 10))

        ax1 = fig.add_subplot(2, 2, 1)
        ax1.hist(fw_n, bins=20)
        ax1.set_title("orign")

        mean = np.nanmean(fw_n)
        std = np.nanstd(fw_n)

        nan_index = np.isnan(fw_n)
        fw_n[nan_index] = mean

        fw2 = (fw_n - mean) / std
        ax2 = fig.add_subplot(222)
        ax2.hist(fw2, bins=20)
        ax2.set_title("orign-normalize")

        plt.show()
    def captital_loss(self):
        self.X_train["capital-loss"] = pd.to_numeric(self.X_train["capital-loss"])
        fw = self.X_train["capital-loss"]
        fw_n = np.array(fw)
        fig = plt.figure(figsize=(16, 10))

        ax1 = fig.add_subplot(2, 2, 1)
        ax1.hist(fw_n, bins=20)
        ax1.set_title("orign")

        mean = np.nanmean(fw_n)
        std = np.nanstd(fw_n)

        nan_index = np.isnan(fw_n)
        fw_n[nan_index] = mean

        fw2 = (fw_n - mean) / std
        ax2 = fig.add_subplot(222)
        ax2.hist(fw2, bins=20)
        ax2.set_title("orign-normalize")

        plt.show()
    def hours_per_week(self):
        self.X_train["hours-per-week"] = pd.to_numeric(self.X_train["hours-per-week"])
        fw = self.X_train["hours-per-week"]
        fw_n = np.array(fw)
        fig = plt.figure(figsize=(16, 10))

        ax1 = fig.add_subplot(2, 2, 1)
        ax1.hist(fw_n, bins=20)
        ax1.set_title("orign")

        mean = np.nanmean(fw_n)
        std = np.nanstd(fw_n)

        nan_index = np.isnan(fw_n)
        fw_n[nan_index] = mean

        fw2 = (fw_n - mean) / std
        ax2 = fig.add_subplot(222)
        ax2.hist(fw2, bins=20)
        ax2.set_title("orign-normalize")

        plt.show()

    def workclass(self):
        for key, value in workclass.items():
            self.X_train.replace(value, key, inplace=True)

        age = self.X_train["workclass"]

        age_n = np.array(age)
        fig = plt.figure(figsize=(16, 10))

        ax1 = fig.add_subplot(2, 2, 1)
        ax1.hist(age_n, bins=20)
        ax1.set_title("orign")

        # # 正则化
        # mean = np.nanmean(age_n)
        # std = np.nanstd(age_n)
        #
        # nan_index = np.isnan(age_n)
        # age_n[nan_index] = mean
        #
        # age2 = (age_n - mean) / std
        # ax2 = fig.add_subplot(222)
        # ax2.hist(age2, bins=20)
        # ax2.set_title("orign-normalize")
        plt.show()
if __name__ == '__main__':

    dp = DataProcess()
    dp.split_save_trainval()
    dp.run()
    # # dp.age()
    # # dp.final_weight()
    # # dp.education_num()
    # # dp.capital_gain()
    # # dp.captital_loss()
    # # dp.hours_per_week()
    # dp.workclass()

    print()