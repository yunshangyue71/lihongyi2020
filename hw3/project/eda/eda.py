import os
root = "/media/q/data/lihongyi_2020/hw3/data/food-11/"
train_num = len(os.listdir(root + "training"))
val_num = len(os.listdir(root + "validation"))
test_num = len(os.listdir(root + "testing"))
print("train_num, val_num, test_num", train_num, val_num, test_num)

train_pathes = os.listdir(root + "training")
cls_num = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
for i in range(len(train_pathes)):
    path = train_pathes[i]
    cls = int(path.split("_")[0])
    cls_num[cls] += 1

print("cls_num:", cls_num)

print()