import os
import shutil

root = "/root/autodl-tmp/"  # "/media/q/data/lihongyi_2020/hw3/"
workspace_dir = root + 'data/food-11/'

# val_data = ImgDataset(workspace_dir + "validation/", mode="val")
img_dir = workspace_dir + "training/"
a = sorted(os.listdir(img_dir))
shutil.rmtree(img_dir + '.ipynb_checkpoints')
# os.remove(img_dir + '.ipynb_checkpoints')
print("")