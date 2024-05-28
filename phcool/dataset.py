import torch
from torch.utils.data import Dataset
from PIL import Image
import os


class MyData(Dataset):
    def __init__(self, root_dir, lable_dir):
        self.root_dir = root_dir
        self.lable_dir = lable_dir
        self.path = os.path.join(self.root_dir, self.lable_dir)
        self.img_path = os.listdir(self.path)

    def __getitem__(self, index):
        img_name = self.img_path[index]
        img_item_path = os.path.join(self.root_dir, self.lable_dir, img_name)
        img = Image.open(img_item_path)
        lable = self.lable_dir
        return img, lable

    def __len__(self):
        return len(self.img_path)


root_dir = "hymenoptera_data\\train"
ants_lable_dir = "ants"
bees_lable_dir = "bees"


ants_dataset = MyData(
    root_dir, ants_lable_dir)

bees_dataset = MyData(
    root_dir, bees_lable_dir
)
# img, lable = ants_dataset.__getitem__(0)
# img.show()
train_dataset = ants_dataset+bees_dataset
img, lable = train_dataset.__getitem__(200)
img.show()
