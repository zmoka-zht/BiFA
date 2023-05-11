"""
CD Dataset
"""
import os
from PIL import Image
import numpy as np
from torch.utils import data
import data.util as Util
from torch.utils.data import Dataset

"""
CD Dataset 
├─image
├─image_post
├─label
└─list
"""

IMG_FOLDER_NAME = 'A'
IMG_POST_FOLDER_NAME = 'B'
LABEL_FOLDER_NAME = 'label'
LIST_FOLDER_NAME = 'list'

label_suffix = ".png"

#list内存放image_name 构建读取图片名字函数
def load_img_name_list(dataset_path):
    img_name_list = np.loadtxt(dataset_path, dtype=np.str)
    if img_name_list.ndim == 2:
        return img_name_list[:, 0]
    return img_name_list


#获取各个文件夹的路径
def get_img_path(root_dir, img_name):
    return os.path.join(root_dir, IMG_FOLDER_NAME, img_name)

def get_img_post_path(root_dir, img_name):
    return os.path.join(root_dir, IMG_POST_FOLDER_NAME, img_name)

def get_label_path(root_dir, img_name):
    return os.path.join(root_dir, LABEL_FOLDER_NAME, img_name)


class CDDataset(Dataset):
    def __init__(self, root_dir, resolution=256, split='train', data_len=-1, label_transform=None):

        self.root_dir = root_dir
        self.resolution = resolution
        self.data_len = data_len
        self.split = split #train / val / test
        self.label_transform = label_transform

        self.list_path = os.path.join(self.root_dir, LIST_FOLDER_NAME, self.split + '.txt')

        self.img_name_list = load_img_name_list(self.list_path)

        self.dataset_len = len(self.img_name_list)

        if self.data_len <= 0:
            self.data_len = self.dataset_len
        else:
            self.data_len = min(self.dataset_len, self.data_len)

    def __len__(self):
        return self.data_len

    def __getitem__(self, index):
        A_path = get_img_path(self.root_dir, self.img_name_list[index % self.data_len])
        B_path = get_img_post_path(self.root_dir, self.img_name_list[index % self.data_len])

        img_A = Image.open(A_path).convert('RGB')
        img_B = Image.open(B_path).convert('RGB')

        L_path = get_label_path(self.root_dir, self.img_name_list[index % self.data_len])
        img_label = Image.open(L_path).convert("RGB")

        img_A = Util.transform_augment_cd(img_A, min_max=(-1, 1))
        img_B = Util.transform_augment_cd(img_B, min_max=(-1, 1))
        img_label = Util.transform_augment_cd(img_label, min_max=(0, 1))
        if img_label.dim() > 2:
            img_label = img_label[0]

        return {'A':img_A, 'B':img_B, 'L':img_label, 'Index':index}



        return A_path

if __name__ == '__main__':
    root_dir = '/data02/zht_vqa/change_detection/ddpm/LEVIR-CD256/'
    cddata = CDDataset(root_dir=root_dir)
    list_path = os.path.join(root_dir, 'list', 'train', '.txt')
    l = len(load_img_name_list)
