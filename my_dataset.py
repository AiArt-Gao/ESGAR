import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

class FS2K_DataSet(Dataset):
    """自定义数据集"""

    def __init__(self, attrs, transform=None):
        self.images_path = attrs['image_name']
        self.hair = attrs['hair']
        self.hair_color = attrs['hair_color']  # sketch不用hair_color
        self.gender = attrs['gender']
        self.earring = attrs['earring']
        self.smile = attrs['smile']
        self.frontal_face = attrs['frontal_face']
        self.style = attrs['style']  # photo不用style
        self.transform = transform

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, item):
        # 图像路径
        img_path = self.images_path[item]

        # 读取图像
        # img = Image.open(img_path).convert('L')
        img = Image.open(img_path).convert('RGB')

        # 图像增强
        if self.transform:
            img = self.transform(img)

        # 返回图像和所有相关标签
        dict_data = {
            'img': img,
            'labels': {
                'hair': self.hair[item],
                'hair_color': self.hair_color[item],
                'gender': self.gender[item],
                'earring': self.earring[item],
                'smile': self.smile[item],
                'frontal_face': self.frontal_face[item],
                'style': self.style[item]
            }
        }
        return dict_data



def get_parsing(seg_parsing):
    seg_parsing = np.array(seg_parsing)
    seg_parsing = seg_parsing.mean(axis=2)
    seg_parsing[seg_parsing >= 150] = 255
    seg_parsing[seg_parsing < 150] = 0
    seg_parsing = seg_parsing // 255
    seg_parsing = seg_parsing.reshape(7, 32, 7, 32)
    # seg_earring = seg_earring.reshape(14, 16, 14, 16)
    seg_parsing = seg_parsing.mean(axis=(1, 3))
    seg_parsing[seg_parsing >= 0.031] = 1
    seg_parsing = seg_parsing.flatten()
    return seg_parsing

def seg_to_img(seg_path):
    seg_img = Image.open(seg_path)
    seg_img = np.array(seg_img)
    seg_img = np.expand_dims(seg_img, axis=0)
    seg_img = np.repeat(seg_img, 3, axis=0)
    seg_img = seg_img.transpose((1, 2, 0))
    seg_img = Image.fromarray(seg_img)
    return seg_img

class FS2K_DataSet_With_Seg(Dataset):
    """自定义数据集"""

    def __init__(self, attrs, transform=None, t2=None):
        self.images_path = attrs['image_name']
        self.hair = attrs['hair']
        self.hair_color = attrs['hair_color']  # sketch不用hair_color
        self.gender = attrs['gender']
        self.earring = attrs['earring']
        self.smile = attrs['smile']
        self.frontal_face = attrs['frontal_face']
        self.style = attrs['style']  # photo不用style
        self.transform = transform
        self.t2 = t2

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, item):
        # 图像路径
        img_path = self.images_path[item]
        img_name = img_path.split('/')[-1]
        style_label, index = img_name.split('_')

        seg_earring_path = f'/data2/yuhao/FS2K/FS2K/seg_earring/photo{style_label}/{index}'
        seg_earring = seg_to_img(seg_earring_path)
        seg_smile_path = f'/data2/yuhao/FS2K/FS2K/seg_face/photo{style_label}/{index}'
        seg_smile = seg_to_img(seg_smile_path)
        seg_hair_path = f'/data2/yuhao/FS2K/FS2K/seg_hair/photo{style_label}/{index}'
        seg_hair = seg_to_img(seg_hair_path)
        seg_total_path = f'/data2/yuhao/FS2K/FS2K/seg_ex_background/photo{style_label}/{index}'
        seg_total = seg_to_img(seg_total_path)

        # 读取图像
        img = Image.open(img_path).convert('RGB')

        # 图像增强
        if self.transform:
            img, seg_earring, seg_smile, seg_hair, seg_total = self.transform(img, seg_earring, seg_smile, seg_hair, seg_total)
            img = self.t2(img)

        seg_smile = get_parsing(seg_smile)
        seg_earring = get_parsing(seg_earring)
        seg_hair = get_parsing(seg_hair)
        seg_total = get_parsing(seg_total)

        # 返回图像和所有相关标签
        dict_data = {
            'img': img,
            'labels': {
                'hair': self.hair[item],
                'hair_color': self.hair_color[item],
                'gender': self.gender[item],
                'earring': self.earring[item],
                'smile': self.smile[item],
                'frontal_face': self.frontal_face[item],
                'style': self.style[item],
                'seg_earring': seg_earring.astype(np.float32),
                'seg_smile': seg_smile.astype(np.float32),
                'seg_hair': seg_hair.astype(np.float32),
                'seg_total': seg_total.astype(np.float32)
            }
        }
        return dict_data
