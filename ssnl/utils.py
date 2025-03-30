from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import torchvision.transforms as T
import config as cfgs
import logging
from datetime import datetime
import torch
import torchvision.transforms.functional as F
import numpy as np
import csv
import kornia


class AverageMeter(object):
    """
    各loss值计算
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        if val != np.nan and val != np.inf:
            self.val = val
            self.sum += val * n
            self.count += n
            self.avg = self.sum / self.count



# 自定义数据集类
class T_V_Dataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.images = sorted(os.listdir(image_dir))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.images[idx])
        image = Image.open(image_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image



class Test_Dataset_Randcrop(Dataset):
    def __init__(self, pre_dir, post_dir, transform=None, crop_size=(128, 128)):
        self.pre_dir = pre_dir
        self.post_dir = post_dir
        self.transform = transform
        self.crop_size = crop_size
        self.pre_images = sorted(os.listdir(pre_dir))
        self.post_images = sorted(os.listdir(post_dir))

    def __len__(self):
        return len(self.pre_images)

    def __getitem__(self, idx):
        pre_image_path = os.path.join(self.pre_dir, self.post_images[idx])
        post_image_path = os.path.join(self.post_dir, self.post_images[idx])
        pre_image = Image.open(pre_image_path).convert('RGB')
        post_image = Image.open(post_image_path).convert('RGB')

        # 获取随机裁剪参数
        i, j, h, w = self.get_random_crop_params(pre_image, self.crop_size)
        
        # 对两张图像进行相同的随机裁剪
        pre_image = F.crop(pre_image, i, j, h, w)
        post_image = F.crop(post_image, i, j, h, w)

        if self.transform:
            pre_image = self.transform(pre_image)
            post_image = self.transform(post_image)

        return pre_image, post_image

    def get_random_crop_params(self, img, output_size):
        w, h = img.size
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w
        i = torch.randint(0, h - th + 1, size=(1,)).item()
        j = torch.randint(0, w - tw + 1, size=(1,)).item()
        return i, j, th, tw


class Test_Dataset(Dataset):
    def __init__(self, pre_dir, post_dir, transform=None, crop_size=(128, 128)):
        self.pre_dir = pre_dir
        self.post_dir = post_dir
        self.transform = transform
        self.crop_size = crop_size
        self.pre_images = sorted(os.listdir(pre_dir))
        self.post_images = sorted(os.listdir(post_dir))

    def __len__(self):
        return len(self.post_images)

    def __getitem__(self, idx):
        pre_image_path = os.path.join(self.pre_dir, self.pre_images[idx])
        post_image_path = os.path.join(self.post_dir, self.post_images[idx])
        pre_image = Image.open(pre_image_path).convert('RGB')
        post_image = Image.open(post_image_path).convert('RGB')

        if self.transform:
            pre_image = self.transform(pre_image)
            post_image = self.transform(post_image)
            # psnr = kornia.losses.psnr_loss(pre_image, post_image, 2)
            # print(psnr)

        return pre_image, post_image


def load_data_randcrop(istest=False, isINN=False, iscrop=False):
    if istest:
        # 数据预处理
        if iscrop:
            transform = T.Compose([
                T.CenterCrop((128, 128)),
                T.ToTensor(),
                T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ])
        else:
            transform = T.Compose([
                T.ToTensor(),
                T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ])

        # 加载数据集
        if isINN:
            dataset = Test_Dataset_Randcrop(cfgs.TRAIN_ORI_PATH, cfgs.TRAIN_OSN_PATH, transform=transform)
            train_dataloader = DataLoader(dataset, batch_size=cfgs.train_batchsize_INN, shuffle=True)
            dataset = Test_Dataset_Randcrop(cfgs.VALID_ORI_PATH, cfgs.VALID_OSN_PATH, transform=transform)
            valid_dataloader = DataLoader(dataset, batch_size=cfgs.train_batchsize_INN, shuffle=True)
        else:
            dataset = Test_Dataset_Randcrop(cfgs.TRAIN_ORI_PATH, cfgs.TRAIN_OSN_PATH, transform=transform)
            train_dataloader = DataLoader(dataset, batch_size=cfgs.train_batchsize, shuffle=True)
            dataset = Test_Dataset_Randcrop(cfgs.VALID_ORI_PATH, cfgs.VALID_OSN_PATH, transform=transform)
            valid_dataloader = DataLoader(dataset, batch_size=cfgs.train_batchsize, shuffle=True)

        return train_dataloader, valid_dataloader
    


def load_data(istest=False, isINN=False, iscrop=False):
    if istest:
        # 数据预处理
        if iscrop:
            transform = T.Compose([
                T.CenterCrop((128, 128)),
                T.ToTensor(),
                T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ])
        else:
            transform = T.Compose([
                T.ToTensor(),
                T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ])

        # 加载数据集
        if isINN:
            dataset = Test_Dataset(cfgs.TRAIN_ORI_PATH, cfgs.TRAIN_OSN_PATH, transform=transform)
            train_dataloader = DataLoader(dataset, batch_size=cfgs.train_batchsize_INN, shuffle=True)
            dataset = Test_Dataset(cfgs.VALID_ORI_PATH, cfgs.VALID_OSN_PATH, transform=transform)
            valid_dataloader = DataLoader(dataset, batch_size=cfgs.train_batchsize_INN, shuffle=False)
        else:
            dataset = Test_Dataset(cfgs.TRAIN_ORI_PATH, cfgs.TRAIN_OSN_PATH, transform=transform)
            train_dataloader = DataLoader(dataset, batch_size=cfgs.train_batchsize, shuffle=True)
            dataset = Test_Dataset(cfgs.VALID_ORI_PATH, cfgs.VALID_OSN_PATH, transform=transform)
            valid_dataloader = DataLoader(dataset, batch_size=cfgs.train_batchsize, shuffle=False)

        return train_dataloader, valid_dataloader


def setup_logger(logger_name, root, phase, level=logging.INFO, screen=False, tofile=False):
    lg = logging.getLogger(logger_name)
    formatter = logging.Formatter('%(asctime)s.%(msecs)03d - %(levelname)s: %(message)s',
                                  datefmt='%y-%m-%d %H:%M:%S')
    lg.setLevel(level)
    if tofile:
        log_file = os.path.join(root, phase + '_{}.log'.format(get_timestamp()))
        fh = logging.FileHandler(log_file, mode='w')
        fh.setFormatter(formatter)
        lg.addHandler(fh)
    if screen:
        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        lg.addHandler(sh)

def get_timestamp():
    return datetime.now().strftime('%y%m%d-%H%M%S')