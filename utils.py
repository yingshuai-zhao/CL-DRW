import os
import logging
from datetime import datetime
import torch
import numpy as np
from torchvision import datasets, transforms
import torch.nn.functional as F
import torchvision
from torch.nn.functional import mse_loss as mse


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


def get_data_loaders(train_folder, valid_folder, train_batchsize, valid_batchsize, cropsize=128):
    """
    从指定文件夹中加载图片数据并进行处理
    """
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomCrop([cropsize, cropsize], pad_if_needed=True),
            torchvision.transforms.RandomHorizontalFlip(),    # 水平镜像
            torchvision.transforms.RandomVerticalFlip(),      # 竖直镜像
            torchvision.transforms.RandomRotation(45),        # 随机旋转
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ]),
        'valid': transforms.Compose([
            transforms.CenterCrop((cropsize, cropsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ]),
        'test': transforms.Compose([
            transforms.Resize([cropsize, cropsize]),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
    }

    train_images = datasets.ImageFolder(train_folder, data_transforms['train'])
    train_loader = torch.utils.data.DataLoader(train_images, batch_size=train_batchsize,
                                                shuffle=True, pin_memory=True, num_workers=8, drop_last=True)

    valid_images = datasets.ImageFolder(valid_folder, data_transforms['valid'])
    valid_loader = torch.utils.data.DataLoader(valid_images, batch_size=valid_batchsize, 
                                                shuffle=False, pin_memory=True, num_workers=8, drop_last=True)

    print('数据读取成功！')
    return train_loader, valid_loader


def get_data_loaders_test(test_folder, test_batchsize, cropsize=128):
    """
    从指定文件夹中加载图片数据并进行处理
    """
    data_transforms = transforms.Compose([
            transforms.Resize([cropsize, cropsize]),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])

    test_images = datasets.ImageFolder(test_folder, data_transforms)
    test_loader = torch.utils.data.DataLoader(test_images, batch_size=test_batchsize, 
                                                shuffle=False, pin_memory=True, num_workers=8, drop_last=True)

    print('数据读取成功！')
    return test_loader


def save_valid_images(imgs, imgs_w, imgs_num, epoch, folder, resize_to=None):
    """
    保存第epoch轮验证过程第一个batch的前imgs_num张图片
    """
    images = imgs[:imgs_num, :, :, :].cpu()
    imgs_w = imgs_w[:imgs_num, :, :, :].cpu()

    # scale values to range [0, 1] from original range of [-1, 1]
    images = (images + 1) / 2
    imgs_w = (imgs_w + 1) / 2

    if resize_to is not None:
        images = F.interpolate(images, size=resize_to)
        imgs_w = F.interpolate(imgs_w, size=resize_to)
    imgs_residual = images - imgs_w

    stacked_images = torch.cat([images, imgs_w, imgs_residual*2], dim=0)
    filename = os.path.join(folder, 'epoch-{}.png'.format(epoch))
    torchvision.utils.save_image(stacked_images, filename, normalize=False, nrow=imgs_num, padding=3, pad_value=1)


def save_images(imgs, folder, num=1, name='results', resize_to=None):
    """
    保存图片并以序号num命名
    """
    # scale values to range [0, 1] from original range of [-1, 1]
    imgs = (imgs + 1) / 2

    if resize_to is not None:
        imgs = F.interpolate(imgs, size=resize_to)
    for i in range(num, num+imgs.shape[0]):
        filename = os.path.join(folder, name+'-{}.jpg'.format(i))
        torchvision.utils.save_image(imgs[i-num], filename, normalize=False)
    return num+imgs.shape[0]


def load(model, name):
    state_dicts = torch.load(name)
    network_state_dict = {k: v for k, v in state_dicts['net'].items() if 'tmp_var' not in k}
    model.load_state_dict(network_state_dict)
    
    
def clamp(input, min=None, max=None):
    ndim = input.ndimension()
    if min is None:
        pass
    elif isinstance(min, (float, int)):
        input = torch.clamp(input, min=min)
    elif isinstance(min, torch.Tensor):
        if min.ndimension() == ndim - 1 and min.shape == input.shape[1:]:
            input = torch.max(input, min.view(1, *min.shape))
        else:
            assert min.shape == input.shape
            input = torch.max(input, min)
    else:
        raise ValueError("min can only be None | float | torch.Tensor")

    if max is None:
        pass
    elif isinstance(max, (float, int)):
        input = torch.clamp(input, max=max)
    elif isinstance(max, torch.Tensor):
        if max.ndimension() == ndim - 1 and max.shape == input.shape[1:]:
            input = torch.min(input, max.view(1, *max.shape))
        else:
            assert max.shape == input.shape
            input = torch.min(input, max)
    else:
        raise ValueError("max can only be None | float | torch.Tensor")
    return input


def decoded_message_error_rate(message, decoded_message):
    message = message.view(message.shape[0], -1).squeeze()
    length = message.shape[0]
    message = message.gt(0)
    decoded_message = decoded_message.gt(0)
    error_rate = float(sum(message != decoded_message)) / length
    return error_rate


def decoded_message_error_rate_batch(messages, decoded_messages):
    error_rate = 0.0
    batch_size = len(messages)
    for i in range(batch_size):
        error_rate += decoded_message_error_rate(messages[i], decoded_messages[i])
    error_rate /= batch_size
    return error_rate


def setup_logger(logger_name, root, phase, level=logging.INFO, screen=False, tofile=False):
    lg = logging.getLogger(logger_name)
    formatter = logging.Formatter('%(asctime)s.%(msecs)03d - %(levelname)s: %(message)s',
                                  datefmt='%y-%m-%d %H:%M:%S')
    lg.setLevel(level)
    if tofile:
        log_file = os.path.join(root, phase + '_{}.log'.format(datetime.now().strftime('%y%m%d-%H%M%S')))
        fh = logging.FileHandler(log_file, mode='w')
        fh.setFormatter(formatter)
        lg.addHandler(fh)
    if screen:
        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        lg.addHandler(sh)


def psnr(input: torch.Tensor, target: torch.Tensor, max_val: float) -> torch.Tensor:
    input = clamp(((input.detach().cpu().squeeze()/2)+0.5) * max_val, 0, max_val)
    target = clamp(((target.detach().cpu().squeeze()/2)+0.5) * max_val, 0, max_val)
    if not isinstance(input, torch.Tensor):
        raise TypeError(f"Expected torch.Tensor but got {type(target)}.")

    if not isinstance(target, torch.Tensor):
        raise TypeError(f"Expected torch.Tensor but got {type(input)}.")

    if input.shape != target.shape:
        raise TypeError(f"Expected tensors of equal shapes, but got {input.shape} and {target.shape}")

    return 10.0 * torch.log10(max_val ** 2 / mse(input, target, reduction='mean'))


