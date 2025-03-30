import torch
from torchvision import datasets, transforms
import numpy as np
import random
import torch.nn.functional as F
import torchvision.utils
import os
import kornia
import torch.nn as nn
import time
# from .jpeg import Jpeg_combined
import warnings
warnings.filterwarnings("ignore")
np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)



def Perspective_mapping(imgs, d=8):
    """
    透视变换
    """
    image_size = imgs.shape[-1]
    batch_size = imgs.shape[0]
    Ms = np.zeros((batch_size, 1, 3, 3))

    # 随机生成透视变换矩阵
    p_src = torch.ones(batch_size, 4, 2)
    p_dst = torch.ones(batch_size, 4, 2)
    for i in range(batch_size):
        x_lt = random.uniform(-d, d)    # 左上x坐标变换
        y_lt = random.uniform(-d, d)    # 左上y坐标变换
        x_lb = random.uniform(-d, d)    # 左下x坐标变换
        y_lb = random.uniform(-d, d)    # 左下y坐标变换
        x_rt = random.uniform(-d, d)    # 右上x坐标变换
        y_rt = random.uniform(-d, d)    # 右上y坐标变换
        x_rb = random.uniform(-d, d)    # 右下x坐标变换
        y_rb = random.uniform(-d, d)    # 右下y坐标变换

        p_src[i, :, :] = torch.tensor([
            [0, 0],
            [0, image_size],
            [image_size, 0],
            [image_size, image_size]])
            
        p_dst[i, :, :] = torch.tensor([
            [x_lt, y_lt],
            [x_lb, image_size + y_lb],
            [image_size + x_rt, y_rt],
            [image_size + x_rb, image_size + y_rb]])

    Ms = kornia.geometry.get_perspective_transform(p_src, p_dst).to(imgs.device)

    return kornia.geometry.warp_perspective(imgs.float(), Ms, dsize=(imgs.shape[-2], imgs.shape[-1])).to(imgs.device)


def Light_Distortion(imgs):
    """
    点光源噪声
    """
    img_size = imgs.shape[-1]
    mask = np.zeros((imgs.shape))
    x = np.random.randint(0,mask.shape[2])
    y = np.random.randint(0,mask.shape[3])
    max_len = np.max([np.sqrt(x**2+y**2),np.sqrt((x-img_size)**2+y**2),np.sqrt(x**2+(y-img_size)**2),np.sqrt((x-img_size)**2+(y-img_size)**2)])/2
    for i in range(mask.shape[2]):
        for j in range(mask.shape[3]):
            l = np.sqrt((i-x)**2+(j-y)**2)
            if l < max_len:
                mask[:,:,i,j] = 1 - l/max_len
    O = mask
    return torch.from_numpy(O.copy()).to(imgs.device)


def Moire_Distortion(imgs):
    """
    摩尔纹失真
    """
    masks = np.zeros((imgs.shape))
    for i in range(3):
        mask = np.zeros((imgs.shape[-2], imgs.shape[-1]))
        theta = np.random.randint(0,180)
        center_x = np.random.rand(1)*imgs.shape[-2]
        center_y = np.random.rand(1)*imgs.shape[-1]
        for j in range(imgs.shape[-2]):
            for k in range(imgs.shape[-1]):
                z1 = 0.5+0.5*np.cos(2*np.pi*np.sqrt((j+1-center_x)**2+(k+1-center_y)**2))
                z2 = 0.5+0.5*np.cos(np.cos(theta/180*np.pi)*(k+1)+np.sin(theta/180*np.pi)*(j+1))
                mask[j,k] = np.minimum(z1,z2)
        masks[:,i,:,:] = (mask+1)/2
    return torch.from_numpy((masks*2-1).copy()).to(imgs.device)


def Moire_Distortion_torch(imgs):
    """
    摩尔纹失真-pytorch版本
    """
    masks = torch.zeros_like(imgs).to(imgs.device)
    
    for i in range(3):
        # 随机生成角度和中心位置
        theta = torch.randint(0,180,(1,)).to(imgs.device)
        center_x = torch.rand(1)*imgs.shape[-2]
        center_y = torch.rand(1)*imgs.shape[-1]
        
        # 生成像素网络，以代替循环操作
        x = torch.arange(imgs.shape[-2], dtype=torch.float)
        y = torch.arange(imgs.shape[-1], dtype=torch.float)
        grid_x, grid_y = torch.meshgrid(x, y)

        # 计算中心距离
        dist_x = grid_x - center_x
        dist_y = grid_y - center_y

        grid_x, grid_y = grid_x.to(imgs.device), grid_y.to(imgs.device)
        dist_x, dist_y = dist_x.to(imgs.device), dist_y.to(imgs.device)

        # 计算两个模式
        z1 = 0.5 + 0.5*torch.cos(2*3.14159*torch.sqrt(torch.square(dist_x) + torch.square(dist_y)))
        z2 = 0.5 + 0.5*torch.cos(torch.cos(theta/180*3.14159)*grid_y + torch.sin(theta/180*3.14159)*grid_x)
        mask = torch.min(z1, z2)
        
        # 加入掩膜对应的通道
        masks[:,i,:,:] = mask
    
    # 将值映射回(-1， 1)
    masks = (masks*2)-1
    
    return masks


def Color_Manipulation(imgs, trans_rate):
    """
    颜色操作，包括色调、亮度、对比度的修改
    """
    batch_size = imgs.shape[0]
    h = trans_rate * 0.1
    b = trans_rate * 0.3
    s = trans_rate * 0.3
    hue_shift = torch.FloatTensor(batch_size, 3, 1, 1).uniform_(-h, h).to(imgs.device)
    bright_trans = torch.FloatTensor(batch_size, 1, 1, 1).uniform_(-b, b).to(imgs.device)
    sat_weight = torch.FloatTensor([.3, .6, .1]).reshape(1, 3, 1, 1).to(imgs.device)
    c_l = 1. - 0.5 * trans_rate
    c_h = 1. + 0.5 * trans_rate
    contrast_scale = torch.Tensor(batch_size).uniform_(c_l, c_h).reshape(batch_size, 1, 1, 1).to(imgs.device)
    imgs = imgs * contrast_scale
    imgs += hue_shift + bright_trans
    imgs = torch.clamp(imgs, -1, 1)
    encoded_image_lum = torch.mean(imgs * sat_weight, dim=1).unsqueeze_(1)
    encoded_image = (1 - s) * imgs + s * encoded_image_lum
    return encoded_image


class screen_shoot(nn.Module):

    def __init__(self, epoch=0):
        super(screen_shoot, self).__init__()
        self.epoch = epoch
        # self.jepg = Jpeg_combined()

    def forward(self, noised_and_cover, _=None):
        n_a_c = [noised_and_cover[0].clone(), noised_and_cover[1].clone()]
        imgs = n_a_c[0]
        # 先线性增长后固定的噪声率
        trans_rate = min(self.epoch / 50, 1)
        self.epoch += 1
        print(self.epoch)
        d = np.floor(10 * trans_rate)
        # save_images(imgs,'./test', 'orign', num, resize_to=(512, 512))

        # Perspective transform
        noised_imgs = Perspective_mapping(imgs, d)
        # save_images(noised_imgs,'./test', 'perspective',num, resize_to=(512, 512))

        # Color Manipulation
        noised_imgs = Color_Manipulation(noised_imgs, trans_rate)
        # save_images(noised_imgs,'./test', 'color', num, resize_to=(512, 512))

        # Light Distortion
        noised_imgs += Light_Distortion(imgs)*trans_rate
        # save_images(noised_imgs,'./test', 'light', num, resize_to=(512, 512))

        # Moire Distortion
        noised_imgs += Moire_Distortion_torch(imgs)*trans_rate*0.2
        # save_images(noised_imgs,'./test', 'moire', num, resize_to=(512, 512))

        # Gaussian noise
        noised_imgs = noised_imgs + 0.001**0.5*torch.randn(noised_imgs.size()).to(imgs.device)
        # save_images(noised_imgs,'./test', 'gauss', num, resize_to=(512, 512))

        # Jpeg noise
        # noised_imgs = self.jepg(noised_imgs, 0)

        n_a_c[0] = noised_imgs

        return n_a_c


# class screen_shoot_wop(nn.Module):

#     def __init__(self):
#         super(screen_shoot_wop, self).__init__()
#         # self.jepg = Jpeg_combined()

#     def forward(self, imgs, epoch, _=None):
#         # 固定的噪声率
#         trans_rate = 1
#         d = np.floor(10 * trans_rate)
#         # save_images(imgs,'./test', 'orign', num, resize_to=(512, 512))

#         # Random resize
#         # imgs = Random_resize(imgs)
#         # save_images(imgs,'./test', 'resize', num, resize_to=(512, 512))

#         # Perspective transform
#         noised_imgs = Perspective_mapping(imgs, d)
#         # save_images(noised_imgs,'./test', 'perspective',num, resize_to=(512, 512))

#         # Color Manipulation
#         noised_imgs = Color_Manipulation(noised_imgs, trans_rate)
#         # save_images(noised_imgs,'./test', 'color', num, resize_to=(512, 512))

#         # Light Distortion
#         noised_imgs += Light_Distortion(imgs)*trans_rate
#         # save_images(noised_imgs,'./test', 'light', num, resize_to=(512, 512))

#         # Moire Distortion
#         noised_imgs += Moire_Distortion_torch(imgs)*trans_rate*0.2
#         # save_images(noised_imgs,'./test', 'moire', num, resize_to=(512, 512))

#         # Gaussian noise
#         noised_imgs = noised_imgs + 0.001**0.5*torch.randn(noised_imgs.size()).to(imgs.device)
#         # save_images(noised_imgs,'./test', 'gauss', num, resize_to=(512, 512))

#         # Jpeg noise
#         # noised_imgs = self.jepg(noised_imgs, 0)

#         return noised_imgs

def get_data_loaders(path):
    """
    从指定文件夹中加载图片数据并进行处理
    """
    data_transforms = {
        'test': transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
    }

    valid_images = datasets.ImageFolder(path, data_transforms['test'])
    valid_loader = torch.utils.data.DataLoader(valid_images, batch_size=10, 
                                                shuffle=False, num_workers=0)

    return valid_loader
def save_images(imgs, imgs_w, imgs_num, epoch, folder, resize_to=None):
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

    stacked_images = torch.cat([images, imgs_w, imgs_residual*5], dim=0)
    filename = os.path.join(folder, 'epoch-{}.png'.format(epoch))
    torchvision.utils.save_image(stacked_images, filename, normalize=False, nrow=imgs_num)


# if __name__ == '__main__':
#     device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
#     path = './data/valid'
#     test_data = get_data_loaders(path)
#     epoch = 401
#     model = screen_shoot()

#     num = 1
#     for imgs, _ in test_data:
#         t1 = time.time()
#         imgs = imgs.to(device)
#         t2 = time.time()
#         print('计算时间：%.3f'%(t2-t1))
#         noised_imgs = model([imgs, _])[0]
#         save_images(noised_imgs,'./test', 'noise', num)
#         num += imgs.shape[0]
#         epoch += 1

# from gaussian import Gaussian_blur

# if __name__ == '__main__':
#     device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
#     path = './data/valid'
#     test_data = get_data_loaders(path)
#     model = Gaussian_blur(3, 0.2)

#     num = 1
#     for imgs, _ in test_data:
#         imgs_j = model(imgs)
#         save_images(imgs, imgs_j, 8, 1, './test', (256, 256))
#         num += imgs.shape[0]