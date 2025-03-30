import torch
import torch.nn as nn


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


class Moire(nn.Module):

    def __init__(self, factor):
        super(Moire, self).__init__()
        self.factor = factor

    def forward(self, noised_and_cover):
        n_a_c = [noised_and_cover[0].clone(), noised_and_cover[1].clone()]
        imgs = n_a_c[0]

        # Moire Distortion
        noised_imgs = imgs + Moire_Distortion_torch(imgs) * self.factor

        n_a_c[0] = noised_imgs

        return n_a_c