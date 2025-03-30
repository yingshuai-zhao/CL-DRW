import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from .models.diff_jpeg.jpeg import DiffJPEGCoding
from .models.models import Unet_OSN, Unet_OSN_DB, Restormer
from .models.vgg_loss import VGGLoss, Gram_VGGLoss
from .models.unetpp import UNetPP
from .models.discriminator import Discriminator

import config as cfgs
from .utils import *
import kornia
import numpy as np
from collections import defaultdict
import torchvision

def load_model(net, optim, path):
    checkpoint = torch.load(path)
    net.load_state_dict(checkpoint['net'])
    optim.load_state_dict(checkpoint['optim'])


def save_model(net, optim, path):
    checkpoint = {
        'net'       :net.state_dict(),
        'optim':optim.state_dict(),
    }

    torch.save(checkpoint, path)



def save_images(imgs, folder, num=1, name='results', resize_to=None):
    """
    保存第epoch轮验证过程第一个batch的前imgs_num张图片
    """
    # scale values to range [0, 1] from original range of [-1, 1]
    imgs = (imgs + 1) / 2

    if resize_to is not None:
        imgs = F.interpolate(imgs, size=resize_to)
    for i in range(num, num+imgs.shape[0]):
        filename = os.path.join(folder, name+'-{}.jpg'.format(i))
        torchvision.utils.save_image(imgs[i-num], filename, normalize=False)



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

    stacked_images = torch.cat([images, imgs_w, imgs_residual*5], dim=0)
    filename = os.path.join(folder, 'epoch-{}.png'.format(epoch))
    torchvision.utils.save_image(stacked_images, filename, normalize=False, nrow=imgs_num)



def train(net, disc, optim_Unet, optmizer_disc, device):
    mse_loss = nn.MSELoss().to(device)

    # bce_with_logits_loss = nn.BCEWithLogitsLoss().to(device)

    diff_jpeg = DiffJPEGCoding(ste=True)
    
    train_data_loader, valid_data_loader = load_data(istest=True, iscrop=True)

    for epoch in range(cfgs.epochs):

        #######################
        #        train        #
        #######################
        net.train()

        train_losses = defaultdict(AverageMeter)

        for i, data in enumerate(train_data_loader, 0):
            with torch.enable_grad():
                cover_images, groundtruth_images = data
                cover_images, groundtruth_images = cover_images.to(device), groundtruth_images.to(device)

                # 初始化字典
                result_dict = {}

                # # 循环从1到99
                # for i in range(1, 100):
                #     Q = torch.Tensor(np.random.choice([i], (cover_images.shape[0], ))).to(device)
                #     recover_images = diff_jpeg(cover_images, Q)
                #     # 生成一个随机数作为示例
                #     value = -kornia.losses.psnr_loss(recover_images.detach(), groundtruth_images, 2)
                #     # 将循环轮数i作为键，生成的随机数作为值存入字典
                #     result_dict[i] = value

                # # 找到字典中值最大的前五个键
                # top_5_keys = sorted(result_dict, key=result_dict.get, reverse=True)[:5]
                # top_5_values = [result_dict[key] for key in top_5_keys]

                # print(f"前五大值对应的索引是: {top_5_keys}")
                # print(f"前五大值是: {top_5_values}")

                # value1 = -kornia.losses.psnr_loss(cover_images.detach(), groundtruth_images, 2)
                # Q = torch.Tensor(np.random.choice([97], (cover_images.shape[0], ))).to(device)
                # recover_images = diff_jpeg(cover_images, Q)
                # value2 = -kornia.losses.psnr_loss(recover_images.detach(), groundtruth_images, 2)
                # print(value2-value1)

                # continue
                
                

                # # -------------------------训练鉴别器------------------------
                # optmizer_disc.zero_grad()
                # # groundtruth图片上训练鉴别器
                # d_target_no_watered = torch.full((groundtruth_images.shape[0], 1), 0, device=cover_images.device)
                # d_predict_no_watered = disc(groundtruth_images.detach())
                # d_no_watered_loss = bce_with_logits_loss(d_predict_no_watered, d_target_no_watered.float())
                # d_no_watered_loss.backward()
                # # 生成的图片上训练鉴别器
                # d_target_watered = torch.full((groundtruth_images.shape[0], 1), 1, device=cover_images.device)
                # d_predict_watered = disc(recover_images.detach())
                # d_watered_loss = bce_with_logits_loss(d_predict_watered, d_target_watered.float())
                # d_watered_loss.backward()
                # optmizer_disc.step()


                optim_Unet.zero_grad()
                
                recover_images = net(cover_images)
                Q = torch.Tensor(np.random.choice([99], (cover_images.shape[0], ))).to(device)
                recover_images = diff_jpeg(recover_images, Q)

                # g_target_watered = torch.full((groundtruth_images.shape[0], 1), 0, device=cover_images.device)
                # d_predict_watered_for_g = disc(recover_images)
                # g_adv_loss = bce_with_logits_loss(d_predict_watered_for_g, g_target_watered.float())
                
                loss1 = mse_loss(groundtruth_images, recover_images)

                # loss = loss1 + 0.001*g_adv_loss

                loss1.backward()
                optim_Unet.step()

                psnr = kornia.losses.psnr_loss(recover_images.detach(), groundtruth_images, 2)

                train_losses['psnr'].update(-psnr)
                train_losses['loss'].update(loss1.item())


        #######################
        #        valid        #
        #######################
        net.eval()

        valid_losses = defaultdict(AverageMeter)

        for i, data in enumerate(valid_data_loader, 0):
            with torch.no_grad():
                cover_images, groundtruth_images = data
                cover_images, groundtruth_images = cover_images.to(device), groundtruth_images.to(device)


                recover_images = net(cover_images)

                Q = torch.Tensor(np.random.choice([95, 96, 97, 98, 99], (cover_images.shape[0], ))).to(device)
                recover_images = diff_jpeg(recover_images, Q)
                
                loss2 = mse_loss(groundtruth_images, recover_images) # + 0.1*vgg_loss(groundtruth_images, recover_images)

                psnr1 = kornia.losses.psnr_loss(recover_images.detach(), groundtruth_images, 2)
                    
                save_valid_images(recover_images, groundtruth_images, 4, epoch, './test_pics')

                exit(0)

                valid_losses['psnr'].update(-psnr1)
                valid_losses['loss'].update(loss2.item())

            
        if (epoch % cfgs.SAVE_freq) == 0:
            save_model(net, optim_Unet, cfgs.MODEL_PATH + 'sim_osn_' + '_%.3i' % epoch + '.pt')

        
        logger_train.info(
            f"Train epoch {epoch}:   "
            # f'LOSS: {train_losses["loss"].avg:.6f} | '
            # f'PSNR: {train_losses["psnr"].avg:.6f} | '
            f'VALID_LOSS: {valid_losses["loss"].avg:.6f} | '
            f'VALID_PSNR: {valid_losses["psnr"].avg:.6f} | '
            # f'E-T-D PSNR: {-psnr2:.3f} | '
        )
            
    print('Finished Training')



if __name__ == '__main__':
    device = torch.device("cuda:6" if torch.cuda.is_available() else "cpu")

    net = Restormer().to(device)
    disc = Discriminator()

    setup_logger('train', 'logging', 'train_', level=logging.INFO, screen=True, tofile=True)
    logger_train = logging.getLogger('train')

    optim_Resformer = torch.optim.Adam(net.parameters(), lr=cfgs.lr, betas=cfgs.betas, eps=1e-6, weight_decay=cfgs.weight_decay)
    optmizer_disc = torch.optim.Adam(disc.parameters(), lr=cfgs.lr)

    scheduler = ReduceLROnPlateau(optim_Resformer, patience=10, factor=0.5,mode='min')
    

    if cfgs.train_continue: 
        load_model(net, optim_Resformer, 'experiments/Uformer-FB/sim_osn__013.pt')
    

    train(net, disc, optim_Resformer, optmizer_disc, device)
    
