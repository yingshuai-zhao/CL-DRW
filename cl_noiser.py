import numpy as np
import torch.nn as nn
from typical_noise_layers.identity import Identity
from typical_noise_layers.jpeg import JpegSS, JpegTest, JpegSS_
from typical_noise_layers import DiffJPEGCoding_
from typical_noise_layers.dropout import Dropout
from typical_noise_layers.Gaussian_noise import Gaussian_Noise
from typical_noise_layers.cropout import Cropout
from typical_noise_layers.crop import Crop
from typical_noise_layers.resize import Resize
from typical_noise_layers.gaussian import Gaussian_blur
from typical_noise_layers.salt_and_pepper import Salt_and_Pepper
from typical_noise_layers.moire import Moire


class Noiser(nn.Module):
    """
    This module allows to combine different noise layers into a sequential noise module. The
    configuration and the sequence of the noise layers is controlled by the noise_config parameter.
    """
    def __init__(self, istest=False):
        super(Noiser, self).__init__()
        if istest:
            self.noise_layers = [Identity(), Gaussian_Noise(0, 1), Dropout([0.3, 0.3]), Salt_and_Pepper(0.05), Resize([0.5, 0.5]), DiffJPEGCoding_(40, ste=True)]
            
        else:
            self.noise_layers = [Identity(), Gaussian_Noise(0, 1), Salt_and_Pepper(0.05), Resize([0.4, 0.5]), Dropout([0.3, 0.4]), DiffJPEGCoding_(40, ste=True)]


    def forward(self, encoded_and_cover, i_epoch):
        if i_epoch <= 5:
            p = [1, 0, 0, 0, 0, 0]
        elif i_epoch <= 10:
            p = [0, 1, 0, 0, 0, 0]
        elif i_epoch <= 15:
            p = [0, 0.5, 0.5, 0, 0, 0]
        elif i_epoch <= 20:
            p = [0, 0.3, 0.3, 0.4, 0, 0]
        elif i_epoch <= 30:
            p = [0, 0.25, 0.25, 0.25, 0.25, 0]
        else:
            p = [0, 0.2, 0.2, 0.2, 0.2, 0.2]
        random_noise_layer = np.random.choice(self.noise_layers, 1, p=p)[0]
        "If you want to use specific noise, you can set it like this. " 
        "self.noise_layers[i](encoded_and_cover)"
        # print(random_noise_layer, end=' ')
        return random_noise_layer(encoded_and_cover)