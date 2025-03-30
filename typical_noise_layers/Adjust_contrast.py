import torch.nn as nn
import torch
from kornia.enhance.adjust import AdjustHue,AdjustSaturation,AdjustContrast,AdjustBrightness,AdjustGamma
import math
from torchvision.transforms import ToPILImage
class Adjust_contrast(nn.Module):
    def __init__(self,factor):
        super(Adjust_contrast, self).__init__()
        self.factor=factor

    def forward(self, noised_and_cover):
        n_a_c = [noised_and_cover[0].clone(), noised_and_cover[1].clone()]
        encoded=((n_a_c[0]))
        encoded=AdjustContrast(contrast_factor=self.factor)(encoded)
        n_a_c[0]=(encoded)
        return n_a_c