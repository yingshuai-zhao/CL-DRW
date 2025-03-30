import torch.nn as nn
import torch.nn.functional as F


class Resize(nn.Module):
    """
    Resize the image.
    """
    def __init__(self, opt):
        super(Resize, self).__init__()
        resize_ratio_down = 0.5
        self.h = 128
        self.w = 128
        self.scaled_h = int(resize_ratio_down * self.h)
        self.scaled_w = int(resize_ratio_down * self.w)
        self.interpolation_method = 'nearest'

    def forward(self, noised_and_cover):
        n_a_c = [noised_and_cover[0].clone(), noised_and_cover[1].clone()]
        wm_imgs = n_a_c[0]
        #
        noised_down = F.interpolate(
                                    wm_imgs,
                                    size=(self.scaled_h, self.scaled_w),
                                    mode=self.interpolation_method
                                    )
        n_a_c[0] = F.interpolate(
                                    noised_down,
                                    size=(self.h, self.w),
                                    mode=self.interpolation_method
                                    )

        return n_a_c
