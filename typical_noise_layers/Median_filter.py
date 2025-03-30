import PIL
import torch
import torch.nn as nn
import kornia.filters as filters
class Median_filter(nn.Module):
    def __init__(self,kernel_size):
        super(Median_filter, self).__init__()
        self.kernel_size=int(kernel_size)
        self.device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    def forward(self, noised_and_cover):
        n_a_c = [noised_and_cover[0].clone(), noised_and_cover[1].clone()]
        encoded=n_a_c[0].clone()
        n_a_c[0]=filters.MedianBlur(kernel_size=(self.kernel_size,self.kernel_size))(encoded)
        # noise_and_cover[0]=filters.motion.MotionBlur(ksize=7,angle=35.,direction=-0.5,border_type='reflect')(encoded)
        return n_a_c

