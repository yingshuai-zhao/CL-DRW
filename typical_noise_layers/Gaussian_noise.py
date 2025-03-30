import numpy as np
import torch.nn as nn
import torch

class Gaussian_Noise(nn.Module):
    def __init__(self,mean,sigma):
        super(Gaussian_Noise, self).__init__()
        self.mean=float(mean)
        self.sigma=float(sigma)

    def forward(self, noise_and_cover):
        n_a_c = [noise_and_cover[0].clone(), noise_and_cover[1].clone()]
        encode_image=n_a_c[0]
        B,C,H,W=encode_image.size()
        noise = np.clip(np.random.normal(self.mean,self.sigma , (B,1,H, W)), 0, 1)
        noise=torch.tensor(noise,device= encode_image.device)
        for i in range(C):
            encode_image[:,i:,:,:,]=encode_image[:,i:,:,:,]+noise*0.25
        n_a_c[0]=encode_image
        return n_a_c