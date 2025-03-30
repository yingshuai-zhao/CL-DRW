import torch
import torch.nn as nn
import numpy as np

class Salt_and_Pepper(nn.Module):
    def __init__(self,ratio):
        super(Salt_and_Pepper, self).__init__()
        self.ratio=float(ratio)


    def forward(self, noised_and_cover):
        n_a_c = [noised_and_cover[0].clone(), noised_and_cover[1].clone()]
        encoded_image=n_a_c[0]
        B,C,H,W=encoded_image.size()
        mask = np.random.choice((0, 1, 2), size=(B,1,H,W), p=[self.ratio, self.ratio, 1 - 2 * self.ratio])
        mask=torch.tensor(np.repeat(mask,C, axis=1),device=encoded_image.device)
        encoded_image[mask==0]=-1
        encoded_image[mask==1]=1
        n_a_c[0]=encoded_image
        return n_a_c

