import torch
import numpy as np


class DCT_IDCT:
    def __init__(self):
        super(DCT_IDCT, self).__init__()
    
    def dct(self, image):
            # coff for dct and idct
            coff = torch.zeros((8, 8), dtype=torch.float).to(image.device)
            coff[0, :] = 1 * np.sqrt(1 / 8)
            for i in range(1, 8):
                for j in range(8):
                    coff[i, j] = np.cos(np.pi * i * (2 * j + 1) / (2 * 8)) * np.sqrt(2 / 8)

            split_num = image.shape[2] // 8
            image_dct = torch.cat(torch.cat(image.split(8, 2), 0).split(8, 3), 0)
            image_dct = torch.matmul(coff, image_dct)
            image_dct = torch.matmul(image_dct, coff.permute(1, 0))
            image_dct = torch.cat(torch.cat(image_dct.chunk(split_num, 0), 3).chunk(split_num, 0), 2)

            return image_dct

    def idct(self, image_dct):
        # coff for dct and idct
        coff = torch.zeros((8, 8), dtype=torch.float).to(image_dct.device)
        coff[0, :] = 1 * np.sqrt(1 / 8)
        for i in range(1, 8):
            for j in range(8):
                coff[i, j] = np.cos(np.pi * i * (2 * j + 1) / (2 * 8)) * np.sqrt(2 / 8)

        split_num = image_dct.shape[2] // 8
        image = torch.cat(torch.cat(image_dct.split(8, 2), 0).split(8, 3), 0)
        image = torch.matmul(coff.permute(1, 0), image)
        image = torch.matmul(image, coff)
        image = torch.cat(torch.cat(image.chunk(split_num, 0), 3).chunk(split_num, 0), 2)

        return image