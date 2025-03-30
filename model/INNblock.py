import torch
import torch.nn as nn
from .layers import *

class IB(nn.Module):
    def __init__(self, H=128, length=64, clamp=2.0):
        super().__init__()

        self.clamp = clamp
        self.r = DownNet(H, length)
        self.y = DownNet(H, length)
        self.f = CopyUpNet(H, length)
        
    def e(self, s):
        return torch.exp(self.clamp * 2 * (torch.sigmoid(s) - 0.5))

    def forward(self, x, rev=False):

        x1, x2 = x[0], x[1]

        if not rev:
            t2 = self.f(x2)
            y1 = x1 + t2
            
            s1, t1 = self.r(y1), self.y(y1)

            y2 = torch.exp(s1) * x2 + t1

            out = [y1, y2]

        else:
            s1, t1 = self.r(x1), self.y(x1)
            y2 = (x2 - t1) / torch.exp(s1)

            t2 = self.f(y2)
            y1 = (x1 - t2)

            out = [y1, y2]
        return out


class CLDRW(nn.Module):
    def __init__(self, H=128, length=64, diff=False):
        super(CLDRW, self).__init__()
        self.diff = diff
        self.inv1 = IB(H, length)
        self.inv2 = IB(H, length)
        self.inv3 = IB(H, length)
        self.inv4 = IB(H, length)
        self.inv5 = IB(H, length)
        self.inv6 = IB(H, length)
        self.inv7 = IB(H, length)
        self.inv8 = IB(H, length)
        if self.diff:
            self.diff_layer_pre = nn.Linear(length, 256)
            self.leakrelu = nn.LeakyReLU(inplace=True)
            self.diff_layer_post = nn.Linear(256, length)

    def forward(self, x, rev=False):
        if not rev:
            if self.diff:
                x1, x2 = x[0], x[1]
                x2 = self.leakrelu(self.diff_layer_pre(x2))
                x = [x1, x2]
            out = self.inv1(x)
            out = self.inv2(out)
            out = self.inv3(out)
            out = self.inv4(out)
            out = self.inv5(out)
            out = self.inv6(out)
            out = self.inv7(out)
            out = self.inv8(out)
        else:
            out = self.inv8(x, rev=True)
            out = self.inv7(out, rev=True)
            out = self.inv6(out, rev=True)
            out = self.inv5(out, rev=True)
            out = self.inv4(out, rev=True)
            out = self.inv3(out, rev=True)
            out = self.inv2(out, rev=True)
            out = self.inv1(out, rev=True)
            if self.diff:
                x1, x2 = out
                x2 = self.diff_layer_post(x2)
                out = [x1, x2]
        return out

