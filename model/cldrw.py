from .INNblock import *

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

# # 使用示例
# if __name__ == "__main__":
#     # 初始化组件
#     model = CLDRW()
    
#     # 模拟输入
#     Io = torch.randn(4, 3, 128, 128)
#     message = torch.Tensor(np.random.choice([-0.5, 0.5], (Io.shape[0], 64)))
    
#     # 计算损失
#     Iem, r = model([Io, message])

#     output_data = [Iem, r]
#     re_img, re_message = model(output_data, rev=True)

#     mse = torch.nn.MSELoss()
#     print(mse(Iem, re_img))