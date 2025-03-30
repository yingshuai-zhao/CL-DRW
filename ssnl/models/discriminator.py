from .basic_layers import *

class Discriminator(nn.Module):
    """
    鉴别器，鉴别传入图像是否带有水印
    """
    def __init__(self):
        super(Discriminator, self).__init__()

        layers = [ConvBNRelu(3, 64)]
        layers.append(DoubleConvBNRelu(64, 64))
        
        layers.append(nn.AdaptiveAvgPool2d(output_size=(1, 1)))

        self.layers = nn.Sequential(*layers)
        self.linear = nn.Linear(64, 1)


    def forward(self, img):
        # (b, c, 1, 1)
        x = self.layers(img)
        # (b, c)
        x.squeeze_(-1).squeeze_(-1)
        x = self.linear(x)
        return x
    