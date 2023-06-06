import torch
from torch import nn
from torch.nn import Linear, Conv2d, BatchNorm1d, BatchNorm2d, PReLU, Sequential, Module, Sigmoid, LeakyReLU, Dropout
from Models.StyleGan2.model import EqualLinear
from Models.UtilModels.encoders.helpers import get_blocks, bottleneck_IR, bottleneck_IR_SE, Flatten, l2_norm


class Encoder_Swap(nn.Module):
    def __init__(self, num_layers=50, mode='ir_se', drop_ratio=0.2, affine=True):
        super(Encoder_Swap, self).__init__()
        print('Loading ResNet ArcFace')

        assert num_layers in [50, 100, 152], "num_layers should be 50, 100 or 152"
        assert mode in ['ir', 'ir_se'], "mode should be ir or ir_se"
        blocks = get_blocks(num_layers)
        if mode == 'ir':
            unit_module = bottleneck_IR
        elif mode == 'ir_se':
            unit_module = bottleneck_IR_SE
        self.input_layer = Sequential(Conv2d(3, 64, (3, 3), 1, 1, bias=False),
                                      BatchNorm2d(64),
                                      PReLU(64))

        self.output_layer = Sequential(BatchNorm2d(512),
                                       Conv2d(512, 512, 3, 1, 1, bias=False),  # 512*16*16, 1*1卷积
                                       LeakyReLU(0.2, inplace=True),
                                       Conv2d(512, 32, 3, 1, 1, bias=False),  # 512*16*16, 1*1卷积
                                       BatchNorm2d(32),
                                       LeakyReLU(0.2, inplace=True),
                                       Flatten(),
                                       Linear(32 * 16 * 16, 512),
                                       BatchNorm1d(512, affine=affine),
                                       # LeakyReLU(0.2, inplace=True),
                                       )

        self.output_layer2 = Sequential(BatchNorm2d(512),                         # 512*16*16,
                                        Conv2d(512, 1024, 4, 2, 1, bias=False),   # 1024*8*8, 1*1卷积
                                        LeakyReLU(0.2, inplace=True),
                                        Conv2d(1024, 2048, 4, 2, 1, bias=False),  # 2048*4*4, 1*1卷积
                                        BatchNorm2d(2048),
                                        LeakyReLU(0.2, inplace=True),
                                        nn.AdaptiveAvgPool2d(1),
                                        Conv2d(2048, 2048, 1, 1, 0, bias=False),  # 2048*1*1, 1*1卷积
                                        BatchNorm2d(2048),
                                        # LeakyReLU(0.2, inplace=True),
                                        Flatten(),)

        modules = []
        for block in blocks:
            for bottleneck in block:
                modules.append(unit_module(bottleneck.in_channel,
                                           bottleneck.depth,
                                           bottleneck.stride))
        self.body = Sequential(*modules)  # 512*16*16

    def forward(self, x):
        x = self.input_layer(x)
        x = self.body(x)

        identity = self.output_layer(x)
        attribute = self.output_layer2(x)
        return l2_norm(identity), l2_norm(attribute)


if __name__ == '__main__':
    swap_encoder = Encoder_Swap()
    print(swap_encoder)
