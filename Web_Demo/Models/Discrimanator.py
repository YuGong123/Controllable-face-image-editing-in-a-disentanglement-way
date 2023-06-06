import torch.nn as nn


# class Discriminator(nn.Module):
#     def __init__(self):
#         super(Discriminator, self).__init__()
#         slope = 0.2
#         self.model = nn.Sequential(
#             nn.Linear(512, 256),
#             nn.LeakyReLU(negative_slope=slope),
#             nn.Linear(256, 128),
#             nn.LeakyReLU(negative_slope=slope),
#             nn.Linear(128, 64),
#             nn.LeakyReLU(negative_slope=slope),
#             nn.Linear(64, 1)
#         )
#         for m in self.model:
#             if isinstance(m, nn.Linear):
#                 nn.init.kaiming_normal_(m.weight, a= slope)
#                 nn.init.constant_(m.bias, 0)
#
#     def forward(self, input_data):
#         return self.model(input_data)


class Discriminator(nn.Module):
    def __init__(self, in_channels=3):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, normalization=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalization:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(in_channels, 64, normalization=False),  # 32
            *discriminator_block(64, 128),      # 16
            *discriminator_block(128, 256),     # 8
            *discriminator_block(256, 512, normalization=False),    # 4
            *discriminator_block(512, 1, normalization=False),    # 2
            nn.Conv2d(1, 1, 4, stride=1, padding=1, bias=False)   # 1
        )


    def forward(self, img_input):
        # Concatenate image and condition image by channels to produce input
        return self.model(img_input)

