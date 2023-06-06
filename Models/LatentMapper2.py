import torch.nn as nn

from Models.UtilModels.encoders.helpers import l2_norm


class LatentMapper(nn.Module):
    def __init__(self):
        super().__init__()
        slope = 0.2
        self.model = nn.Sequential(
            nn.Linear(2560, 2048),
            nn.BatchNorm1d(2048,  affine=True),
            nn.ReLU(inplace=True),

            nn.Linear(2048,1024),
            nn.BatchNorm1d(1024,  affine=True),
            nn.ReLU(inplace=True),

            nn.Linear(1024, 512),
            nn.BatchNorm1d(512,  affine=True),
            nn.ReLU(inplace=True),

            nn.Linear(512, 512),
            # nn.BatchNorm1d(512, affine=True),
        )
        for m in self.model:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, a= slope)
                nn.init.constant_(m.bias, 0)

    def forward(self, input_tensor):
        return self.model(input_tensor)
