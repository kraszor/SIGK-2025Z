# ———— MODEL: SRCNN ————
import torch.nn as nn

class SRCNN(nn.Module):

    def __init__(
        self,
        in_channels: int = 3,
        feat1: int = 64,
        feat2: int = 32,
        k1: int = 9,
        k2: int = 1,
        k3: int = 5,
        out_channels: int = 3,
    ):
        super().__init__()
        padding1 = k1 // 2
        padding2 = k2 // 2
        padding3 = k3 // 2

        self.conv1 = nn.Conv2d(in_channels, feat1, kernel_size=k1, padding=padding1)
        self.conv2 = nn.Conv2d(feat1, feat2, kernel_size=k2, padding=padding2)
        self.conv3 = nn.Conv2d(feat2, out_channels, kernel_size=k3, padding=padding3)

        self.relu = nn.ReLU(inplace=True)


    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        return x
    


class VDSR(nn.Module):
    def __init__(self, in_channels: int = 3, num_layers: int = 20, num_features: int = 64):
        super().__init__()
        
        self.input_conv = nn.Conv2d(in_channels, num_features, kernel_size=3, padding=1)
        
        middle_layers = []
        for _ in range(num_layers - 2):
            middle_layers.append(nn.Conv2d(num_features, num_features, kernel_size=3, padding=1))
            middle_layers.append(nn.ReLU(inplace=True))
        self.middle = nn.Sequential(*middle_layers)
        
        self.output_conv = nn.Conv2d(num_features, in_channels, kernel_size=3, padding=1)
        
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.relu(self.input_conv(x))
        out = self.middle(out)
        out = self.output_conv(out)
        out = out + x
        return out
