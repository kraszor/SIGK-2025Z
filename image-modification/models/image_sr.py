# ———— MODEL: SRCNN ————
import torch
import torch.nn as nn
import torch.nn.functional as F

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


class SRUNet(nn.Module):
    
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        init_features: int = 64,
        use_bnorm: bool = True,
        input_size: int = 32
    ):
        upscale_factor = 256 // input_size
        super(SRUNet, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.init_features = init_features
        self.upscale_factor = upscale_factor
        self.use_bnorm = use_bnorm
        
        features = init_features
        
        self.encoder1 = self._make_conv_block(in_channels, features)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.encoder2 = self._make_conv_block(features, features * 2)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        self.encoder3 = self._make_conv_block(features * 2, features * 4)
        self.pool3 = nn.MaxPool2d(2, 2)
        
        self.bottleneck = self._make_conv_block(features * 4, features * 8)
        
        self.upconv3 = nn.ConvTranspose2d(features * 8, features * 4, kernel_size=2, stride=2)
        self.decoder3 = self._make_conv_block(features * 8, features * 4)
        
        self.upconv2 = nn.ConvTranspose2d(features * 4, features * 2, kernel_size=2, stride=2)
        self.decoder2 = self._make_conv_block(features * 4, features * 2)
        
        self.upconv1 = nn.ConvTranspose2d(features * 2, features, kernel_size=2, stride=2)
        self.decoder1 = self._make_conv_block(features * 2, features)
        
        self.sr_layers = self._make_sr_layers(features, upscale_factor)
        
        self.final_conv = nn.Conv2d(features, out_channels, kernel_size=3, padding=1)
        

    def _make_conv_block(self, in_channels: int, out_channels: int):
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=not self.use_bnorm),
            nn.BatchNorm2d(out_channels) if self.use_bnorm else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=not self.use_bnorm),
            nn.BatchNorm2d(out_channels) if self.use_bnorm else nn.Identity(),
            nn.ReLU(inplace=True)
        ]
        return nn.Sequential(*layers)
    
    def _make_sr_layers(self, features: int, upscale_factor: int):
        layers = []
        
        num_upsample = 0
        factor = upscale_factor
        while factor > 1:
            if factor % 2 == 0:
                num_upsample += 1
                factor //= 2
            else:
                raise ValueError(f"Upscale factor {upscale_factor} must be a power of 2")
        
        for i in range(num_upsample):
            layers.extend([
                nn.ConvTranspose2d(features, features, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(features) if self.use_bnorm else nn.Identity(),
                nn.ReLU(inplace=True),
                nn.Conv2d(features, features, kernel_size=3, padding=1),
                nn.BatchNorm2d(features) if self.use_bnorm else nn.Identity(),
                nn.ReLU(inplace=True)
            ])
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        
        bottleneck = self.bottleneck(self.pool3(enc3))
        
        dec3 = self.upconv3(bottleneck)
        dec3 = torch.cat([dec3, enc3], dim=1)
        dec3 = self.decoder3(dec3)
        
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat([dec2, enc2], dim=1)
        dec2 = self.decoder2(dec2)
        
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat([dec1, enc1], dim=1)
        dec1 = self.decoder1(dec1)
        
        sr_output = self.sr_layers(dec1)
        
        output = self.final_conv(sr_output)
        
        return output