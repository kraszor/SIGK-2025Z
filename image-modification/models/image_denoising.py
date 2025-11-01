import torch
import torch.nn as nn
from typing import Optional


class DnCNN(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = None,
        num_layers: int = 17,
        num_features: int = 64,
        kernel_size: int = 3,
        use_bnorm: bool = True,
    ):
        super(DnCNN, self).__init__()
        
        if out_channels is None:
            out_channels = in_channels
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.num_features = num_features
        self.kernel_size = kernel_size
        self.use_bnorm = use_bnorm
        
        padding = kernel_size // 2
        layers = []
        layers.extend([
            nn.Conv2d(in_channels, num_features, kernel_size=kernel_size, padding=padding, bias=False),
            nn.ReLU(inplace=True)
        ])
        for _ in range(num_layers - 2):
            layers.append(nn.Conv2d(num_features, num_features, kernel_size=kernel_size, padding=padding, bias=False))
            if use_bnorm:
                layers.append(nn.BatchNorm2d(num_features))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(num_features, out_channels, kernel_size=kernel_size, padding=padding, bias=False))
        
        self.dncnn = nn.Sequential(*layers)
    
    def forward(self, x):
        residual = self.dncnn(x)
        output = x - residual
        
        return output


class UNet(nn.Module):
    
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = None,
        init_features: int = 64,
        use_bnorm: bool = True,
    ):
        super(UNet, self).__init__()
        
        if out_channels is None:
            out_channels = in_channels
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.init_features = init_features
        self.use_bnorm = use_bnorm
        
        features = init_features
        
        self.encoder1 = self._make_layer(in_channels, features, use_bnorm)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.encoder2 = self._make_layer(features, features * 2, use_bnorm)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.encoder3 = self._make_layer(features * 2, features * 4, use_bnorm)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.encoder4 = self._make_layer(features * 4, features * 8, use_bnorm)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.bottleneck = self._make_layer(features * 8, features * 16, use_bnorm)
        
        self.upconv4 = nn.ConvTranspose2d(features * 16, features * 8, kernel_size=2, stride=2)
        self.decoder4 = self._make_layer(features * 16, features * 8, use_bnorm)
        
        self.upconv3 = nn.ConvTranspose2d(features * 8, features * 4, kernel_size=2, stride=2)
        self.decoder3 = self._make_layer(features * 8, features * 4, use_bnorm)
        
        self.upconv2 = nn.ConvTranspose2d(features * 4, features * 2, kernel_size=2, stride=2)
        self.decoder2 = self._make_layer(features * 4, features * 2, use_bnorm)
        
        self.upconv1 = nn.ConvTranspose2d(features * 2, features, kernel_size=2, stride=2)
        self.decoder1 = self._make_layer(features * 2, features, use_bnorm)
        
        self.conv_out = nn.Conv2d(features, out_channels, kernel_size=1)
    
    def _make_layer(self, in_channels: int, out_channels: int, use_bnorm: bool):
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=not use_bnorm),
            nn.BatchNorm2d(out_channels) if use_bnorm else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=not use_bnorm),
            nn.BatchNorm2d(out_channels) if use_bnorm else nn.Identity(),
            nn.ReLU(inplace=True)
        ]
        return nn.Sequential(*layers)
    
    def forward(self, x):

        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))
        
        bottleneck = self.bottleneck(self.pool4(enc4))
        
        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat([dec4, enc4], dim=1)
        dec4 = self.decoder4(dec4)
        
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat([dec3, enc3], dim=1)
        dec3 = self.decoder3(dec3)
        
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat([dec2, enc2], dim=1)
        dec2 = self.decoder2(dec2)
        
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat([dec1, enc1], dim=1)
        dec1 = self.decoder1(dec1)
        
        output = self.conv_out(dec1)
        
        return output
