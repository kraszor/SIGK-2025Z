import torch
import torch.nn as nn


class ToneMappingEncoder(nn.Module):
    def __init__(self):
        super(ToneMappingEncoder, self).__init__()
        self.first_cov = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding="same"), nn.ReLU()
        )
        self.second_cov = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding="same"), nn.ReLU()
        )
        self.third_cov = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding="same"), nn.ReLU()
        )

    def forward(self, x):
        y = self.first_cov(x)
        y = self.second_cov(y)
        y = self.third_cov(y)
        return y


class ToneMappingDecoder(nn.Module):
    def __init__(self):
        super(ToneMappingDecoder, self).__init__()
        self.first_deconv = nn.Sequential(
            nn.ConvTranspose2d(192, 32, kernel_size=3, stride=1, padding="same"),
            nn.ReLU(),
        )
        self.second_deconv = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=1, padding="same"),
            nn.ReLU(),
        )
        self.final_cov = nn.Sequential(
            nn.Conv2d(16, 3, kernel_size=3, stride=1, padding="same"), nn.Sigmoid()
        )

    def forward(self, x, e_low, e_mid, e_high):
        y = self.first_deconv(x)
        y = self.second_deconv(y)
        y = self.final_cov(y)
        return y + e_low + e_mid + e_high


class ToneMappingNetwork(nn.Module):
    def __init__(self):
        super(ToneMappingNetwork, self).__init__()
        self.encoder = ToneMappingEncoder()
        self.fusion_module_1 = nn.Conv2d(
            in_channels=64, out_channels=192, kernel_size=3, stride=1, padding="same"
        )
        self.fusion_module_2 = nn.Conv2d(
            in_channels=192, out_channels=192, kernel_size=1, stride=1, padding="same"
        )
        self.decoder = ToneMappingDecoder()

    def forward(self, e_1, e_2, e_3):
        e_1 = self.encoder(e_1)
        e_2 = self.encoder(e_2)
        e_3 = self.encoder(e_3)
        combined_encoding = torch.cat((e_1, e_2, e_3), dim=1)
        fuse_1 = self.fusion_module_1(combined_encoding)
        fuse_2 = self.fusion_module_2(fuse_1)
        decoded = self.decoder(fuse_2, e_1, e_2, e_3)
        return decoded
