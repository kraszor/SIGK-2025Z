"""
Loss functions for image denoising
Includes Perceptual Loss and Charbonnier Loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg16
from typing import Optional


class CharbonnierLoss(nn.Module):

    def __init__(self, epsilon: float = 1e-3, reduction: str = 'mean'):
        super(CharbonnierLoss, self).__init__()
        self.epsilon = epsilon
        self.reduction = reduction
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        diff = pred - target
        loss = torch.sqrt(diff * diff + self.epsilon * self.epsilon)

        return torch.mean(loss)


class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        vgg = vgg16(pretrained=True).features[:16]
        for param in vgg.parameters():
            param.requires_grad = False
        self.vgg = vgg.eval()

    def forward(self, denoised, gt):
        denoised_features = self.vgg(denoised)
        gt_features = self.vgg(gt)
        loss = F.mse_loss(denoised_features, gt_features)
        return loss


class CombinedLoss(nn.Module):

    def __init__(
        self,
        charbonnier_loss: nn.Module,
        perceptual_loss: Optional[nn.Module] = None,
        mse_weight: float = 1.0,
        charbonnier_weight: float = 1.0,
        perceptual_weight: float = 0.1
    ):
        super(CombinedLoss, self).__init__()
        self.charbonnier_loss = charbonnier_loss
        self.perceptual_loss = perceptual_loss
        self.charbonnier_weight = charbonnier_weight
        self.perceptual_weight = perceptual_weight
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        mse_loss = nn.MSELoss()(pred, target)
        loss = self.charbonnier_weight * self.charbonnier_loss(pred, target)
        loss += mse_loss * self.mse_weight
        if self.perceptual_loss is not None:
            loss += self.perceptual_weight * self.perceptual_loss(pred, target)
        
        return loss


def get_charbonnier_loss() -> CharbonnierLoss:
    return CharbonnierLoss()


def get_perceptual_loss(
) -> PerceptualLoss:
    return PerceptualLoss()


def get_combined_loss(
    pixel_weight: float = 1.0,
    perceptual_weight: float = 0.1,
) -> CombinedLoss:
    pixel_loss = get_charbonnier_loss()
    perceptual_loss = get_perceptual_loss()
    return CombinedLoss(pixel_loss, perceptual_loss, pixel_weight, perceptual_weight)
