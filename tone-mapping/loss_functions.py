import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class ToneMappingVGG(nn.Module):
    def __init__(self, target_layers: list[str] = ["0", "1", "2"]):
        super().__init__()
        self.target_layers = target_layers

        vgg_model = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features
        self.model = vgg_model.eval()
        self._freeze_parameters()

        self.model_out: dict[str, torch.Tensor] = {}
        self._register_hooks()

    def _freeze_parameters(self) -> None:
        for parameter in self.model.parameters():
            parameter.requires_grad = False

    def _register_hooks(self) -> None:
        layer_dict = dict(self.model._modules)
        for layer_name in self.target_layers:
            layer = layer_dict[layer_name]
            layer.register_forward_hook(self._create_hook(layer_name))

    def _create_hook(self, layer_name: str) -> callable:
        def hook(module: nn.Module, input: torch.Tensor, output: torch.Tensor) -> None:
            self.model_out[layer_name] = output

        return hook

    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        _ = self.model(image)
        return {name: self.model_out[name] for name in self.target_layers}


class ToneMappingFeatureConstrastMaskingLoss(nn.Module):
    def __init__(
        self,
        patch_size: int = 13,
        eps: float = 1e-6,
        alpha_hdr: float = 0.5,
        alpha_tm: float = 1.0,
    ):
        super().__init__()
        self.eps = eps
        self.alpha_hdr = alpha_hdr
        self.alpha_tm = alpha_tm
        self.avg_pool = nn.AvgPool2d(
            kernel_size=patch_size,
            stride=1,
            padding=patch_size // 2,
            count_include_pad=False,
        )

    def _calculate_feature_contrast(self, f_p: torch.Tensor):
        mu_b = self.avg_pool(f_p)
        C_p = (f_p - mu_b) / (torch.abs(mu_b) + self.eps)
        var_b = self.avg_pool(torch.pow(f_p - mu_b, 2))
        sigma_b = torch.sqrt(var_b.clamp(min=0) + self.eps)

        return C_p, mu_b, sigma_b

    def _calculate_fcm_masking(
        self, C_p: torch.Tensor, mu_b: torch.Tensor, sigma_b: torch.Tensor, alpha: float
    ):
        sign_C = C_p / (torch.abs(C_p) + self.eps)
        M_s = sign_C * torch.pow(torch.abs(C_p), alpha)
        M_n = sigma_b / (torch.abs(mu_b) + self.eps)
        f_VGG_I = M_s / (1 + M_n)

        return f_VGG_I

    def forward(
        self,
        features_hdr: dict[str, torch.Tensor],
        features_tm: dict[str, torch.Tensor],
    ):
        loss = 0.0
        for layer_name in features_hdr.keys():
            f_hdr = features_hdr[layer_name]
            f_tm = features_tm[layer_name]
            C_hdr, mu_b_hdr, sigma_b_hdr = self._calculate_feature_contrast(f_hdr)
            f_VGG_hdr = self._calculate_fcm_masking(
                C_hdr, mu_b_hdr, sigma_b_hdr, self.alpha_hdr
            )
            C_tm, mu_b_tm, sigma_b_tm = self._calculate_feature_contrast(f_tm)
            f_VGG_tm = self._calculate_fcm_masking(
                C_tm, mu_b_tm, sigma_b_tm, self.alpha_tm
            )
            layer_loss = F.l1_loss(f_VGG_tm, f_VGG_hdr)
            loss += layer_loss

        return loss
