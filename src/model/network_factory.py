"""Model factory for 2D CXR BMD estimation.

Two stages are supported:
1. MAE (masked autoencoder) self-supervised pretraining. The chosen backbone is
   used as a spatial encoder and a lightweight convolutional decoder reconstructs
   the original image from the encoded features. Only the masked patches
   contribute to the reconstruction loss (MSE + L1).
2. Supervised finetuning. The same backbone is reused (encoder weights loaded
   from the MAE checkpoint), followed by global average pooling and two heads:
   a classification head (3-class or binary) and an optional regression head
   (continuous BMD / T-score).
"""

from typing import Tuple

import torch
import torch.nn as nn
from torchvision.models import (
    convnext_base,
    densenet121,
    resnet50,
    swin_t,
    vit_b_16,
)

SUPPORTED_MODELS = ("resnet", "densenet", "convnext", "vit", "swin")


def _weights(pretrained: bool):
    return "IMAGENET1K_V1" if pretrained else None


class _ViTSpatialEncoder(nn.Module):
    """Wrap ``vit_b_16`` so that it outputs a (B, C, H, W) feature map.

    The class token is dropped and the remaining patch tokens are reshaped back
    onto the 2D grid so the same convolutional decoder can be reused.
    """

    def __init__(self, vit: nn.Module) -> None:
        super().__init__()
        self.vit = vit
        self.hidden_dim = vit.hidden_dim
        self.patch_size = vit.patch_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n, _, h, w = x.shape
        grid_h = h // self.patch_size
        grid_w = w // self.patch_size
        tokens = self.vit._process_input(x)
        batch_class_token = self.vit.class_token.expand(n, -1, -1)
        tokens = torch.cat([batch_class_token, tokens], dim=1)
        tokens = self.vit.encoder(tokens)
        tokens = tokens[:, 1:, :]  # drop class token
        feat = tokens.transpose(1, 2).reshape(n, self.hidden_dim, grid_h, grid_w)
        return feat


class _SwinSpatialEncoder(nn.Module):
    """Wrap ``swin_t`` so that it outputs a (B, C, H, W) feature map."""

    def __init__(self, swin: nn.Module) -> None:
        super().__init__()
        self.features = swin.features
        self.norm = swin.norm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.features(x)  # (B, H, W, C)
        feat = self.norm(feat)
        feat = feat.permute(0, 3, 1, 2).contiguous()  # (B, C, H, W)
        return feat


def create_encoder(model_name: str, pretrained: bool = True) -> Tuple[nn.Module, int, int]:
    """Build a spatial encoder producing a (B, C, H, W) feature map.

    Args:
        model_name: one of ``SUPPORTED_MODELS``.
        pretrained: whether to load ImageNet weights.

    Returns:
        (encoder, feat_channels, feat_size) where ``feat_size`` is the spatial
        side length of the feature map for a 224x224 input.
    """
    name = model_name.lower()
    if name == "resnet":
        model = resnet50(weights=_weights(pretrained))
        encoder = nn.Sequential(*list(model.children())[:-2])
        return encoder, 2048, 7
    if name == "densenet":
        model = densenet121(weights=_weights(pretrained))
        encoder = model.features
        return encoder, 1024, 7
    if name == "convnext":
        model = convnext_base(weights=_weights(pretrained))
        encoder = model.features
        return encoder, 1024, 7
    if name == "vit":
        model = vit_b_16(weights=_weights(pretrained))
        return _ViTSpatialEncoder(model), model.hidden_dim, 14
    if name == "swin":
        model = swin_t(weights=_weights(pretrained))
        return _SwinSpatialEncoder(model), 768, 7
    raise ValueError(
        f"Invalid model name '{model_name}'. Choose from {SUPPORTED_MODELS}."
    )


class ConvDecoder(nn.Module):
    """Upsample a (B, C, feat_size, feat_size) feature map to (B, 3, 224, 224)."""

    def __init__(self, in_channels: int, feat_size: int, out_size: int = 224, out_channels: int = 3) -> None:
        super().__init__()
        num_upsamples = 0
        size = feat_size
        while size < out_size:
            size *= 2
            num_upsamples += 1

        layers = []
        channels = in_channels
        for i in range(num_upsamples):
            out_c = max(channels // 2, 32)
            layers += [
                nn.ConvTranspose2d(channels, out_c, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(out_c),
                nn.GELU(),
            ]
            channels = out_c
        self.upsample = nn.Sequential(*layers)
        self.out_size = out_size
        self.head = nn.Conv2d(channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        if x.shape[-1] != self.out_size:
            x = nn.functional.interpolate(
                x, size=(self.out_size, self.out_size), mode="bilinear", align_corners=False
            )
        return self.head(x)


def _random_patch_mask(
    batch_size: int,
    image_size: int,
    patch_size: int,
    mask_ratio: float,
    device: torch.device,
) -> torch.Tensor:
    """Return a per-pixel binary mask (B, 1, H, W); 1 = masked (hidden)."""
    grid = image_size // patch_size
    num_patches = grid * grid
    num_mask = int(round(mask_ratio * num_patches))

    noise = torch.rand(batch_size, num_patches, device=device)
    ids_shuffle = torch.argsort(noise, dim=1)
    mask_flat = torch.zeros(batch_size, num_patches, device=device)
    mask_flat.scatter_(1, ids_shuffle[:, :num_mask], 1.0)

    mask = mask_flat.view(batch_size, 1, grid, grid)
    mask = mask.repeat_interleave(patch_size, dim=2).repeat_interleave(patch_size, dim=3)
    return mask


class MaskedAutoEncoder(nn.Module):
    """Generic masked convolutional autoencoder for self-supervised pretraining."""

    def __init__(
        self,
        model_name: str,
        pretrained: bool = True,
        image_size: int = 224,
        patch_size: int = 16,
        mask_ratio: float = 0.75,
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.image_size = image_size
        self.patch_size = patch_size
        self.mask_ratio = mask_ratio

        self.encoder, feat_channels, feat_size = create_encoder(model_name, pretrained)
        self.decoder = ConvDecoder(feat_channels, feat_size, out_size=image_size, out_channels=3)

    def _reconstruct(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mask = _random_patch_mask(
            batch_size=x.shape[0],
            image_size=self.image_size,
            patch_size=self.patch_size,
            mask_ratio=self.mask_ratio,
            device=x.device,
        )
        masked_input = x * (1.0 - mask)
        feat = self.encoder(masked_input)
        recon = self.decoder(feat)
        return recon, mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        recon, mask = self._reconstruct(x)
        diff_sq = (recon - x).pow(2)
        diff_abs = (recon - x).abs()
        denom = mask.sum().clamp_min(1.0) * x.shape[1]
        loss = (diff_sq * mask).sum() / denom + (diff_abs * mask).sum() / denom
        return loss

    def export_encoder_state(self) -> dict:
        return self.encoder.state_dict()


class FinetuneModel(nn.Module):
    """Backbone encoder + global pooling + classification/regression heads."""

    def __init__(
        self,
        model_name: str,
        num_classes: int,
        regression: bool = False,
        pretrained: bool = True,
        num_regression_outputs: int = 1,
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.regression = regression

        self.encoder, self.num_features, _ = create_encoder(model_name, pretrained)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classification_head = nn.Linear(self.num_features, num_classes)
        self.regression_head = (
            nn.Linear(self.num_features, num_regression_outputs) if regression else None
        )

    def load_encoder_state(self, state: dict, strict: bool = True) -> None:
        self.encoder.load_state_dict(state, strict=strict)

    def forward(self, x: torch.Tensor):
        feat = self.encoder(x)
        feat = self.pool(feat).flatten(1)
        logits = self.classification_head(feat)
        if self.regression and self.regression_head is not None:
            reg_out = self.regression_head(feat)
            return logits, reg_out
        return logits
