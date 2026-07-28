import torch
import torch.nn as nn
import torchvision.models as tv_models

class DenseNet121Backbone(nn.Module):
    """
    Wrapper for DenseNet121 backbone model.
    Extracts deep 2D feature maps before global average pooling.
    """
    def __init__(self, pretrained: bool = True, freeze_early: bool = False):
        super().__init__()
        # Try loading using newer weights API first, fallback to older pretrained argument
        try:
            if hasattr(tv_models, "DenseNet121_Weights"):
                weights = tv_models.DenseNet121_Weights.DEFAULT if pretrained else None
                self.backbone = tv_models.densenet121(weights=weights)
            else:
                self.backbone = tv_models.densenet121(pretrained=pretrained)
        except Exception as e:
            # Fallback in case of network issues or environment quirks
            print(f"Pretrained load failed ({e}), creating random initialized backbone.")
            self.backbone = tv_models.densenet121(pretrained=False)
            
        self.features = self.backbone.features
        self.num_features = 1024
        
        if freeze_early:
            # Freeze features up to transition2 (stem, denseblock1, transition1, denseblock2, transition2)
            freeze_keywords = ["conv0", "norm0", "denseblock1", "transition1", "denseblock2", "transition2"]
            for name, param in self.features.named_parameters():
                if any(kw in name for kw in freeze_keywords):
                    param.requires_grad = False
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input chest X-ray images of shape (B, 3, H, W)
        Returns:
            2D feature maps of shape (B, 1024, H/32, W/32), i.e., (B, 1024, 7, 7) for 224x224 input
        """
        return self.features(x)
