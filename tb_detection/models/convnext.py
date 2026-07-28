import torch
import torch.nn as nn
import timm

class ConvNeXtV2Backbone(nn.Module):
    """
    Wrapper for ConvNeXt V2 backbone model.
    Extracts deep 2D feature maps before global average pooling.
    """
    def __init__(self, model_name: str = "convnextv2_nano", pretrained: bool = True, freeze_early: bool = False):
        super().__init__()
        try:
            self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        except Exception as e:
            print(f"Pretrained load failed for timm {model_name} ({e}), loading randomly initialized backbone.")
            self.backbone = timm.create_model(model_name, pretrained=False, num_classes=0)
            
        self.num_features = self.backbone.num_features  # 640 for nano, 768 for tiny
        
        if freeze_early:
            # Freeze stem, stage 0, and stage 1
            freeze_keywords = ["stem", "stages.0", "stages.1"]
            for name, param in self.backbone.named_parameters():
                if any(kw in name for kw in freeze_keywords):
                    param.requires_grad = False
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input chest X-ray images of shape (B, 3, H, W)
        Returns:
            2D feature maps of shape (B, C_convnext, H/32, W/32), i.e., (B, 640, 7, 7) for 224x224 input
        """
        return self.backbone.forward_features(x)
