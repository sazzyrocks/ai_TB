import torch
import torch.nn as nn
from models.densenet import DenseNet121Backbone
from models.convnext import ConvNeXtV2Backbone
from models.attention import BidirectionalCrossAttention

class TBHybridModel(nn.Module):
    """
    Tuberculosis Detection Hybrid Model combining DenseNet121 and ConvNeXt V2 features
    using 1x1 Conv projections, bidirectional cross-attention, adaptive gated fusion,
    and an MLP classification head.
    """
    def __init__(
        self,
        pretrained: bool = True,
        freeze_early: bool = True,
        projection_dim: int = 512,
        num_heads: int = 4,
        dropout: float = 0.4,
        convnext_model_name: str = "convnextv2_nano"
    ):
        super().__init__()
        
        # 1. Backbone Feature Extractors
        self.densenet = DenseNet121Backbone(pretrained=pretrained, freeze_early=freeze_early)
        self.convnext = ConvNeXtV2Backbone(model_name=convnext_model_name, pretrained=pretrained, freeze_early=freeze_early)
        
        # 2. 1x1 Conv Projection Layers (Channel Alignment)
        self.proj_densenet = nn.Conv2d(self.densenet.num_features, projection_dim, kernel_size=1)
        self.proj_convnext = nn.Conv2d(self.convnext.num_features, projection_dim, kernel_size=1)
        
        # 3. Bidirectional Cross-Attention Layer
        self.cross_attention = BidirectionalCrossAttention(
            embed_dim=projection_dim,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # 4. Learnable Adaptive Gating Parameter (initialized to zero)
        self.gate = nn.Parameter(torch.zeros(1, 1, projection_dim))
        
        # 5. Post-Fusion Processing
        self.ln = nn.LayerNorm(projection_dim)
        
        # 6. Classification Head (MLP)
        self.classifier = nn.Sequential(
            nn.Linear(projection_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 2)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Extract backbone features
        # f_dense: (B, 1024, 7, 7)
        # f_convnext: (B, 640, 7, 7)
        f_dense = self.densenet(x)
        f_convnext = self.convnext(x)
        
        # Channel alignment projections
        # both shapes: (B, projection_dim, 7, 7)
        f_dense_proj = self.proj_densenet(f_dense)
        f_convnext_proj = self.proj_convnext(f_convnext)
        
        # Spatial Tokenization: (B, C, H, W) -> (B, H*W, C)
        # flattening spatial dimension (dim 2 and 3)
        f_dense_flat = f_dense_proj.flatten(2).transpose(1, 2)       # (B, 49, 512)
        f_convnext_flat = f_convnext_proj.flatten(2).transpose(1, 2)   # (B, 49, 512)
        
        # Bidirectional Cross-Attention
        f_dense_refined, f_convnext_refined = self.cross_attention(f_dense_flat, f_convnext_flat)
        
        # Adaptive Gated Fusion
        alpha = torch.sigmoid(self.gate)  # shape: (1, 1, 512)
        fused = alpha * f_dense_refined + (1.0 - alpha) * f_convnext_refined  # (B, 49, 512)
        
        # Post-Fusion LayerNorm & Mean Pooling over spatial dimension
        fused = self.ln(fused)
        fused_pooled = fused.mean(dim=1)  # (B, 512)
        
        # Classification MLP
        logits = self.classifier(fused_pooled)  # (B, 2)
        return logits
