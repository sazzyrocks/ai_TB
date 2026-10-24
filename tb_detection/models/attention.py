import torch
import torch.nn as nn

class BidirectionalCrossAttention(nn.Module):
    """
    Bidirectional cross-attention module that processes flattened tokens
    from two branches (DenseNet and ConvNeXt) to learn collaborative features.
    """
    def __init__(self, embed_dim: int = 512, num_heads: int = 4, dropout: float = 0.4):
        super().__init__()
        
        # Cross-Attention A: DenseNet queries ConvNeXt (Key/Value)
        self.cross_attn_dense = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Cross-Attention B: ConvNeXt queries DenseNet (Key/Value)
        self.cross_attn_convnext = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
    def forward(self, f_dense_flat: torch.Tensor, f_convnext_flat: torch.Tensor):
        """
        Args:
            f_dense_flat: Projected DenseNet tokens of shape (B, N, C), i.e., (B, 49, 512)
            f_convnext_flat: Projected ConvNeXt tokens of shape (B, N, C), i.e., (B, 49, 512)
        Returns:
            f_dense_refined: Refined DenseNet features of shape (B, N, C)
            f_convnext_refined: Refined ConvNeXt features of shape (B, N, C)
        """
        # Cross-Attention A: Q=DenseNet, K/V=ConvNeXt
        attn_dense_out, _ = self.cross_attn_dense(
            query=f_dense_flat,
            key=f_convnext_flat,
            value=f_convnext_flat
        )
        # Residual Connection
        f_dense_refined = f_dense_flat + attn_dense_out
        
        # Cross-Attention B: Q=ConvNeXt, K/V=DenseNet
        attn_convnext_out, _ = self.cross_attn_convnext(
            query=f_convnext_flat,
            key=f_dense_flat,
            value=f_dense_flat
        )
        # Residual Connection
        f_convnext_refined = f_convnext_flat + attn_convnext_out
        
        return f_dense_refined, f_convnext_refined
