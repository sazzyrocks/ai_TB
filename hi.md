# Tuberculosis (TB) Detection Hybrid Model Architecture

This document describes the design and flow of the hybrid deep learning model used for Tuberculosis detection on Chest X-Rays. The model leverages two strong feature extractors: **DenseNet121** (efficient feature reuse and gradient flow) and **ConvNeXt V2** (modern convolutional architecture with sparse convolutions and GRN).

## Architecture Flow Diagram

```mermaid
graph TD
    %% Define Styles
    classDef input fill:#eceff1,stroke:#37474f,stroke-width:2px;
    classDef backbone fill:#bbdefb,stroke:#1565c0,stroke-width:2px,stroke-dasharray: 5 5;
    classDef projection fill:#ffe0b2,stroke:#f57c00,stroke-width:2px;
    classDef attention fill:#e1bee7,stroke:#6a1b9a,stroke-width:2px;
    classDef fusion fill:#c8e6c9,stroke:#2e7d32,stroke-width:2px;
    classDef head fill:#ffcdd2,stroke:#c62828,stroke-width:2px;

    %% Nodes
    Input["Input CXR Image <br> (B, 3, 224, 224)"]:::input

    %% DenseNet Branch
    DN121["DenseNet-121 Backbone <br> (Pretrained / Partially Frozen)"]:::backbone
    DN_Feat["DenseNet Feature Map <br> (B, 1024, 7, 7)"]
    DN_Proj["1x1 Conv Projection <br> (proj_densenet)"]:::projection
    DN_Proj_Feat["Aligned DenseNet Features <br> (B, 512, 7, 7)"]
    DN_Flat["Flatten & Transpose <br> (B, 49, 512)"]

    %% ConvNeXt Branch
    CNV2["ConvNeXt V2 Backbone <br> (Pretrained / Partially Frozen)"]:::backbone
    CN_Feat["ConvNeXt Feature Map <br> (B, 640, 7, 7)"]
    CN_Proj["1x1 Conv Projection <br> (proj_convnext)"]:::projection
    CN_Proj_Feat["Aligned ConvNeXt Features <br> (B, 512, 7, 7)"]
    CN_Flat["Flatten & Transpose <br> (B, 49, 512)"]

    %% Connections to backbones
    Input --> DN121
    Input --> CNV2

    DN121 --> DN_Feat
    DN_Feat --> DN_Proj
    DN_Proj --> DN_Proj_Feat
    DN_Proj_Feat --> DN_Flat

    CNV2 --> CN_Feat
    CN_Feat --> CN_Proj
    CN_Proj --> CN_Proj_Feat
    CN_Proj_Feat --> CN_Flat

    %% Cross Attention Block
    subgraph Cross_Attention_Mechanism ["Multi-Head Cross-Attention Layer"]
        CA_Dense["Cross-Attention A <br> (DenseNet queries ConvNeXt) <br> Q: Dense, K/V: ConvNeXt"]:::attention
        CA_Conv["Cross-Attention B <br> (ConvNeXt queries DenseNet) <br> Q: ConvNeXt, K/V: Dense"]:::attention
        
        Res_Dense["Residual Addition <br> (Dense + Attn Out)"]
        Res_Conv["Residual Addition <br> (ConvNeXt + Attn Out)"]
    end

    DN_Flat -->|Query| CA_Dense
    CN_Flat -->|Key / Value| CA_Dense
    CA_Dense --> Res_Dense
    DN_Flat -->|Residual Connection| Res_Dense

    CN_Flat -->|Query| CA_Conv
    DN_Flat -->|Key / Value| CA_Conv
    CA_Conv --> Res_Conv
    CN_Flat -->|Residual Connection| Res_Conv

    %% Gated Fusion
    Gate_Sig["Gated Parameter (alpha) <br> alpha = Sigmoid(gate) <br> Shape: (1, 1, 512)"]:::fusion
    Gated_Fusion["Adaptive Gated Fusion <br> F = alpha * Dense_Refined + (1 - alpha) * ConvNeXt_Refined"]:::fusion

    Res_Dense --> Gated_Fusion
    Res_Conv --> Gated_Fusion
    Gate_Sig --> Gated_Fusion

    %% Final layers
    LN["Layer Normalization"]
    GAP["Global Average Pooling <br> (Mean pooling over 49 tokens) <br> Shape: (B, 512)"]
    
    subgraph Classification_MLP ["Classification Head (MLP)"]
        FC1["Linear(512 -> 256)"]:::head
        LN2["LayerNorm(256)"]
        ReLU["ReLU Activation"]
        Dropout["Dropout (0.4)"]
        FC2["Linear(256 -> 2)"]:::head
    end

    Gated_Fusion --> LN
    LN --> GAP
    GAP --> FC1
    FC1 --> LN2
    LN2 --> ReLU
    ReLU --> Dropout
    Dropout --> FC2
    
    FC2 --> Logits["Class Logits <br> (B, 2) <br> [Normal, Tuberculosis]"]:::input
```

---

## Detailed Step-by-Step Architecture Breakdown

### 1. Dual Feature Extraction (Backbones)
We use two complementary backbones to extract spatial features from the input Chest X-ray images of size `(B, 3, 224, 224)`:
* **DenseNet121 Backbone ([densenet.py](file:///d:/TB/Final/models/densenet.py))**: Extracts feature map of shape `(B, 1024, 7, 7)`. DenseNet utilizes dense connections where each layer obtains inputs from all preceding layers, making it highly effective at preserving low-level details.
* **ConvNeXt V2 Backbone ([convnext.py](file:///d:/TB/Final/models/convnext.py))**: Extracts feature map of shape `(B, 640, 7, 7)`. ConvNeXt V2 uses modern convolutional design blocks inspired by Transformers (e.g. depthwise convolutions, LayerNorm, and GELU).

### 2. Channel Alignment (Linear Projections)
Since the feature channel dimensions of the two backbones do not match (`1024` for DenseNet and `640` for ConvNeXt), we project both to a common projection dimension (`projection_dim = 512`):
* `proj_densenet`: A `1x1` 2D Convolution mapping channels from `1024 -> 512`.
* `proj_convnext`: A `1x1` 2D Convolution mapping channels from `640 -> 512`.

### 3. Spatial Tokenization
The projected features are flattened from `(B, 512, 7, 7)` to `(B, 512, 49)` and transposed to `(B, 49, 512)`. This transforms the 2D spatial dimensions into 49 spatial "tokens," matching the sequence-style input expected by PyTorch's Multihead Attention module.

### 4. Bidirectional Cross-Attention
We use two independent Multi-Head Cross-Attention modules (`num_heads = 4`, `dropout = 0.4`) to let the two models share their complementary representations:
* **Cross-Attention A (DenseNet queries ConvNeXt)**: DenseNet features act as Queries ($Q$), while ConvNeXt features act as Keys ($K$) and Values ($V$). This refines DenseNet features by attending to contextual regions highlighted by ConvNeXt.
* **Cross-Attention B (ConvNeXt queries DenseNet)**: ConvNeXt features act as Queries ($Q$), while DenseNet features act as Keys ($K$) and Values ($V$). This refines ConvNeXt features by attending to context highlighted by DenseNet.

*Residual connections* are applied to preserve original branch features:
$$\text{Refined DenseNet} = \text{DenseNet} + \text{CrossAttention}_A(\text{DenseNet}, \text{ConvNeXt})$$
$$\text{Refined ConvNeXt} = \text{ConvNeXt} + \text{CrossAttention}_B(\text{ConvNeXt}, \text{DenseNet})$$

### 5. Learnable Adaptive Gated Fusion
Instead of simple concatenation or addition, we fuse the refined features using a learnable gating parameter (`self.gate` of shape `(1, 1, 512)` initialized to zero):
$$\alpha = \text{Sigmoid}(\text{gate})$$
$$\text{Fused Features} = \alpha \times \text{Refined DenseNet} + (1 - \alpha) \times \text{Refined ConvNeXt}$$

Since $\alpha$ is a vector of size 512, the model learns a **channel-wise gate** that decides how much information to retain from each branch dynamically.

### 6. Pooling & Normalization
* A **Layer Normalization** (`self.ln`) is applied to stabilize feature distributions.
* A **Global Average Pooling** layer averages the 49 spatial tokens along the sequence dimension to generate a single feature vector of shape `(B, 512)`.

### 7. Classification Head (MLP)
The fused feature vector is passed to a Multi-Layer Perceptron (MLP) for final diagnosis:
1. `Linear(512 -> 256)`
2. `LayerNorm(256)`
3. `ReLU` Activation
4. `Dropout(p=0.4)` (regularization to prevent overfitting)
5. `Linear(256 -> 2)` (outputs logits for [Normal, Tuberculosis] classification)
