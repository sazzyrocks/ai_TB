import os
import argparse
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from config import Config
from data.dataset import get_val_transforms
from models.hybrid_model import TBHybridModel

class GradCAM:
    """
    Grad-CAM class to generate activation saliency heatmaps for interpretability.
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.handlers = []
        
        def save_gradient(module, grad_input, grad_output):
            self.gradients = grad_output[0]
            
        def save_activation(module, input, output):
            self.activations = output
            
        self.handlers.append(target_layer.register_forward_hook(save_activation))
        self.handlers.append(target_layer.register_full_backward_hook(save_gradient))
        
    def generate_heatmap(self, input_tensor, class_idx):
        self.model.zero_grad()
        output = self.model(input_tensor)
        score = output[0, class_idx]
        score.backward()
        
        gradients = self.gradients.cpu().data.numpy()[0]
        activations = self.activations.cpu().data.numpy()[0]
        
        # Global average pool the gradients
        weights = np.mean(gradients, axis=(1, 2))
        heatmap = np.zeros(activations.shape[1:], dtype=np.float32)
        
        # Weighted combination of activation maps
        for i, w in enumerate(weights):
            heatmap += w * activations[i]
            
        heatmap = np.maximum(heatmap, 0)  # ReLU
        if np.max(heatmap) > 0:
            heatmap /= np.max(heatmap)  # Normalize
            
        return heatmap
        
    def remove_hooks(self):
        for h in self.handlers:
            h.remove()

def overlay_heatmap(img_path, heatmap, alpha=0.4):
    """
    Overlays Grad-CAM heatmap on the original chest X-ray image.
    """
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Resize heatmap to match image size
    heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    
    # Convert heatmap to RGB coloring (apply COLORMAP_JET)
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    
    # Overlay image and heatmap
    blended = cv2.addWeighted(img, 1 - alpha, heatmap_colored, alpha, 0)
    return img, blended

def run_inference(image_path: str, checkpoint_path: str, visualize: bool):
    """
    Runs model inference on a single image, printing predictions and displaying Grad-CAM.
    """
    image_path = os.path.abspath(image_path)
    checkpoint_path = os.path.abspath(checkpoint_path)
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Prepare image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Apply standard validation transformations
    transform = get_val_transforms()
    augmented = transform(image=image)
    input_tensor = augmented['image'].unsqueeze(0).to(device)  # (1, 3, 224, 224)
    
    # 2. Instantiate and load model
    model = TBHybridModel(
        pretrained=False,
        freeze_early=False,
        projection_dim=Config.PROJECTION_DIM,
        num_heads=Config.NUM_HEADS,
        dropout=Config.DROPOUT,
        convnext_model_name=Config.CONVNEXT_MODEL_NAME
    ).to(device)
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    # Enable gradient tracking if visualization is required
    gradcam = None
    if visualize:
        input_tensor.requires_grad = True
        # Hook target projection layer (proj_densenet is a Conv2d mapping DenseNet features)
        gradcam = GradCAM(model, model.proj_densenet)
        
    # 3. Model Prediction
    outputs = model(input_tensor)
    probs = torch.softmax(outputs, dim=1).detach().cpu().numpy()[0]
    pred_idx = np.argmax(probs)
    
    classes = [name.capitalize() for name in Config.CLASS_NAMES]
    print("\n--- Diagnostic Prediction ---")
    print(f"File: {os.path.basename(image_path)}")
    print(f"Predicted Diagnosis: {classes[pred_idx]}")
    print(f"Confidence Scores:")
    for c_name, c_prob in zip(classes, probs):
        print(f"  - {c_name}: {c_prob*100:.2f}%")
    
    # 4. Grad-CAM Visualization
    if visualize and gradcam is not None:
        # Generate heatmap for predicted class (or class 1 for Tuberculosis target highlighting)
        target_class = 1  
        heatmap = gradcam.generate_heatmap(input_tensor, target_class)
        orig_img, blended_img = overlay_heatmap(image_path, heatmap)
        
        # Display plot
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        axes[0].imshow(orig_img)
        axes[0].set_title("Original CXR")
        axes[0].axis("off")
        
        axes[1].imshow(blended_img)
        axes[1].set_title("Grad-CAM Heatmap (DenseNet Projection)")
        axes[1].axis("off")
        
        plt.tight_layout()
        output_plot_path = os.path.join(Config.OUTPUT_DIR, f"gradcam_{os.path.basename(image_path)}")
        plt.savefig(output_plot_path, dpi=150)
        plt.show()
        print(f"Saved Grad-CAM saliency plot to: {output_plot_path}")
        
        gradcam.remove_hooks()

if __name__ == "__main__":
    default_ckpt = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")
    
    parser = argparse.ArgumentParser(description="Run single-image Tuberculosis CXR inference.")
    parser.add_argument("--image", type=str, required=True, help="Path to input CXR image file")
    parser.add_argument("--checkpoint", type=str, default=default_ckpt, help="Path to model checkpoint (.pth)")
    parser.add_argument("--gradcam", action="store_true", help="Generate and save Grad-CAM saliency maps")
    args = parser.parse_args()
    
    run_inference(
        image_path=args.image,
        checkpoint_path=args.checkpoint,
        visualize=args.gradcam
    )
