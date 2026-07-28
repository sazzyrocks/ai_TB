import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    roc_curve, 
    auc, 
    ConfusionMatrixDisplay
)
from config import Config
from data.dataset import TBDataset, get_val_transforms
from models.hybrid_model import TBHybridModel

def evaluate_model(checkpoint_path: str, output_dir: str):
    """
    Evaluates a trained model checkpoint on the test dataset, saving metrics and plots.
    """
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating model using device: {device}")
    
    # 1. Load Dataset
    test_dir = os.path.join(Config.PROCESSED_DATA_DIR, "test")
    if not os.path.exists(test_dir):
        raise FileNotFoundError(f"Test directory not found: {test_dir}. Run prepare_data.py first.")
        
    test_dataset = TBDataset(root_dir=test_dir, transform=get_val_transforms())
    test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=Config.NUM_WORKERS)
    
    # 2. Instantiate and load model weights
    model = TBHybridModel(
        pretrained=False,  # Weights are loaded from checkpoint
        freeze_early=False,
        projection_dim=Config.PROJECTION_DIM,
        num_heads=Config.NUM_HEADS,
        dropout=Config.DROPOUT,
        convnext_model_name=Config.CONVNEXT_MODEL_NAME
    ).to(device)
    
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    
    model.eval()
    all_targets = []
    all_preds = []
    all_probs = []
    
    # 3. Inference loop
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            
            probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            
            all_targets.extend(targets.numpy())
            all_preds.extend(preds)
            all_probs.extend(probs)
            
    all_targets = np.array(all_targets)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    
    # 4. Generate & Save Classification Report
    target_names = [name.capitalize() for name in Config.CLASS_NAMES]
    cls_report = classification_report(
        all_targets, 
        all_preds, 
        target_names=target_names,
        zero_division=0
    )
    print("\nTest Classification Report:")
    print(cls_report)
    
    report_path = os.path.join(output_dir, "classification_report.txt")
    with open(report_path, "w") as f:
        f.write(cls_report)
    print(f"Saved classification report to: {report_path}")
    
    # 5. Generate & Save Confusion Matrix
    cm = confusion_matrix(all_targets, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
    fig, ax = plt.subplots(figsize=(6, 6))
    disp.plot(cmap=plt.cm.Blues, ax=ax, values_format='d')
    plt.title("Confusion Matrix")
    cm_path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(cm_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"Saved confusion matrix plot to: {cm_path}")
    
    # 6. Generate & Save ROC Curve
    fpr, tpr, _ = roc_curve(all_targets, all_probs)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    
    roc_path = os.path.join(output_dir, "roc_curve.png")
    plt.savefig(roc_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"Saved ROC curve plot to: {roc_path}")
    print("Evaluation completed successfully!")

if __name__ == "__main__":
    default_ckpt = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")
    
    parser = argparse.ArgumentParser(description="Evaluate a trained Tuberculosis detection model.")
    parser.add_argument("--checkpoint", type=str, default=default_ckpt, help="Path to model checkpoint (.pth)")
    parser.add_argument("--output_dir", type=str, default=Config.OUTPUT_DIR, help="Directory to save evaluation artifacts")
    args = parser.parse_args()
    
    evaluate_model(checkpoint_path=args.checkpoint, output_dir=args.output_dir)
