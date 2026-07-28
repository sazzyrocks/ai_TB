import os
import csv
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler, Subset
from torch.amp import autocast, GradScaler
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, accuracy_score
from sklearn.model_selection import StratifiedKFold
import numpy as np
import pandas as pd
from tqdm import tqdm

from config import Config
from data.dataset import TBDataset, get_train_transforms, get_val_transforms
from models.hybrid_model import TBHybridModel

# Set seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

def get_class_imbalance_weights(labels):
    """
    Computes sample weights for WeightedRandomSampler.
    """
    class_counts = np.bincount(labels)
    class_weights = 1.0 / class_counts
    sample_weights = [class_weights[l] for l in labels]
    return torch.tensor(sample_weights, dtype=torch.float)

def compute_metrics(y_true, y_pred, y_probs):
    """
    Computes Accuracy, Precision, Recall, F1-score, and ROC-AUC.
    """
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
    try:
        auc = roc_auc_score(y_true, y_probs)
    except Exception:
        auc = 0.5  # Fallback if only one class is present in batch
    return acc, precision, recall, f1, auc

def train_epoch(model, dataloader, criterion, optimizer, scaler, device):
    model.train()
    epoch_loss = 0.0
    all_targets = []
    all_preds = []
    all_probs = []

    progress = tqdm(
        enumerate(dataloader),
        total=len(dataloader),
        desc="Training",
        ncols=120
    )

    for batch_idx, (inputs, targets) in progress:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad()

        if device.type == "cuda":
            with autocast("cuda"):
                outputs = model(inputs)
                loss = criterion(outputs, targets)

            scaler.scale(loss).backward()

            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), Config.GRAD_CLIP)

            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), Config.GRAD_CLIP)
            optimizer.step()

        epoch_loss += loss.item() * inputs.size(0)

        probs = torch.softmax(outputs, dim=1)[:, 1].detach().cpu().numpy()
        preds = torch.argmax(outputs, dim=1).detach().cpu().numpy()

        all_targets.extend(targets.cpu().numpy())
        all_preds.extend(preds)
        all_probs.extend(probs)

        progress.set_postfix({
            "Loss": f"{loss.item():.4f}"
        })

    epoch_loss /= len(dataloader.dataset)

    acc, prec, rec, f1, auc = compute_metrics(
        all_targets,
        all_preds,
        all_probs
    )

    return epoch_loss, acc, prec, rec, f1, auc

@torch.no_grad()
def validate(model, dataloader, criterion, device):
    model.eval()
    val_loss = 0.0
    all_targets = []
    all_preds = []
    all_probs = []
    
    for inputs, targets in dataloader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        val_loss += loss.item() * inputs.size(0)
        probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
        preds = torch.argmax(outputs, dim=1).cpu().numpy()
        
        all_targets.extend(targets.cpu().numpy())
        all_preds.extend(preds)
        all_probs.extend(probs)
        
    val_loss /= len(dataloader.dataset)
    acc, prec, rec, f1, auc = compute_metrics(all_targets, all_preds, all_probs)
    return val_loss, acc, prec, rec, f1, auc

def run_training_session(train_dataset, val_dataset, fold_idx=None):
    """
    Executes a training session on a given train/val dataset.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")
    
    # Checkpoint name configuration
    prefix = f"fold_{fold_idx}_" if fold_idx is not None else ""
    best_chk_path = os.path.join(Config.CHECKPOINT_DIR, f"{prefix}best_model.pth")
    last_chk_path = os.path.join(Config.CHECKPOINT_DIR, f"{prefix}last_model.pth")
    csv_log_path = os.path.join(Config.OUTPUT_DIR, f"{prefix}metrics.csv")
    
    # Dataloaders
    train_labels = [train_dataset.dataset.labels[i] if isinstance(train_dataset, Subset) else train_dataset.labels[i] for i in range(len(train_dataset))]
    
    loader_kwargs = {
        "pin_memory": True,
    }
    if Config.NUM_WORKERS > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2

    if Config.USE_WEIGHTED_SAMPLER:
        sample_weights = get_class_imbalance_weights(train_labels)
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
        train_loader = DataLoader(
            train_dataset, 
            batch_size=Config.BATCH_SIZE, 
            sampler=sampler, 
            num_workers=Config.NUM_WORKERS,
            **loader_kwargs
        )
    else:
        train_loader = DataLoader(
            train_dataset, 
            batch_size=Config.BATCH_SIZE, 
            shuffle=True, 
            num_workers=Config.NUM_WORKERS,
            **loader_kwargs
        )
        
    val_loader = DataLoader(
        val_dataset, 
        batch_size=Config.BATCH_SIZE, 
        shuffle=False, 
        num_workers=Config.NUM_WORKERS,
        **loader_kwargs
    )
    
    # Initialize Model
    model = TBHybridModel(
        pretrained=Config.PRETRAINED,
        freeze_early=Config.FREEZE_EARLY,
        projection_dim=Config.PROJECTION_DIM,
        num_heads=Config.NUM_HEADS,
        dropout=Config.DROPOUT,
        convnext_model_name=Config.CONVNEXT_MODEL_NAME
    ).to(device)
    
    # Separate parameter groups
    backbone_params = []
    head_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "densenet" in name or "convnext" in name:
            if "proj_" in name:
                head_params.append(param)
            else:
                backbone_params.append(param)
        else:
            head_params.append(param)
            
    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": Config.BACKBONE_LR},
        {"params": head_params, "lr": Config.HEAD_LR}
    ], weight_decay=Config.WEIGHT_DECAY)
    
    # LR Scheduler (reduces learning rate when validation loss plates)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )
    
    # Loss Function with Label Smoothing
    criterion = nn.CrossEntropyLoss(label_smoothing=Config.LABEL_SMOOTHING)
    scaler = GradScaler("cuda") if device.type == "cuda" else None
    
    # Metrics logger setup
    csv_file = open(csv_log_path, mode='w', newline='')
    log_writer = csv.writer(csv_file)
    log_writer.writerow([
        "epoch", "train_loss", "train_acc", "train_prec", "train_rec", "train_f1", "train_auc",
        "val_loss", "val_acc", "val_prec", "val_rec", "val_f1", "val_auc"
    ])
    
    # Early Stopping variables
    best_val_auc = 0.0
    patience_counter = 0
    
    for epoch in range(1, Config.NUM_EPOCHS + 1):

        print("\n" + "=" * 70)
        print(f"Starting Epoch {epoch}/{Config.NUM_EPOCHS}")
        print("=" * 70)

        tr_loss, tr_acc, tr_prec, tr_rec, tr_f1, tr_auc = train_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            scaler,
            device
        )

        print("Training Finished. Starting Validation...")

        v_loss, v_acc, v_prec, v_rec, v_f1, v_auc = validate(
            model,
            val_loader,
            criterion,
            device
        )

        scheduler.step(v_loss)

        print(
            f"Epoch {epoch}/{Config.NUM_EPOCHS} | "
            f"Train Loss={tr_loss:.4f} | "
            f"Val Loss={v_loss:.4f} | "
            f"Train Acc={tr_acc:.4f} | "
            f"Val Acc={v_acc:.4f} | "
            f"Val AUC={v_auc:.4f}"
        )
              
        log_writer.writerow([
            epoch, tr_loss, tr_acc, tr_prec, tr_rec, tr_f1, tr_auc,
            v_loss, v_acc, v_prec, v_rec, v_f1, v_auc
        ])
        csv_file.flush()
        
        # Save checkpoints
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_auc": v_auc
        }, last_chk_path)
        
        if v_auc > best_val_auc:
            best_val_auc = v_auc
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_auc": v_auc
            }, best_chk_path)
            print(f"  --> Saved new best checkpoint with Val AUC: {best_val_auc:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
                print(f"Early stopping triggered after {epoch} epochs. Best Val AUC: {best_val_auc:.4f}")
                break
                
    csv_file.close()
    print(f"Training session complete. Logs saved to {csv_log_path}")

def main():
    Config.create_dirs()
    
    # Load dataset
    train_dir = os.path.join(Config.PROCESSED_DATA_DIR, "train")
    val_dir = os.path.join(Config.PROCESSED_DATA_DIR, "val")
    
    if not os.path.exists(train_dir) or not os.path.exists(val_dir):
        print("Processed dataset folders not found. Please run 'python data/prepare_data.py' first.")
        return
        
    if Config.USE_K_FOLD:
        print(f"Starting {Config.NUM_FOLDS}-Fold Stratified Cross Validation...")
        # Combine train & val datasets for K-Fold splits
        full_dataset_train = TBDataset(root_dir=train_dir, transform=get_train_transforms())
        full_dataset_val = TBDataset(root_dir=val_dir, transform=get_val_transforms())
        
        # Merge file paths and labels
        all_paths = full_dataset_train.image_paths + full_dataset_val.image_paths
        all_labels = full_dataset_train.labels + full_dataset_val.labels
        
        skf = StratifiedKFold(n_splits=Config.NUM_FOLDS, shuffle=True, random_state=42)
        
        # Custom dataset to dynamically handle combined lists
        class CombinedDataset(torch.utils.data.Dataset):
            def __init__(self, paths, labels, transform):
                self.paths = paths
                self.labels = labels
                self.transform = transform
            def __len__(self):
                return len(self.paths)
            def __getitem__(self, idx):
                image = cv2.imread(self.paths[idx])
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                if self.transform:
                    augmented = self.transform(image=image)
                    image = augmented['image']
                return image, torch.tensor(self.labels[idx], dtype=torch.long)
                
        for fold, (train_idx, val_idx) in enumerate(skf.split(all_paths, all_labels)):
            print(f"\n--- Fold {fold + 1}/{Config.NUM_FOLDS} ---")
            
            fold_train_paths = [all_paths[i] for i in train_idx]
            fold_train_labels = [all_labels[i] for i in train_idx]
            fold_val_paths = [all_paths[i] for i in val_idx]
            fold_val_labels = [all_labels[i] for i in val_idx]
            
            fold_train_dataset = CombinedDataset(fold_train_paths, fold_train_labels, get_train_transforms())
            fold_val_dataset = CombinedDataset(fold_val_paths, fold_val_labels, get_val_transforms())
            
            run_training_session(fold_train_dataset, fold_val_dataset, fold_idx=fold+1)
    else:
        print("Starting standard train/val session...")
        train_dataset = TBDataset(root_dir=train_dir, transform=get_train_transforms())
        val_dataset = TBDataset(root_dir=val_dir, transform=get_val_transforms())
        run_training_session(train_dataset, val_dataset)

if __name__ == "__main__":
    import cv2  # ensure imports for script
    main()
