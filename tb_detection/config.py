import os

class Config:
    # --- Paths ---
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    RAW_DATA_DIR = os.path.join(BASE_DIR, "dataset", "raw")
    PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "dataset", "processed")
    CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
    OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
    
    # --- Dataset Config ---
    IMAGE_SIZE = (224, 224)
    BATCH_SIZE = 32
    NUM_WORKERS = 0  # 0 is safest for Windows systems to prevent multi-processing issues
    
    # Class folder names (can be adjusted for TBX11K, e.g. ["healthy", "tb"])
    CLASS_NAMES = ["normal", "tuberculosis"]
    
    # Stratified Split ratios
    TRAIN_SPLIT = 0.70
    VAL_SPLIT = 0.15
    TEST_SPLIT = 0.15
    
    # --- Model Config ---
    PRETRAINED = True
    FREEZE_EARLY = True  # If True, freezes earlier layers of backbones
    PROJECTION_DIM = 512
    NUM_HEADS = 4
    DROPOUT = 0.4
    CONVNEXT_MODEL_NAME = "convnextv2_nano"
    
    # --- Training Config ---
    NUM_EPOCHS = 20  # Default epochs, early stopping will prevent overfitting
    BACKBONE_LR = 1e-5  # Fine-tuning learning rate for pretrained backbones
    HEAD_LR = 1e-4      # Higher learning rate for custom attention, projections, classifier
    WEIGHT_DECAY = 1e-4
    GRAD_CLIP = 1.0     # Maximum gradient norm for clipping
    LABEL_SMOOTHING = 0.1
    EARLY_STOPPING_PATIENCE = 5
    
    # --- Class Imbalance Handling ---
    USE_WEIGHTED_SAMPLER = True  # Use WeightedRandomSampler to balance classes in training loader
    
    # --- K-Fold Config ---
    USE_K_FOLD = False
    NUM_FOLDS = 5
    
    @classmethod
    def create_dirs(cls):
        """Creates the necessary directories if they do not exist."""
        for path in [cls.CHECKPOINT_DIR, cls.OUTPUT_DIR, cls.PROCESSED_DATA_DIR]:
            os.makedirs(path, exist_ok=True)
