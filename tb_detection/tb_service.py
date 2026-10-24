import os
import io
import time
import base64
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from config import Config
from data.dataset import get_val_transforms
from models.hybrid_model import TBHybridModel
from explainers import (
    GradCAMExplainer, 
    GradCAMPlusPlusExplainer, 
    generate_superpixel_lime, 
    apply_heatmap_overlay
)

def image_to_base64(img_rgb: np.ndarray, format: str = "PNG") -> str:
    """Encodes an RGB NumPy image to a base64 data URI."""
    pil_img = Image.fromarray(np.uint8(img_rgb))
    buffer = io.BytesIO()
    pil_img.save(buffer, format=format)
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/{format.lower()};base64,{encoded}"

class TBInferenceService:
    """
    Singleton service managing the Deep Learning Hybrid Model and Explainable AI pipelines.
    """
    _instance = None

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.transform = get_val_transforms()
        self.is_real_model = False
        self.checkpoint_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")
        self._load_model()

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _load_model(self):
        """Loads trained weights if checkpoint exists, otherwise initializes baseline architecture."""
        try:
            print(f"[TBService] Initializing Hybrid Model on device: {self.device}")
            self.model = TBHybridModel(
                pretrained=False,
                freeze_early=False,
                projection_dim=Config.PROJECTION_DIM,
                num_heads=Config.NUM_HEADS,
                dropout=Config.DROPOUT,
                convnext_model_name=Config.CONVNEXT_MODEL_NAME
            ).to(self.device)

            if os.path.exists(self.checkpoint_path):
                print(f"[TBService] Loading checkpoint from: {self.checkpoint_path}")
                checkpoint = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
                state_dict = checkpoint.get("model_state_dict", checkpoint)
                self.model.load_state_dict(state_dict)
                self.is_real_model = True
                print("[TBService] Successfully loaded trained weights!")
            else:
                print(f"[TBService] No checkpoint found at {self.checkpoint_path}. Using base model.")
                self.is_real_model = False

            self.model.eval()
        except Exception as e:
            print(f"[TBService] Warning: Failed to load model weights: {e}")
            self.model = None
            self.is_real_model = False

    def predict_and_explain(self, image_bytes: bytes) -> dict:
        """
        Executes prediction and generates LIME, Grad-CAM, and Grad-CAM++ visualizations.
        """
        start_time = time.perf_counter()

        # 1. Decode image bytes to RGB NumPy array
        np_arr = np.frombuffer(image_bytes, np.uint8)
        img_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise ValueError("Invalid image file provided. Could not decode image.")
        
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        # Standardize display dimensions for clean UI rendering
        display_img = cv2.resize(img_rgb, (384, 384))
        
        # 2. Check if model is available
        if self.model is None:
            # Fallback mock service layer as specified in requirements
            return self._mock_prediction(display_img)

        # 3. Model Preprocessing
        aug = self.transform(image=img_rgb)
        input_tensor = aug['image'].unsqueeze(0).to(self.device)
        input_tensor.requires_grad = True

        target_class = 1  # Always explain Tuberculosis evidence

        # 4. Single-pass Forward pass, Grad-CAM and Grad-CAM++
        gradcam_map = None
        try:
            cam_explainer = GradCAMExplainer(self.model, self.model.proj_densenet)
            gradcam_map, gradcam_pp_map, outputs = cam_explainer.generate_both(input_tensor, class_idx=target_class)
            gradcam_overlay = apply_heatmap_overlay(display_img, gradcam_map, alpha=0.45)
            gradcam_pp_overlay = apply_heatmap_overlay(display_img, gradcam_pp_map, alpha=0.45)
            cam_explainer.cleanup()
            
            probs = F.softmax(outputs, dim=1).detach().cpu().numpy()[0]
            normal_prob = float(probs[0])
            tb_prob = float(probs[1])
        except Exception as e:
            print(f"[TBService] CAM error: {e}")
            with torch.no_grad():
                outputs = self.model(input_tensor)
                probs = F.softmax(outputs, dim=1).detach().cpu().numpy()[0]
                normal_prob = float(probs[0])
                tb_prob = float(probs[1])
            gradcam_overlay = display_img
            gradcam_pp_overlay = display_img

        # Determine classification
        # Class 0: Normal, Class 1: Tuberculosis
        if tb_prob >= 0.5:
            pred_label = "Tuberculosis"
        else:
            pred_label = "Normal"

        # 5. Fast Thoracic-Constrained Superpixel LIME Explanations
        try:
            lime_overlay = generate_superpixel_lime(
                model=self.model,
                input_image_rgb=display_img,
                transform=self.transform,
                device=self.device,
                target_class=target_class,
                saliency_map=gradcam_map,
                num_samples=28,
                n_segments=60
            )
        except Exception as e:
            print(f"[TBService] LIME error: {e}")
            lime_overlay = display_img

        elapsed_time = time.perf_counter() - start_time

        return {
            "prediction": pred_label,
            "tb_probability": round(tb_prob, 4),
            "normal_probability": round(normal_prob, 4),
            "original_image": image_to_base64(display_img),
            "lime_image": image_to_base64(lime_overlay),
            "gradcam_image": image_to_base64(gradcam_overlay),
            "gradcam_plus_plus_image": image_to_base64(gradcam_pp_overlay),
            "inference_time_ms": round(elapsed_time * 1000, 1),
            "inference_time_sec": round(elapsed_time, 2)
        }

    def _mock_prediction(self, display_img: np.ndarray) -> dict:
        """Fallback mock service layer if deep learning pipeline is offline."""
        mock_heatmap = np.zeros((14, 14), dtype=np.float32)
        mock_heatmap[4:10, 4:10] = 0.85
        cam_overlay = apply_heatmap_overlay(display_img, mock_heatmap)
        return {
            "prediction": "Tuberculosis",
            "tb_probability": 0.9432,
            "normal_probability": 0.0568,
            "original_image": image_to_base64(display_img),
            "lime_image": image_to_base64(display_img),
            "gradcam_image": image_to_base64(cam_overlay),
            "gradcam_plus_plus_image": image_to_base64(cam_overlay)
        }
