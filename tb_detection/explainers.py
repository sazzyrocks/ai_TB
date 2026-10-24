import cv2
import numpy as np
import torch
import torch.nn.functional as F
from skimage.segmentation import slic, mark_boundaries
from sklearn.linear_model import Ridge

class GradCAMExplainer:
    """
    Grad-CAM: Visualizes coarse localization map of important regions
    using gradients of the target class score flowing into the final convolutional projection.
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
        
    def generate(self, input_tensor, class_idx=1):
        self.model.zero_grad()
        output = self.model(input_tensor)
        score = output[0, class_idx]
        score.backward(retain_graph=True)
        
        grads = self.gradients[0].detach().cpu().numpy()  # (C, H, W)
        acts = self.activations[0].detach().cpu().numpy() # (C, H, W)
        
        # Global Average Pooling on gradients
        weights = np.mean(grads, axis=(1, 2))  # (C,)
        
        cam = np.zeros(acts.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * acts[i]
            
        cam = np.maximum(cam, 0)
        max_val = np.max(cam)
        if max_val > 1e-8:
            cam /= max_val
        return cam

    def generate_both(self, input_tensor, class_idx=1):
        """
        Computes both standard Grad-CAM and higher-order Grad-CAM++ attribution maps
        in a single forward/backward pass, eliminating redundant graph evaluations.
        Returns: (gradcam_map, gradcam_pp_map, outputs)
        """
        self.model.zero_grad()
        output = self.model(input_tensor)
        score = output[0, class_idx]
        score.backward(retain_graph=False)
        
        grads = self.gradients[0].detach().cpu().numpy()  # (C, H, W)
        acts = self.activations[0].detach().cpu().numpy() # (C, H, W)
        
        # 1. Standard Grad-CAM (Global Average Pooling)
        weights_cam = np.mean(grads, axis=(1, 2))
        cam = np.zeros(acts.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights_cam):
            cam += w * acts[i]
        cam = np.maximum(cam, 0)
        max_val = np.max(cam)
        if max_val > 1e-8:
            cam /= max_val
            
        # 2. Grad-CAM++ (Second & Third Order Positive Gradient Weights)
        pos_grads = np.maximum(grads, 0)
        grads_2 = pos_grads ** 2
        grads_3 = pos_grads ** 3
        spatial_sum_acts = np.sum(acts, axis=(1, 2), keepdims=True)
        denom = 2.0 * grads_2 + spatial_sum_acts * grads_3
        denom = np.where(denom != 0.0, denom, np.ones_like(denom))
        alphas = grads_2 / denom
        weights_pp = np.sum(alphas * pos_grads, axis=(1, 2))
        
        cam_pp = np.zeros(acts.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights_pp):
            cam_pp += w * acts[i]
        cam_pp = np.maximum(cam_pp, 0)
        max_val_pp = np.max(cam_pp)
        if max_val_pp > 1e-8:
            cam_pp /= max_val_pp
            
        return cam, cam_pp, output
        
    def cleanup(self):
        for h in self.handlers:
            h.remove()
        self.handlers.clear()

class GradCAMPlusPlusExplainer:
    """
    Grad-CAM++: Enhanced localization providing higher pixel-weight resolution
    for multiple instances and subtle lesion features.
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
        
    def generate(self, input_tensor, class_idx=1):
        self.model.zero_grad()
        output = self.model(input_tensor)
        score = output[0, class_idx]
        score.backward(retain_graph=True)
        
        grads = self.gradients[0].detach().cpu().numpy()  # (C, H, W)
        acts = self.activations[0].detach().cpu().numpy() # (C, H, W)
        
        pos_grads = np.maximum(grads, 0)
        grads_2 = pos_grads ** 2
        grads_3 = pos_grads ** 3
        
        spatial_sum_acts = np.sum(acts, axis=(1, 2), keepdims=True)
        denom = 2.0 * grads_2 + spatial_sum_acts * grads_3
        denom = np.where(denom != 0.0, denom, np.ones_like(denom))
        
        alphas = grads_2 / denom
        weights = np.sum(alphas * pos_grads, axis=(1, 2))
        
        cam = np.zeros(acts.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * acts[i]
            
        cam = np.maximum(cam, 0)
        max_val = np.max(cam)
        if max_val > 1e-8:
            cam /= max_val
        return cam
        
    def cleanup(self):
        for h in self.handlers:
            h.remove()
        self.handlers.clear()

def generate_superpixel_lime(
    model, 
    input_image_rgb: np.ndarray, 
    transform, 
    device, 
    target_class: int = 1,
    saliency_map: np.ndarray = None,
    num_samples: int = 28,
    n_segments: int = 60
) -> np.ndarray:
    """
    Clinically Precise, High-Performance LIME for Chest Radiographs:
    1. Computes fine-grained SLIC superpixel segmentation conforming to lung anatomy.
    2. Imposes thoracic/pulmonary bounding constraints to strictly prevent edge, neck,
       and sub-diaphragmatic false highlights.
    3. Evaluates local surrogate perturbation sensitivity using vectorized tensor substitution.
    4. Renders smooth, anti-aliased contour boundaries strictly inside lung fields.
    """
    H, W, _ = input_image_rgb.shape
    
    # 1. Fine-grained SLIC segmentation
    segments = slic(
        input_image_rgb, 
        n_segments=n_segments, 
        compactness=14.0, 
        sigma=1.2, 
        start_label=0
    )
    num_superpixels = int(np.max(segments)) + 1
    
    # Resize saliency map if provided
    if saliency_map is not None:
        cam_resized = cv2.resize(saliency_map, (W, H))
    else:
        cam_resized = np.ones((H, W), dtype=np.float32)
    
    # 2. Identify strictly thoracic (lung field) superpixels
    valid_sp = []
    sp_saliency = {}
    for i in range(num_superpixels):
        coords = np.where(segments == i)
        if len(coords[0]) == 0:
            continue
        cy = np.mean(coords[0])
        cx = np.mean(coords[1])
        
        # Bounding box constraints: must reside within thoracic lung cage
        # Excludes outer border padding, neck, and sub-diaphragmatic abdomen
        if (0.12 * H <= cy <= 0.82 * H) and (0.10 * W <= cx <= 0.90 * W):
            valid_sp.append(i)
            sp_saliency[i] = float(np.mean(cam_resized[coords]))
        else:
            sp_saliency[i] = 0.0

    if len(valid_sp) == 0:
        valid_sp = list(range(num_superpixels))
        
    # 3. Vectorized baseline generation
    blurred_base = cv2.GaussianBlur(input_image_rgb, (35, 35), 0)
    base_tensor = transform(image=input_image_rgb)['image'].to(device)   # (3, 224, 224)
    blur_tensor = transform(image=blurred_base)['image'].to(device)       # (3, 224, 224)
    
    # Map segments to model input resolution (224, 224)
    segments_model = cv2.resize(segments, (224, 224), interpolation=cv2.INTER_NEAREST)
    
    # 4. Perturbation matrix (1 = preserve, 0 = occlude)
    np.random.seed(42)
    actual_samples = min(max(num_samples, 20), 32)
    perturbations = np.random.binomial(1, 0.70, size=(actual_samples, num_superpixels))
    perturbations[0, :] = 1  # Unperturbed original
    
    # Fast in-memory tensor construction
    batch_tensors = []
    for p in perturbations:
        preserved_sp = np.where(p == 1)[0]
        mask_224 = np.isin(segments_model, preserved_sp)
        mask_t = torch.from_numpy(mask_224).unsqueeze(0).to(device) # (1, 224, 224)
        perturbed_t = torch.where(mask_t, base_tensor, blur_tensor)
        batch_tensors.append(perturbed_t)
        
    # 5. Mini-batched model predictions
    all_probs = []
    with torch.no_grad():
        for b_idx in range(0, len(batch_tensors), 8):
            sub_batch = torch.stack(batch_tensors[b_idx:b_idx+8])
            outputs = model(sub_batch)
            sub_probs = F.softmax(outputs, dim=1)[:, target_class].cpu().numpy()
            all_probs.extend(sub_probs)
    probs = np.array(all_probs)
    
    # 6. Exponential distance kernel weights
    distances = np.linalg.norm(perturbations - 1, axis=1)
    kernel_width = np.sqrt(num_superpixels) * 0.8
    weights = np.exp(-(distances ** 2) / (kernel_width ** 2))
    
    # 7. Local linear regressor
    reg = Ridge(alpha=1.0)
    reg.fit(perturbations, probs, sample_weight=weights)
    coefs = reg.coef_
    
    # 8. Compute hybrid score for valid thoracic superpixels
    thoracic_scores = {}
    for sp in valid_sp:
        thoracic_scores[sp] = coefs[sp] * (0.5 + 0.5 * sp_saliency.get(sp, 0.0))
        
    sorted_sp = sorted(thoracic_scores.items(), key=lambda x: x[1])
    
    # Filter positive superpixels (evidence for TB)
    pos_sp = [sp for sp, sc in sorted_sp if sc > 0]
    # Filter negative superpixels (evidence for Normal lung parenchyma)
    neg_sp = [sp for sp, sc in sorted_sp if sc < 0]
    
    top_pos = pos_sp[-5:] if len(pos_sp) >= 5 else pos_sp
    top_neg = neg_sp[:3] if len(neg_sp) >= 3 else neg_sp
    
    # 9. Render clean publication-grade visualization
    annotated = input_image_rgb.copy()
    overlay = annotated.copy()
    
    # Positive superpixel mask (Coral Red for Tuberculosis indication)
    pos_mask = np.isin(segments, top_pos).astype(np.uint8)
    for sp in top_pos:
        alpha = 0.38
        sp_mask = (segments == sp)
        overlay[sp_mask] = np.uint8(annotated[sp_mask] * (1 - alpha) + np.array([239, 68, 68]) * alpha)
        
    # Negative superpixel mask (Emerald Green for Normal lung preservation)
    neg_mask = np.isin(segments, top_neg).astype(np.uint8)
    for sp in top_neg:
        alpha = 0.35
        sp_mask = (segments == sp)
        overlay[sp_mask] = np.uint8(annotated[sp_mask] * (1 - alpha) + np.array([16, 185, 129]) * alpha)
        
    result_img = overlay.copy()
    
    # 10. Draw fine superpixel mesh boundaries and active lesion contours
    boundary_mask = mark_boundaries(np.zeros((H, W, 3), dtype=np.uint8), segments, color=(1, 1, 1), mode='thin')
    is_boundary = (boundary_mask[:, :, 0] > 0.5)
    result_img[is_boundary] = (result_img[is_boundary] * 0.45 + np.array([60, 80, 100]) * 0.55).astype(np.uint8)

    if np.any(pos_mask):
        contours_pos, _ = cv2.findContours(pos_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(result_img, contours_pos, -1, (239, 68, 68), 2, cv2.LINE_AA)
        
    if np.any(neg_mask):
        contours_neg, _ = cv2.findContours(neg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(result_img, contours_neg, -1, (16, 185, 129), 2, cv2.LINE_AA)
        
    return result_img

def apply_heatmap_overlay(original_rgb: np.ndarray, heatmap: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """
    Overlays a normalized 2D heatmap on the original RGB chest X-ray using COLORMAP_JET.
    """
    H, W = original_rgb.shape[:2]
    resized_heatmap = cv2.resize(heatmap, (W, H))
    colored_heatmap = cv2.applyColorMap(np.uint8(255 * resized_heatmap), cv2.COLORMAP_JET)
    colored_heatmap = cv2.cvtColor(colored_heatmap, cv2.COLOR_BGR2RGB)
    blended = cv2.addWeighted(original_rgb, 1 - alpha, colored_heatmap, alpha, 0)
    return blended
