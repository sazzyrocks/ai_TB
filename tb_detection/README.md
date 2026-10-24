# Tuberculosis Detection Hybrid Model Pipeline

A complete PyTorch pipeline for Tuberculosis (TB) detection from chest X-rays. The architecture combines **DenseNet121** and **ConvNeXt V2** backbones using channel projections, bidirectional cross-attention, learnable adaptive gated fusion, and an MLP classification head.

---

## 1. Setup & Installation

Ensure you have Python installed, then clone/copy the project and install requirements:

```bash
pip install -r requirements.txt
```

---

## 2. Dataset Preparation

1. Create a `data/raw/` folder in the project root:
   ```bash
   mkdir -p data/raw
   ```
2. Copy your raw chest X-ray images (Shenzhen / Montgomery) into `data/raw/`. Ensure files end with `_0` (for Normal) and `_1` (for Tuberculosis) before the extension. E.g. `CHNCXR_0001_0.png` or `CHNCXR_0327_1.png`.
3. Organize and split your raw dataset (70% train, 15% val, 15% test) by running:
   ```bash
   python data/prepare_data.py
   ```
This will create a structured directory split under `data/processed/`. See `DATA_FORMAT.md` for more details.

---

## 3. Training the Model

Configure hyperparameters, paths, and flags directly in `config.py`.

Run the training loop:
```bash
python train.py
```

### Key Training Features:
* **Custom Learning Rates:** Backbones are fine-tuned with a lower LR (`1e-5`) while custom modules (attention, projection, classifier MLP) are trained with a higher LR (`1e-4`).
* **Weighted Sampler:** Balances the dataset during mini-batch loading to handle class imbalances.
* **Mixed Precision:** Uses automated mixed precision (`AMP`) on GPU to speed up computation.
* **Checkpoints:** Saves the best checkpoints based on validation performance in the `checkpoints/` folder.

---

## 4. Evaluation

Evaluate the trained checkpoint (by default `checkpoints/best_model.pth`) on the test set:

```bash
python evaluate.py --checkpoint checkpoints/best_model.pth
```

This generates performance reports and saves plots to `outputs/`:
* `classification_report.txt` (Accuracy, Recall, F1, Precision)
* `confusion_matrix.png` (Confusion Matrix display)
* `roc_curve.png` (Receiver Operating Characteristic curve plot)

---

## 5. Single-Image Inference & Interpretability (Grad-CAM)

Predict diagnosis on a single chest X-ray image:

```bash
python inference.py --image path/to/image_0.png --checkpoint checkpoints/best_model.pth --gradcam
```

* **`--gradcam` flag:** Generates and saves a saliency heatmap overlay in `outputs/` displaying the areas in the image that the model's DenseNet branch focused on to make the diagnosis.
