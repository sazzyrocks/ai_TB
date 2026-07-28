# Dataset Format & Structure Specification

This document details the expected layout and naming conventions for loading chest X-ray images.

## 1. Expected Raw Folder Structure

To import raw images, place them inside the `data/raw` folder (or customize `RAW_DATA_DIR` in `config.py`). The files can be structured inside subdirectories or mixed within a single folder. 

`prepare_data.py` recursively scans the directories to find all images.

```
tb_detection/
└── data/
    └── raw/
        ├── CHNCXR_0001_0.png   <-- Normal (Shenzhen)
        ├── CHNCXR_0327_1.png   <-- Tuberculosis (Shenzhen)
        ├── MCUCXR_0001_0.png   <-- Normal (Montgomery)
        └── MCUCXR_0080_1.png   <-- Tuberculosis (Montgomery)
```

## 2. File Naming Convention

The data preparation pipeline determines the diagnosis labels directly from the image filenames:

* **Normal (Class 0):** The filename (excluding extension) must end with `_0` (e.g. `CHNCXR_0001_0.png`, `image_sample_0.jpg`).
* **Tuberculosis (Class 1):** The filename (excluding extension) must end with `_1` (e.g. `CHNCXR_0327_1.png`, `patient_scan_1.png`).

Images not matching these suffixes will be ignored.

## 3. Output Processed Directory Layout

After running `python data/prepare_data.py`, images are split and copied into structured subdirectories ready for PyTorch data loaders:

```
tb_detection/
└── data/
    └── processed/
        ├── train/
        │   ├── normal/
        │   └── tuberculosis/
        ├── val/
        │   ├── normal/
        │   └── tuberculosis/
        └── test/
            ├── normal/
            └── tuberculosis/
```
