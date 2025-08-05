# SSD300 Object Detection Pipeline

This directory contains a complete SSD300 object detection pipeline for training and evaluating on Pascal VOC 2012 dataset.

## Overview

The pipeline implements SSD300 (Single Shot MultiBox Detector) with VGG16 backbone for object detection on Pascal VOC 2012 dataset. It provides comprehensive training, evaluation, and visualization capabilities in a single standalone notebook.

## Main Component

### `notebooks/ssd300_standalone_kaggle.ipynb`

A complete standalone Jupyter notebook containing the entire SSD300 pipeline. This notebook is designed to run in **Kaggle environment** without any external dependencies.

**What it does:**
- Converts Pascal VOC dataset to COCO format
- Trains SSD300 model with VGG16 backbone on 20 VOC classes
- Evaluates model using COCO metrics (mAP@0.5, mAP@0.5:0.95)
- Generates comprehensive visualizations and comparisons
- Tests on standardized comparison images
- Saves all results, checkpoints, and reports

**What it produces:**
```
/kaggle/working/ssd300_outputs/
├── models/
│   ├── ssd300_epoch_*.pth          # Model checkpoints
│   ├── ssd300_final.pth            # Final trained model
│   └── ssd300_epoch_*_best.pth     # Best performing model
├── logs/
│   └── ssd300_voc_*.log           # Training logs
├── predictions/
│   ├── ssd300_predictions.json     # Model predictions
│   └── comparison_results.json     # Comparison test results
├── visualizations/
│   ├── training_progress.png       # Training metrics plots
│   ├── sample_predictions.png      # Prediction visualizations
│   └── comparison_predictions.png  # Ground truth vs predictions
├── data/
│   └── voc2012_coco.json         # Converted COCO format dataset
└── *_final_report.json           # Comprehensive experiment report
```

## Kaggle Setup

The notebook is configured for Kaggle with these default paths:

```python
CONFIG = {
    'voc_root': '/kaggle/input/voc2012/VOCdevkit/VOC2012',  # Kaggle dataset input
    'output_dir': '/kaggle/working/ssd300_outputs',        # Kaggle working directory
    # ... other settings
}
```

### Required Kaggle Dataset
Add "Pascal VOC 2012" dataset to your Kaggle notebook from: https://www.kaggle.com/datasets/huanghanchina/pascal-voc-2012

## Running Outside Kaggle

To run the notebook in other environments, modify the configuration in **Cell 2**:

### Local Machine Example:
```python
CONFIG = {
    'voc_root': '/path/to/your/VOCdevkit/VOC2012',     # Local VOC dataset path
    'output_dir': './ssd300_outputs',                  # Local output directory
    'batch_size': 8,                                   # Adjust based on your GPU memory
    'num_epochs': 10,                                  # Increase for better results
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    # ... keep other settings
}
```

### Google Colab Example:
```python
CONFIG = {
    'voc_root': '/content/drive/MyDrive/VOC2012',      # Google Drive mounted path
    'output_dir': '/content/ssd300_outputs',           # Colab working directory
    'batch_size': 4,                                   # Adjust for Colab GPU limits
    'num_epochs': 5,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    # ... keep other settings
}
```

## Features

- **Standalone**: No external .py files required - everything embedded in the notebook
- **Comprehensive Logging**: Detailed training logs and metrics tracking
- **Visualization**: Training progress, prediction samples, and comparison results
- **Model Checkpointing**: Automatic saving of best models and regular checkpoints
- **COCO Evaluation**: Standard object detection metrics (mAP@0.5, mAP@0.5:0.95)
- **Comparison Testing**: Standardized test images for model comparison
- **Complete Reports**: JSON reports with all experiment details

## Model Details

- **Architecture**: SSD300 with VGG16 backbone
- **Dataset**: Pascal VOC 2012 (20 object classes)
- **Input Size**: 300x300 pixels
- **Classes**: aeroplane, bicycle, bird, boat, bottle, bus, car, cat, chair, cow, diningtable, dog, horse, motorbike, person, pottedplant, sheep, sofa, train, tvmonitor
- **Evaluation**: COCO-style metrics with IoU thresholds from 0.5 to 0.95

## Requirements

The notebook uses standard deep learning packages:
- PyTorch & Torchvision
- PIL, Matplotlib, Seaborn
- NumPy, Pandas
- pycocotools
- Standard Python libraries (json, logging, pathlib, etc.)

All packages are pre-installed in Kaggle environment.

## Usage

1. **In Kaggle**: Simply add Pascal VOC 2012 dataset and run all cells
2. **Elsewhere**: Update the `CONFIG` paths in Cell 2 and run all cells

The notebook will automatically handle data conversion, training, evaluation, and generate all outputs in the specified directory.

## Legacy Components

This directory also contains modular source files in `src/` for reference:
- `train_ssd.py` - Training script
- `eval_ssd.py` - Evaluation script  
- `coco_voc.py` - Dataset loader
- `voc2coco.py` - Data converter

However, the **recommended approach is to use the standalone notebook** which contains all functionality in one file.