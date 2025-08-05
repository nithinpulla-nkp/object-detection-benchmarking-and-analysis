# Faster R-CNN Object Detection Pipeline

This directory contains a complete Faster R-CNN object detection pipeline for training and evaluating on Pascal VOC 2012 dataset.

## Overview

The pipeline implements Faster R-CNN with ResNet50 + Feature Pyramid Network (FPN) backbone for object detection on Pascal VOC 2012 dataset. It provides comprehensive training, evaluation, and visualization capabilities in a single standalone notebook.

## Main Component

### `notebooks/fasterrcnn_standalone_kaggle.ipynb`

A complete standalone Jupyter notebook containing the entire Faster R-CNN pipeline. This notebook is designed to run in **Kaggle environment** without any external dependencies.

**What it does:**
- Converts Pascal VOC dataset to COCO format
- Trains Faster R-CNN model with ResNet50+FPN backbone on 20 VOC classes
- Evaluates model using COCO metrics (mAP@0.5, mAP@0.5:0.95)
- Benchmarks inference speed (FPS)
- Generates comprehensive visualizations and comparisons
- Tests on standardized comparison images
- Saves all results, checkpoints, and reports

**What it produces:**
```
/kaggle/working/fasterrcnn_outputs/
├── models/
│   ├── fasterrcnn_epoch_*.pth      # Model checkpoints
│   ├── fasterrcnn_final.pth        # Final trained model
│   ├── fasterrcnn_epoch_*_best.pth # Best performing model
│   └── checkpoint_latest.pth       # Latest checkpoint (for resuming)
├── logs/
│   └── fasterrcnn_voc_*.log       # Training logs
├── predictions/
│   ├── fasterrcnn_predictions.json # Model predictions
│   ├── fasterrcnn_metrics.json     # Performance metrics with FPS
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
    'output_dir': '/kaggle/working/fasterrcnn_outputs',    # Kaggle working directory
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
    'output_dir': './fasterrcnn_outputs',              # Local output directory
    'batch_size': 4,                                   # Adjust based on your GPU memory
    'num_epochs': 10,                                  # Increase for better results
    'learning_rate': 0.005,                           # SGD learning rate
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    # ... keep other settings
}
```

### Google Colab Example:
```python
CONFIG = {
    'voc_root': '/content/drive/MyDrive/VOC2012',      # Google Drive mounted path
    'output_dir': '/content/fasterrcnn_outputs',       # Colab working directory
    'batch_size': 2,                                   # Adjust for Colab GPU limits
    'num_epochs': 5,
    'learning_rate': 0.005,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    # ... keep other settings
}
```

## Features

- **Standalone**: No external .py files required - everything embedded in the notebook
- **Resume Training**: Automatic checkpoint loading and resuming from interruptions
- **Comprehensive Logging**: Detailed training logs and metrics tracking
- **Visualization**: Training progress, prediction samples, and comparison results
- **Model Checkpointing**: Automatic saving of best models and regular checkpoints
- **COCO Evaluation**: Standard object detection metrics (mAP@0.5, mAP@0.5:0.95)
- **Speed Benchmarking**: FPS measurement and performance analysis
- **Comparison Testing**: Standardized test images for model comparison
- **Complete Reports**: JSON reports with all experiment details

## Model Details

- **Architecture**: Faster R-CNN with ResNet50 backbone + Feature Pyramid Network (FPN)
- **Dataset**: Pascal VOC 2012 (20 object classes)
- **Input Size**: Variable (maintains aspect ratio)
- **Classes**: aeroplane, bicycle, bird, boat, bottle, bus, car, cat, chair, cow, diningtable, dog, horse, motorbike, person, pottedplant, sheep, sofa, train, tvmonitor
- **Evaluation**: COCO-style metrics with IoU thresholds from 0.5 to 0.95
- **Optimizer**: SGD with momentum (0.9) and weight decay (0.0005)
- **Scheduler**: StepLR (decay by 0.1 every 3 epochs)

## Performance Expectations

### Typical Results (Pascal VOC 2012)
- **mAP@0.5**: ~0.65-0.75 (depending on training epochs)
- **mAP@0.5:0.95**: ~0.45-0.55 (depending on training epochs)
- **Inference Speed**: ~6-10 FPS (GPU dependent)
- **Model Size**: ~160MB
- **Training Time**: ~15-30 minutes per epoch (GPU dependent)

### Hardware Requirements
- **GPU**: Highly recommended (CUDA compatible) - CPU training is very slow
- **RAM**: 8GB+ recommended
- **VRAM**: 4GB+ recommended for batch_size=2
- **Storage**: 3GB+ for dataset and outputs

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

## Training Configuration

### Default Parameters
- **Epochs**: 5 (increase for better results)
- **Batch Size**: 2 (adjust based on GPU memory)
- **Learning Rate**: 0.005 (SGD with momentum)
- **Momentum**: 0.9
- **Weight Decay**: 0.0005
- **Confidence Threshold**: 0.5 (for evaluation)

### Memory Optimization
If you encounter GPU memory issues:
- Reduce `batch_size` to 1
- Reduce `num_workers` to 2 or 0
- Use gradient accumulation for effective larger batch sizes

## Troubleshooting

### Common Issues
1. **CUDA Out of Memory**: Reduce batch_size to 1
2. **Slow Training**: Ensure GPU is being used, check device setting
3. **Import Errors**: All dependencies are embedded in the notebook
4. **Dataset Not Found**: Update `voc_root` path in configuration

### Resume Training
The notebook automatically saves checkpoints and can resume from interruptions. Simply re-run the training cell to continue from the latest checkpoint.

## Comparison with Other Models

This Faster R-CNN implementation is designed for fair comparison with:
- **SSD300**: `../SSD300/notebooks/ssd300_standalone_kaggle.ipynb`
- **YOLOv5**: `../YOLOv5_custom/notebooks/` (when available)
- **Comparison Images**: Uses standardized test images from `comparison_images.json`

All models use the same evaluation metrics and test images for consistent benchmarking.

## Output Analysis

### Key Files to Check
- **Training Log**: Monitor loss progression and training time
- **Metrics JSON**: Compare mAP scores and FPS with other models
- **Comparison Results**: Analyze per-image performance on standard test set
- **Visualizations**: Review training curves and prediction quality
- **Final Report**: Complete experiment summary with all details

The notebook provides comprehensive analysis tools to understand model performance and compare with other detection methods.