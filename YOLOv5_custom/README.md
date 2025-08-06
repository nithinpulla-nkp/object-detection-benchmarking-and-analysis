# YOLOv5 Object Detection Pipeline

This directory contains a comprehensive YOLOv5 object detection pipeline for training and evaluating on Pascal VOC 2012 dataset.

## Overview

The pipeline implements YOLOv5s with CSPDarknet53 backbone for object detection on Pascal VOC 2012 dataset. It provides comprehensive training, evaluation, and visualization capabilities in standalone notebooks designed for Kaggle compatibility.

## Main Components

### `notebooks/yolov5_enhanced_comprehensive.ipynb`

A complete standalone Jupyter notebook containing the entire YOLOv5 pipeline. This notebook is designed to run in **Kaggle environment** without any external dependencies.

**What it does:**
- Converts Pascal VOC dataset to YOLO format
- Trains YOLOv5s model on 20 VOC classes
- Evaluates model using COCO-style metrics (mAP@0.5, mAP@0.5:0.95)
- Benchmarks inference speed (FPS)
- Generates comprehensive visualizations and comparisons
- Tests on standardized comparison images
- Saves all results, checkpoints, and reports

**What it produces:**
```
/kaggle/working/yolov5_outputs/
├── models/
│   └── yolov5s_voc_enhanced/      # YOLOv5 trained model and weights
│       ├── weights/
│       │   ├── best.pt            # Best performing model
│       │   └── last.pt            # Latest checkpoint
│       ├── results.csv            # Training metrics
│       └── *.png                  # Training visualizations
├── logs/
│   └── yolov5_*.log              # Training logs
├── metrics/
│   ├── yolov5_comprehensive_metrics.json    # Performance metrics
│   ├── yolov5_speed_benchmark.json         # FPS benchmarking results
│   └── dataset_conversion_summary.json     # Dataset statistics
├── visualizations/
│   ├── training_progress.png               # Custom training plots
│   ├── yolov5_results.png                 # YOLOv5 built-in plots
│   ├── yolov5_confusion_matrix.png        # Confusion matrix
│   └── yolov5_*.png                       # Additional YOLOv5 visualizations
├── comparisons/
│   ├── yolov5_comparison_results/          # Inference on comparison images
│   ├── comparison_images_config.json       # Standardized test images
│   └── yolov5_comparison_summary.json      # Comparison test summary
└── reports/
    ├── *_final_report.json                 # Comprehensive experiment report
    └── *_summary.md                        # Human-readable summary
```

### `notebooks/yolov5.ipynb` (Original)

The original basic YOLOv5 training notebook with fundamental functionality:
- VOC to YOLO conversion
- Basic YOLOv5 training
- Simple visualization
- Model export

## Kaggle Setup

The enhanced notebook is configured for Kaggle with these default paths:

```python
CONFIG = {
    'voc_root': '/kaggle/input/voc2012/VOCdevkit/VOC2012',  # Kaggle dataset input
    'output_dir': '/kaggle/working/yolov5_outputs',         # Kaggle working directory
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
    'output_dir': './yolov5_outputs',                  # Local output directory
    'batch_size': 8,                                   # Adjust based on your GPU memory
    'num_epochs': 10,                                  # Increase for better results
    'learning_rate': 0.01,                            # YOLOv5 default learning rate
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    # ... keep other settings
}
```

### Google Colab Example:
```python
CONFIG = {
    'voc_root': '/content/drive/MyDrive/VOC2012',      # Google Drive mounted path
    'output_dir': '/content/yolov5_outputs',           # Colab working directory
    'batch_size': 4,                                   # Adjust for Colab GPU limits
    'num_epochs': 5,
    'learning_rate': 0.01,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    # ... keep other settings
}
```

## Features

- **Standalone**: No external .py files required - everything embedded in the notebook
- **VOC to YOLO Conversion**: Automatic dataset format conversion with validation
- **Comprehensive Logging**: Detailed training logs and metrics tracking
- **Enhanced Visualization**: Custom training progress plots + YOLOv5 built-in visualizations
- **Model Checkpointing**: Automatic saving of best models and regular checkpoints
- **COCO-Style Evaluation**: Standard object detection metrics (mAP@0.5, mAP@0.5:0.95)
- **Speed Benchmarking**: FPS measurement and detailed performance analysis
- **Comparison Testing**: Standardized test images for fair model comparison
- **Complete Reports**: JSON reports and human-readable summaries
- **Professional Output Structure**: Organized directory structure for all outputs

## Model Details

- **Architecture**: YOLOv5s with CSPDarknet53 backbone + PANet neck
- **Dataset**: Pascal VOC 2012 (20 object classes)
- **Input Size**: 640x640 (configurable)
- **Classes**: aeroplane, bicycle, bird, boat, bottle, bus, car, cat, chair, cow, diningtable, dog, horse, motorbike, person, pottedplant, sheep, sofa, train, tvmonitor
- **Evaluation**: COCO-style metrics with IoU thresholds from 0.5 to 0.95
- **Optimizer**: SGD with momentum (0.937) and weight decay (0.0005)
- **Scheduler**: Built-in YOLOv5 cosine LR scheduler

## Performance Expectations

### Typical Results (Pascal VOC 2012)
- **mAP@0.5**: ~0.60-0.75 (depending on training epochs)
- **mAP@0.5:0.95**: ~0.40-0.50 (depending on training epochs)
- **Inference Speed**: ~15-25 FPS (GPU dependent)
- **Model Size**: ~14.1MB (YOLOv5s)
- **Training Time**: ~5-15 minutes per epoch (GPU dependent)

### Hardware Requirements
- **GPU**: Highly recommended (CUDA compatible) - CPU training is very slow
- **RAM**: 8GB+ recommended
- **VRAM**: 4GB+ recommended for batch_size=16
- **Storage**: 3GB+ for dataset and outputs

## Requirements

The notebook uses standard deep learning packages:
- PyTorch & Torchvision
- YOLOv5 repository (auto-cloned)
- PIL, Matplotlib, Seaborn
- NumPy, Pandas
- OpenCV (cv2)
- Standard Python libraries (json, logging, pathlib, etc.)

All packages are pre-installed in Kaggle environment.

## Usage

1. **In Kaggle**: Simply add Pascal VOC 2012 dataset and run all cells
2. **Elsewhere**: Update the `CONFIG` paths in Cell 2 and run all cells

The notebook will automatically handle:
- YOLOv5 repository cloning and setup
- VOC to YOLO dataset conversion
- Model training with checkpointing
- Comprehensive evaluation and benchmarking
- Professional visualization generation
- Structured output organization

## Training Configuration

### Default Parameters
- **Epochs**: 5 (increase for better results)
- **Batch Size**: 16 (adjust based on GPU memory)
- **Learning Rate**: 0.01 (YOLOv5 default)
- **Image Size**: 640x640
- **Confidence Threshold**: 0.25 (for evaluation)
- **IoU Threshold**: 0.45 (for NMS)

### Memory Optimization
If you encounter GPU memory issues:
- Reduce `batch_size` to 8 or 4
- Use `--cache ram` for faster training (if enough RAM)
- Consider mixed precision training (automatically handled by YOLOv5)

## Troubleshooting

### Common Issues
1. **CUDA Out of Memory**: Reduce batch_size to 8 or 4
2. **Slow Training**: Ensure GPU is being used, check CUDA availability
3. **Dataset Not Found**: Update `voc_root` path in configuration
4. **YOLOv5 Clone Issues**: Check internet connection, may need to manually clone

### Training Resumption
YOLOv5 automatically handles training resumption from checkpoints. If training is interrupted, simply re-run the training cell.

## Comparison with Other Models

This YOLOv5 implementation is designed for fair comparison with:
- **SSD300**: `../SSD300/notebooks/ssd300_standalone_kaggle.ipynb`
- **Faster R-CNN**: `../FasterRCNN/notebooks/fasterrcnn_standalone_kaggle.ipynb`
- **Comparison Images**: Uses same standardized test images from comparison framework

All models use consistent:
- Evaluation metrics (COCO-style mAP)
- Test images for comparison
- Speed benchmarking methodology
- Output structure and reporting format

## Output Analysis

### Key Files to Check
- **Training Metrics**: `metrics/yolov5_comprehensive_metrics.json`
- **Speed Benchmark**: `metrics/yolov5_speed_benchmark.json`
- **Training Visualizations**: `visualizations/training_progress.png`
- **YOLOv5 Built-in Plots**: `visualizations/yolov5_*.png`
- **Comparison Results**: `comparisons/yolov5_comparison_results/`
- **Final Report**: `reports/*_final_report.json`
- **Summary**: `reports/*_summary.md`

### Comparison Readiness Checklist
✅ Standardized test images evaluated  
✅ Speed benchmarking completed  
✅ COCO metrics computed  
✅ Professional visualizations created  
✅ Structured output organization  
✅ Comprehensive reporting  

The notebook provides all necessary components for fair comparison with other object detection models in this benchmark suite.

## Next Steps

After running this notebook:
1. Compare results with SSD300 and Faster R-CNN models
2. Analyze relative strengths/weaknesses across different metrics
3. Use comparison images to understand per-class performance differences
4. Create final benchmark comparison report across all three models

The enhanced YOLOv5 pipeline ensures consistent, fair, and comprehensive evaluation suitable for academic or professional model comparison studies.