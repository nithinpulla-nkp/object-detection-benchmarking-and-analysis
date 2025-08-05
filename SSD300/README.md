# SSD300 Object Detection

This directory contains the complete implementation of SSD300 (Single Shot MultiBox Detector) for object detection on Pascal VOC 2012 dataset.

## Directory Structure

```
SSD300/
├── src/                          # Source code
│   ├── train_ssd.py             # Training script
│   ├── eval_ssd.py              # Evaluation script
│   ├── coco_voc.py              # COCO dataset loader
│   ├── voc_dataset.py           # VOC dataset loader
│   ├── voc2coco.py              # VOC to COCO converter
│   └── visualize_dataset.py     # Dataset visualization
├── notebooks/                    # Jupyter notebooks
│   └── ssd300_enhanced_pipeline.ipynb  # Complete training pipeline
└── README.md                     # This file
```

## Features

### 🧠 Model Architecture
- **SSD300** with VGG16 backbone
- Pre-trained on COCO, fine-tuned on Pascal VOC
- 21 classes (20 VOC classes + background)
- Multi-scale feature maps for detection

### 📊 Training Pipeline
- **Enhanced logging** with real-time progress tracking
- **Comprehensive visualizations** of training metrics
- **Automatic checkpointing** after each epoch
- **Performance monitoring** with loss curves and timing

### 🎯 Evaluation System
- **COCO metrics** (mAP @ different IoU thresholds)
- **Per-class Average Precision** analysis
- **Inference speed benchmarking**
- **Comparison image testing** for model comparison

### 🎨 Visualization Features
- **Training progress dashboard** with multiple metrics
- **Prediction visualizations** on test images
- **Performance comparison plots**
- **Real-time logging and monitoring**

## Quick Start

### 1. Training via Notebook (Recommended)
```bash
cd SSD300/notebooks/
jupyter notebook ssd300_enhanced_pipeline.ipynb
```

### 2. Training via Scripts
```bash
# Train model
python src/train_ssd.py \
  --coco-json ../../data/voc2012_coco.json \
  --image-dir ../../data/VOCdevkit/VOC2012/JPEGImages \
  --output-path outputs/models/ssd300_voc.pth \
  --epochs 10 --batch-size 4

# Evaluate model
python src/eval_ssd.py \
  --coco-json ../../data/voc2012_coco.json \
  --image-dir ../../data/VOCdevkit/VOC2012/JPEGImages \
  --weights outputs/models/ssd300_voc.pth \
  --results outputs/predictions/ssd300_results.json
```

## Configuration

### Training Parameters
- **Epochs**: 5 (default, adjust based on your needs)
- **Batch Size**: 4 (adjust based on GPU memory)
- **Learning Rate**: 1e-4
- **Image Size**: 300x300 pixels
- **Optimizer**: Adam

### Evaluation Parameters
- **Confidence Threshold**: 0.3
- **NMS Threshold**: 0.45
- **COCO Metrics**: mAP @ IoU 0.5:0.95, 0.5, 0.75

## Output Files

After training and evaluation, the following files will be generated:

### Models
- `outputs/models/ssd300_final.pth` - Final trained model
- `outputs/models/ssd300_epoch_*.pth` - Epoch checkpoints

### Predictions
- `outputs/predictions/ssd300_predictions.json` - Full evaluation results
- `outputs/predictions/ssd300_comparison_results.json` - Comparison image results

### Logs and Reports
- `outputs/logs/ssd300_training_*.log` - Training logs
- `outputs/logs/training_log.json` - Training metrics
- `outputs/ssd300_final_report.json` - Comprehensive final report

### Visualizations
- `outputs/visualizations/training_analysis.png` - Training dashboard
- `outputs/visualizations/comparison_predictions.png` - Prediction visualizations

## Model Comparison

This implementation is designed to work with the comparison framework:

1. **Comparison Images**: Uses `../../comparison_images.json` for standardized testing
2. **Consistent Metrics**: Outputs compatible with Faster R-CNN and YOLOv5 results
3. **Standardized Format**: All outputs follow the same structure for easy comparison

## Performance Expectations

### Typical Results (Pascal VOC 2012)
- **mAP @ IoU=0.5**: ~0.4-0.6 (depending on training epochs)
- **Inference Speed**: ~30-50 FPS (GPU dependent)
- **Model Size**: ~100MB
- **Training Time**: ~5-10 minutes per epoch (GPU dependent)

### Hardware Requirements
- **GPU**: Recommended (CUDA compatible)
- **RAM**: 8GB+ recommended
- **Storage**: 2GB+ for dataset and outputs

## Troubleshooting

### Common Issues
1. **CUDA Out of Memory**: Reduce batch size
2. **Slow Training**: Ensure GPU is being used
3. **Low mAP**: Increase training epochs or adjust learning rate
4. **Import Errors**: Check that project root is in Python path

### Debug Mode
Enable verbose logging by setting logging level to DEBUG in the notebook.

## Integration with Other Models

This SSD300 implementation is part of a larger comparison framework:

- **Faster R-CNN**: `../FasterRCNN/`
- **YOLOv5**: `../YOLOv5_custom/`
- **Comparison Tool**: `../../comparison_images.json`

All models use the same test images and evaluation metrics for fair comparison.