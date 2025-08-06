# Object Detection Benchmarking and Analysis

A comprehensive benchmarking suite comparing three popular object detection models on Pascal VOC 2012 dataset: **SSD300**, **Faster R-CNN**, and **YOLOv5**.

## 🎯 Project Overview

This repository provides a fair and comprehensive comparison of three state-of-the-art object detection models, designed for both academic research and practical evaluation. Each model is implemented in standalone Jupyter notebooks that can run independently in Kaggle, Google Colab, or local environments.

### Why This Project?

- **Fair Comparison**: All models use identical datasets, evaluation metrics, and test images
- **Kaggle Compatible**: Standalone notebooks that work without external dependencies
- **Comprehensive Analysis**: Beyond just accuracy - includes speed, model size, and visualization
- **Research Ready**: Professional reporting and structured outputs for academic use
- **Easy to Use**: Clear documentation and simple setup process

## 🏗️ Repository Structure

```
object-detection-benchmarking-and-analysis/
├── README.md                           # This file - project overview
├── comparison_images.json              # Standardized test images (20 images)
├── SSD300/                            # SSD300 model implementation
│   ├── notebooks/
│   │   └── ssd300_standalone_kaggle.ipynb     # Complete SSD300 pipeline
│   └── README.md                      # SSD300 specific documentation
├── FasterRCNN/                        # Faster R-CNN model implementation  
│   ├── notebooks/
│   │   └── fasterrcnn_standalone_kaggle.ipynb # Complete Faster R-CNN pipeline
│   └── README.md                      # Faster R-CNN specific documentation
└── YOLOv5_custom/                     # YOLOv5 model implementation
    ├── notebooks/
    │   ├── yolov5_enhanced_comprehensive.ipynb # Enhanced YOLOv5 pipeline
    │   └── yolov5.ipynb               # Original basic YOLOv5 notebook
    └── README.md                      # YOLOv5 specific documentation
```

## 🤖 Models Compared

### 1. SSD300 (Single Shot MultiBox Detector)
- **Architecture**: VGG16 backbone + Multi-scale feature maps
- **Input Size**: 300×300
- **Strength**: Fast inference, good for real-time applications
- **Parameters**: ~26M
- **Notebook**: `SSD300/notebooks/ssd300_standalone_kaggle.ipynb`

### 2. Faster R-CNN
- **Architecture**: ResNet50 + FPN backbone + Two-stage detection
- **Input Size**: Variable (maintains aspect ratio)
- **Strength**: High accuracy, excellent for precision-critical applications
- **Parameters**: ~41M
- **Notebook**: `FasterRCNN/notebooks/fasterrcnn_standalone_kaggle.ipynb`

### 3. YOLOv5s
- **Architecture**: CSPDarknet53 backbone + PANet neck
- **Input Size**: 640×640
- **Strength**: Balanced speed/accuracy, modern architecture
- **Parameters**: ~7.2M
- **Notebook**: `YOLOv5_custom/notebooks/yolov5_enhanced_comprehensive.ipynb`

## 📊 Evaluation Framework

All models are evaluated using identical criteria:

### Performance Metrics
- **mAP@0.5**: Mean Average Precision at IoU threshold 0.5
- **mAP@0.5:0.95**: Mean Average Precision averaged over IoU thresholds 0.5-0.95
- **Precision**: Overall precision across all classes
- **Recall**: Overall recall across all classes

### Speed Benchmarking
- **Preprocessing Time**: Image loading and preprocessing
- **Inference Time**: Model forward pass
- **Postprocessing Time**: NMS and result processing
- **Total FPS**: End-to-end frames per second

### Standardized Testing
- **20 Carefully Selected Images**: Covering all Pascal VOC classes
- **Varying Difficulty Levels**: Easy, medium, and hard detection scenarios
- **Consistent Evaluation**: Same images tested across all models

## 🚀 Quick Start

### Option 1: Run in Kaggle (Recommended)

1. **For SSD300:**
   - Open new Kaggle notebook
   - Add "Pascal VOC 2012" dataset 
   - Upload `SSD300/notebooks/ssd300_standalone_kaggle.ipynb`
   - Run all cells

2. **For Faster R-CNN:**
   - Open new Kaggle notebook
   - Add "Pascal VOC 2012" dataset
   - Upload `FasterRCNN/notebooks/fasterrcnn_standalone_kaggle.ipynb`
   - Run all cells

3. **For YOLOv5:**
   - Open new Kaggle notebook
   - Add "Pascal VOC 2012" dataset
   - Upload `YOLOv5_custom/notebooks/yolov5_enhanced_comprehensive.ipynb`
   - Run all cells

### Option 2: Run Locally

```bash
# Clone repository
git clone <repository-url>
cd object-detection-benchmarking-and-analysis

# Download Pascal VOC 2012 dataset
# Update paths in notebook configuration cells

# Run any model notebook in Jupyter
jupyter notebook SSD300/notebooks/ssd300_standalone_kaggle.ipynb
```

### Required Dataset

**Pascal VOC 2012**: Download from [Kaggle](https://www.kaggle.com/datasets/huanghanchina/pascal-voc-2012) or [official source](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/)

The dataset contains:
- **20 Object Classes**: aeroplane, bicycle, bird, boat, bottle, bus, car, cat, chair, cow, diningtable, dog, horse, motorbike, person, pottedplant, sheep, sofa, train, tvmonitor
- **Training Images**: ~5,717 images
- **Validation Images**: ~5,823 images

## 📈 Expected Results

### Typical Performance (5 epochs training)

| Model | mAP@0.5 | mAP@0.5:0.95 | FPS | Model Size | Parameters |
|-------|---------|--------------|-----|------------|------------|
| SSD300 | 0.60-0.70 | 0.35-0.45 | 8-12 | ~100MB | ~26M |
| Faster R-CNN | 0.65-0.75 | 0.45-0.55 | 6-10 | ~160MB | ~41M |
| YOLOv5s | 0.60-0.75 | 0.40-0.50 | 15-25 | ~14MB | ~7.2M |

*Results vary based on training epochs, hardware, and specific configuration*

### Key Insights
- **Fastest**: YOLOv5s (highest FPS, smallest model)
- **Most Accurate**: Faster R-CNN (highest mAP scores)
- **Most Balanced**: SSD300 (good balance of speed and accuracy)

## 🛠️ Features

### Complete Pipeline for Each Model
- ✅ **Standalone Notebooks**: No external .py files needed
- ✅ **Data Preprocessing**: Automatic format conversion
- ✅ **Model Training**: With checkpointing and resume capability
- ✅ **Comprehensive Evaluation**: COCO-style metrics
- ✅ **Speed Benchmarking**: Detailed performance analysis
- ✅ **Professional Visualizations**: Training curves, predictions, comparisons
- ✅ **Structured Outputs**: Organized results and reports

### Comparison Framework
- ✅ **Standardized Test Images**: Same 20 images across all models
- ✅ **Consistent Metrics**: Identical evaluation criteria
- ✅ **Fair Training**: Same dataset splits and training procedures
- ✅ **Professional Reporting**: JSON reports and summaries

## 📋 Output Structure

Each model generates organized outputs:

```
model_outputs/
├── models/           # Trained model weights and checkpoints
├── metrics/          # Performance metrics and speed benchmarks
├── visualizations/   # Training plots and prediction samples
├── comparisons/      # Standardized test results
├── reports/          # Comprehensive experiment reports
└── logs/             # Training logs and debugging info
```

## 💻 Hardware Requirements

### Minimum Requirements
- **RAM**: 8GB+
- **Storage**: 5GB+ free space
- **GPU**: Not required but highly recommended

### Recommended Setup
- **GPU**: NVIDIA GPU with 4GB+ VRAM (for faster training)
- **RAM**: 16GB+
- **Storage**: 10GB+ free space

### Cloud Options
- **Kaggle**: Free GPU access, perfect for this project
- **Google Colab**: Free GPU with some limitations
- **AWS/GCP**: Paid options with more control

## 🔧 Customization

### Training Parameters
Each notebook has a `CONFIG` section where you can modify:

```python
CONFIG = {
    'num_epochs': 5,        # Increase for better results
    'batch_size': 4,        # Adjust based on GPU memory
    'learning_rate': 0.001, # Model-specific defaults
    'device': 'cuda',       # 'cuda' or 'cpu'
    # ... other parameters
}
```

### Dataset Paths
For local or different cloud environments:

```python
CONFIG = {
    'voc_root': '/path/to/VOCdevkit/VOC2012',  # Update this path
    'output_dir': './outputs',                  # Update output location
    # ... other settings
}
```

## 📚 Documentation

- **Main README** (this file): Project overview and quick start
- **SSD300/README.md**: Detailed SSD300 documentation and usage
- **FasterRCNN/README.md**: Detailed Faster R-CNN documentation and usage  
- **YOLOv5_custom/README.md**: Detailed YOLOv5 documentation and usage

Each model's README contains:
- Architecture details
- Training configuration options
- Troubleshooting guide
- Performance optimization tips

## 🤝 Contributing

This is a research/educational project. Contributions welcome:
- Bug fixes and improvements
- Additional model implementations
- Enhanced visualization features
- Documentation improvements

## 📄 License

This project is for educational and research purposes. Please check individual model licenses:
- SSD: Original implementation license
- Faster R-CNN: Detectron2/PyTorch license
- YOLOv5: Ultralytics license

## 🎓 Citation

If you use this benchmarking suite in your research, please cite:

```
@misc{object-detection-benchmark,
  title={Object Detection Benchmarking Suite: SSD300, Faster R-CNN, and YOLOv5},
  year={2024},
  note={Comprehensive comparison framework for Pascal VOC 2012}
}
```

## 📞 Support

For questions or issues:
1. Check the model-specific READMEs
2. Review the troubleshooting sections
3. Open an issue with detailed error information

## 🎯 Use Cases

### Academic Research
- **Baseline Comparisons**: Established benchmarks for new model development
- **Ablation Studies**: Compare architectural choices
- **Performance Analysis**: Speed vs accuracy trade-offs

### Industry Applications  
- **Model Selection**: Choose the right model for your use case
- **Performance Validation**: Verify model performance claims
- **Deployment Planning**: Understand computational requirements

### Educational Purposes
- **Learning Object Detection**: Hands-on experience with major architectures
- **Understanding Trade-offs**: Speed, accuracy, and resource consumption
- **Practical Implementation**: From theory to working code

---

## 📁 Dataset Setup (Local Development)

If running locally instead of Kaggle:

1. **Download** the official Pascal VOC 2012 dataset:
   ```bash
   # If you're on macOS and don't have wget, first install it using Homebrew:
   # brew install wget
   wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
   ```

2. **Extract the Dataset**  
   After downloading, create a `data/` directory (if it doesn't exist) and extract the contents into it using:
   ```bash
   mkdir -p data/
   tar -xvf VOCtrainval_11-May-2012.tar -C data/
   ```

3. **Verify Directory Structure**  
   Once extracted, your folder structure should look like this:
   ```
   data/
   └── VOCdevkit/
       └── VOC2012/
           ├── JPEGImages/      ← Contains image files
           ├── Annotations/     ← XML annotation files
           ├── ImageSets/       ← Text files defining train/val/test splits
           ├── SegmentationClass/
           └── ...
   ```

4. **Exclude Dataset from Git**  
   The dataset is large and should not be committed to version control. Add the following lines to your `.gitignore`:
   ```
   # .gitignore
   data/VOCdevkit/
   *.tar
   ```

---

**Ready to start?** Pick a model and run its notebook in Kaggle! Each notebook is self-contained and will guide you through the complete process from data loading to final results. 🚀