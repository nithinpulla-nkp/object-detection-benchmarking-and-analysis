# SSD300 File Interaction Flow

## Overview
This diagram shows how the Jupyter notebook interacts with the Python scripts in the `src/` folder during training and evaluation.

## File Structure
```
SSD300/
├── notebooks/
│   └── ssd300_enhanced_pipeline.ipynb    # Main interface
├── src/
│   ├── coco_voc.py                       # Dataset loader
│   ├── train_ssd.py                      # Training script
│   ├── eval_ssd.py                       # Evaluation script
│   ├── voc2coco.py                       # Data converter
│   ├── voc_dataset.py                    # VOC dataset loader
│   └── visualize_dataset.py              # Visualization tools
└── outputs/                              # Generated files
```

## File Interaction Flow

```
┌─────────────────────────────────────────────────────────────────┐
│           📓 ssd300_enhanced_pipeline.ipynb                     │
│                    (Main Control Center)                        │
└─────────────────────────────────────────────────────────────────┘
                                 │
                    ┌─────────────┼─────────────┐
                    │             │             │
                    ▼             ▼             ▼
          
    📁 DATA PREPARATION     🧠 MODEL TRAINING    🔍 EVALUATION
                            
┌─────────────────────┐   ┌─────────────────────┐   ┌─────────────────────┐
│    NOTEBOOK CELL    │   │    NOTEBOOK CELL    │   │    NOTEBOOK CELL    │
│                     │   │                     │   │                     │
│ # Convert VOC→COCO  │   │ # Train SSD300      │   │ # Evaluate Model    │
│ os.system(          │   │ from SSD300.src     │   │ from SSD300.src     │
│   "python src/      │   │ import coco_voc     │   │ import coco_voc     │
│    voc2coco.py"     │   │                     │   │                     │
│ )                   │   │ dataset =           │   │ dataset =           │
└─────────────────────┘   │ COCODetectionDataset│   │ COCODetectionDataset│
          │               │ model = ssd300_vgg16│   │ model.load_state_   │
          ▼               │                     │   │ dict(torch.load())  │
┌─────────────────────┐   │ # Training loop     │   └─────────────────────┘
│  📄 voc2coco.py     │   │ for epoch in range: │             │
│                     │   │   for batch:        │             ▼
│ def convert_xmls_   │   │     loss = model()  │   ┌─────────────────────┐
│ to_cocojson()       │   │     loss.backward() │   │  📄 eval_ssd.py     │
│                     │   │     optimizer.step()│   │                     │
│ # Reads XML files   │   └─────────────────────┘   │ def evaluate_model()│
│ # Converts to COCO  │             │               │                     │
│ # Saves JSON        │             ▼               │ # Loads model       │
└─────────────────────┘   ┌─────────────────────┐   │ # Runs inference    │
          │               │  📄 coco_voc.py     │   │ # Computes metrics  │
          ▼               │                     │   │ # Saves results     │
┌─────────────────────┐   │ class COCODetection │   └─────────────────────┘
│  📊 voc2012_coco.   │   │ Dataset:            │             │
│      json           │   │                     │             ▼
│                     │   │ def __init__()      │   ┌─────────────────────┐
│ # COCO format       │   │ def __getitem__()   │   │  📊 Results Files   │
│ # annotations       │   │ def __len__()       │   │                     │
└─────────────────────┘   │                     │   │ predictions.json    │
                          │ # Loads images      │   │ metrics.json        │
                          │ # Loads annotations │   │ training_log.json   │
                          │ # Applies transforms│   │ visualizations.png  │
                          └─────────────────────┘   └─────────────────────┘
```

## Detailed Flow Steps

### 1. **Data Preparation Flow**
```
Notebook Cell 4 → os.system() → src/voc2coco.py → data/voc2012_coco.json
```

### 2. **Training Flow**
```
Notebook Cell 7 → import src/coco_voc.py → COCODetectionDataset() → Training Loop
```

### 3. **Evaluation Flow**
```
Notebook Cell 9 → import src/coco_voc.py → evaluate_model() → Results JSON
```

## Key Import Statements in Notebook

```python
# In notebook cells:
sys.path.append(os.path.join(os.getcwd(), '..', '..'))
from SSD300.src.coco_voc import COCODetectionDataset  # Dataset loading
```

## File Dependencies

```
📓 ssd300_enhanced_pipeline.ipynb
    │
    ├── 📄 src/voc2coco.py          (called via os.system)
    ├── 📄 src/coco_voc.py          (imported directly)
    ├── 📄 src/eval_ssd.py          (functions imported)
    └── 📄 src/visualize_dataset.py (functions imported)

📄 src/coco_voc.py
    ├── PIL (Image loading)
    ├── torch (Tensor operations)
    └── pycocotools.coco (COCO format)

📄 src/voc2coco.py
    ├── xml.etree.ElementTree (XML parsing)
    └── json (JSON output)
```

## Data Flow Between Files

```
VOC XML Files → voc2coco.py → COCO JSON → coco_voc.py → PyTorch Tensors → Model
                                   ↑
                              Notebook manages this flow
```

## Output Files Generated

```
outputs/
├── models/
│   ├── ssd300_final.pth          # Final trained model
│   └── ssd300_epoch_*.pth        # Epoch checkpoints
├── predictions/
│   ├── ssd300_predictions.json   # Full evaluation results
│   └── ssd300_comparison_results.json # Comparison images
├── logs/
│   ├── ssd300_training_*.log     # Training logs
│   └── training_log.json         # Training metrics
├── visualizations/
│   ├── training_analysis.png     # Training dashboard
│   └── comparison_predictions.png # Prediction visualizations
└── ssd300_final_report.json      # Comprehensive report
```

This flow allows you to control everything from the notebook while leveraging the modular Python scripts for specific tasks.