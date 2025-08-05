#!/usr/bin/env python3
import os
import sys
import argparse
import subprocess
import yaml
import json
import logging
from pathlib import Path
from datetime import datetime

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('training.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def create_yolo_config(output_path, data_dir):
    """Create YOLOv5 configuration file for Pascal VOC"""
    config = {
        'path': str(Path(data_dir).resolve()),
        'train': 'images/train',
        'val': 'images/val',
        'nc': 20,
        'names': [
            'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
            'bus', 'car', 'cat', 'chair', 'cow',
            'diningtable', 'dog', 'horse', 'motorbike', 'person',
            'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
        ]
    }
    
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    return output_path

def train_yolov5(args, logger):
    """Train YOLOv5 model with specified parameters"""
    # Get YOLOv5 path
    yolov5_path = Path(__file__).parent.parent.parent / "yolov5"
    
    if not yolov5_path.exists():
        logger.error(f"YOLOv5 directory not found at {yolov5_path}")
        return False
    
    # Create data config
    config_path = Path(args.output_dir) / "yolov5_voc_config.yaml"
    create_yolo_config(config_path, args.data_dir)
    
    # Prepare training command
    train_cmd = [
        sys.executable,
        str(yolov5_path / "train.py"),
        "--img", str(args.img_size),
        "--batch", str(args.batch_size),
        "--epochs", str(args.epochs),
        "--data", str(config_path),
        "--weights", args.weights,
        "--name", args.experiment_name,
        "--project", str(Path(args.output_dir) / "runs"),
        "--exist-ok"
    ]
    
    if args.device:
        train_cmd.extend(["--device", args.device])
    
    logger.info(f"Starting YOLOv5 training with command: {' '.join(train_cmd)}")
    
    try:
        # Run training
        result = subprocess.run(
            train_cmd,
            cwd=str(yolov5_path),
            capture_output=True,
            text=True,
            timeout=args.timeout
        )
        
        if result.returncode == 0:
            logger.info("Training completed successfully")
            logger.info(f"Training output: {result.stdout}")
            return True
        else:
            logger.error(f"Training failed with return code {result.returncode}")
            logger.error(f"Error output: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        logger.error(f"Training timed out after {args.timeout} seconds")
        return False
    except Exception as e:
        logger.error(f"Training failed with exception: {e}")
        return False

def save_training_report(args, success, logger):
    """Save training report with parameters and results"""
    report = {
        "timestamp": datetime.now().isoformat(),
        "model": "YOLOv5",
        "success": success,
        "parameters": {
            "data_dir": args.data_dir,
            "output_dir": args.output_dir,
            "img_size": args.img_size,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "weights": args.weights,
            "device": args.device,
            "experiment_name": args.experiment_name
        }
    }
    
    report_path = Path(args.output_dir) / f"yolov5_training_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    logger.info(f"Training report saved to {report_path}")

def main():
    parser = argparse.ArgumentParser(description="Train YOLOv5 on Pascal VOC dataset")
    parser.add_argument("--data-dir", required=True, help="Path to YOLO format dataset directory")
    parser.add_argument("--output-dir", default="outputs", help="Output directory for models and logs")
    parser.add_argument("--img-size", type=int, default=640, help="Input image size")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--weights", default="yolov5s.pt", help="Initial weights path")
    parser.add_argument("--device", default="", help="CUDA device (e.g., '0' or '0,1,2,3' or 'cpu')")
    parser.add_argument("--experiment-name", default="yolov5_voc", help="Experiment name")
    parser.add_argument("--timeout", type=int, default=7200, help="Training timeout in seconds")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Setup logging
    logger = setup_logging()
    logger.info(f"Starting YOLOv5 training with parameters: {vars(args)}")
    
    # Train model
    success = train_yolov5(args, logger)
    
    # Save report
    save_training_report(args, success, logger)
    
    if success:
        logger.info("YOLOv5 training completed successfully!")
        return 0
    else:
        logger.error("YOLOv5 training failed!")
        return 1

if __name__ == "__main__":
    sys.exit(main())