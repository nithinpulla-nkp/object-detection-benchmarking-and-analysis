#!/usr/bin/env python3
import os
import sys
import argparse
import subprocess
import json
import logging
from pathlib import Path
from datetime import datetime

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

VOC_CLASSES = [
    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
    'bus', 'car', 'cat', 'chair', 'cow',
    'diningtable', 'dog', 'horse', 'motorbike', 'person',
    'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
]

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('evaluation.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def evaluate_yolov5(args, logger):
    """Run YOLOv5 validation and return metrics"""
    # Get YOLOv5 path
    yolov5_path = Path(__file__).parent.parent.parent / "yolov5"
    
    if not yolov5_path.exists():
        logger.error(f"YOLOv5 directory not found at {yolov5_path}")
        return None
    
    # Prepare validation command
    val_cmd = [
        sys.executable,
        str(yolov5_path / "val.py"),
        "--data", args.data_config,
        "--weights", args.weights,
        "--img", str(args.img_size),
        "--batch", str(args.batch_size),
        "--conf", str(args.conf_threshold),
        "--iou", str(args.iou_threshold),
        "--project", str(Path(args.output_dir) / "validation"),
        "--name", args.experiment_name,
        "--save-txt", "--save-conf", "--save-json",
        "--exist-ok"
    ]
    
    if args.device:
        val_cmd.extend(["--device", args.device])
    
    logger.info(f"Running YOLOv5 validation with command: {' '.join(val_cmd)}")
    
    try:
        # Run validation
        result = subprocess.run(
            val_cmd,
            cwd=str(yolov5_path),
            capture_output=True,
            text=True,
            timeout=args.timeout
        )
        
        if result.returncode == 0:
            logger.info("Validation completed successfully")
            logger.info(f"Validation output: {result.stdout}")
            
            # Parse metrics from output
            metrics = parse_validation_output(result.stdout, logger)
            return metrics
        else:
            logger.error(f"Validation failed with return code {result.returncode}")
            logger.error(f"Error output: {result.stderr}")
            return None
            
    except subprocess.TimeoutExpired:
        logger.error(f"Validation timed out after {args.timeout} seconds")
        return None
    except Exception as e:
        logger.error(f"Validation failed with exception: {e}")
        return None

def parse_validation_output(output, logger):
    """Parse validation metrics from YOLOv5 output"""
    metrics = {}
    lines = output.split('\n')
    
    for line in lines:
        line = line.strip()
        if 'mAP@.5' in line and 'mAP@.5:.95' in line:
            # Extract mAP values
            parts = line.split()
            try:
                # Find mAP values in the output
                for i, part in enumerate(parts):
                    if part == 'all':
                        # Typical format: "all  5972  32.1k  0.XXX  0.XXX  0.XXX  0.XXX"
                        if i + 4 < len(parts):
                            metrics['mAP_0.5'] = float(parts[i + 3])
                            metrics['mAP_0.5_0.95'] = float(parts[i + 4])
                            break
            except ValueError as e:
                logger.warning(f"Could not parse mAP values from line: {line}")
        
        elif 'P' in line and 'R' in line and 'mAP@.5' in line:
            # Alternative format parsing
            try:
                parts = line.split()
                if len(parts) >= 4:
                    metrics['precision'] = float(parts[1])
                    metrics['recall'] = float(parts[2])
                    metrics['mAP_0.5'] = float(parts[3])
                    if len(parts) >= 5:
                        metrics['mAP_0.5_0.95'] = float(parts[4])
            except ValueError:
                continue
    
    logger.info(f"Parsed metrics: {metrics}")
    return metrics

def run_inference_on_comparison_images(args, logger):
    """Run inference on comparison images for model comparison"""
    comparison_file = Path(__file__).parent.parent.parent / "comparison_images.json"
    
    if not comparison_file.exists():
        logger.warning("Comparison images file not found, skipping comparison inference")
        return None
    
    # Load comparison images
    with open(comparison_file, 'r') as f:
        comparison_data = json.load(f)
    
    # Get YOLOv5 path
    yolov5_path = Path(__file__).parent.parent.parent / "yolov5"
    
    results = []
    
    for img_data in comparison_data.get('images', []):
        img_path = Path(args.data_dir) / "images" / img_data['filename']
        
        if not img_path.exists():
            logger.warning(f"Comparison image not found: {img_path}")
            continue
        
        # Run inference
        detect_cmd = [
            sys.executable,
            str(yolov5_path / "detect.py"),
            "--weights", args.weights,
            "--source", str(img_path),
            "--img", str(args.img_size),
            "--conf", str(args.conf_threshold),
            "--iou", str(args.iou_threshold),
            "--project", str(Path(args.output_dir) / "comparison_inference"),
            "--name", f"{args.experiment_name}_{img_data['image_id']}",
            "--save-txt", "--save-conf",
            "--exist-ok"
        ]
        
        try:
            result = subprocess.run(
                detect_cmd,
                cwd=str(yolov5_path),
                capture_output=True,
                text=True,
                timeout=60
            )
            
            if result.returncode == 0:
                # Parse detection results
                detection_result = {
                    'image_id': img_data['image_id'],
                    'filename': img_data['filename'],
                    'expected_objects': img_data.get('expected_objects', []),
                    'difficulty': img_data.get('difficulty', 'unknown'),
                    'inference_success': True
                }
                results.append(detection_result)
                logger.info(f"Inference completed for {img_data['filename']}")
            else:
                logger.warning(f"Inference failed for {img_data['filename']}")
                
        except subprocess.TimeoutExpired:
            logger.warning(f"Inference timeout for {img_data['filename']}")
        except Exception as e:
            logger.warning(f"Inference error for {img_data['filename']}: {e}")
    
    return results

def save_evaluation_report(args, metrics, comparison_results, logger):
    """Save evaluation report with metrics and results"""
    report = {
        "timestamp": datetime.now().isoformat(),
        "model": "YOLOv5",
        "weights": args.weights,
        "parameters": {
            "data_config": args.data_config,
            "img_size": args.img_size,
            "batch_size": args.batch_size,
            "conf_threshold": args.conf_threshold,
            "iou_threshold": args.iou_threshold,
            "device": args.device
        },
        "metrics": metrics,
        "comparison_results": comparison_results
    }
    
    report_path = Path(args.output_dir) / f"yolov5_evaluation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    logger.info(f"Evaluation report saved to {report_path}")

def main():
    parser = argparse.ArgumentParser(description="Evaluate YOLOv5 on Pascal VOC dataset")
    parser.add_argument("--weights", required=True, help="Path to trained model weights")
    parser.add_argument("--data-config", required=True, help="Path to YOLOv5 data configuration file")
    parser.add_argument("--data-dir", required=True, help="Path to dataset directory")
    parser.add_argument("--output-dir", default="outputs", help="Output directory for results")
    parser.add_argument("--img-size", type=int, default=640, help="Input image size")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for evaluation")
    parser.add_argument("--conf-threshold", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou-threshold", type=float, default=0.45, help="IoU threshold for NMS")
    parser.add_argument("--device", default="", help="CUDA device")
    parser.add_argument("--experiment-name", default="yolov5_eval", help="Experiment name")
    parser.add_argument("--timeout", type=int, default=1800, help="Evaluation timeout in seconds")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Setup logging
    logger = setup_logging()
    logger.info(f"Starting YOLOv5 evaluation with parameters: {vars(args)}")
    
    # Run evaluation
    metrics = evaluate_yolov5(args, logger)
    
    # Run comparison inference
    comparison_results = run_inference_on_comparison_images(args, logger)
    
    # Save report
    save_evaluation_report(args, metrics, comparison_results, logger)
    
    if metrics:
        logger.info("YOLOv5 evaluation completed successfully!")
        logger.info(f"Final metrics: {metrics}")
        return 0
    else:
        logger.error("YOLOv5 evaluation failed!")
        return 1

if __name__ == "__main__":
    sys.exit(main())