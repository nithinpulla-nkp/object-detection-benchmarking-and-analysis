#!/usr/bin/env python3
import os
import sys
import argparse
import xml.etree.ElementTree as ET
import logging
import json
from pathlib import Path
from shutil import copy2

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

VOC_CLASSES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]

CLASS_TO_IDX = {cls: idx for idx, cls in enumerate(VOC_CLASSES)}

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('conversion.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def convert_bbox(size, box):
    """Convert VOC bbox format (xmin, ymin, xmax, ymax) to YOLO format (x_center, y_center, width, height)"""
    dw = 1.0 / size[0]  # width normalization factor
    dh = 1.0 / size[1]  # height normalization factor
    
    x_center = (box[0] + box[2]) / 2.0 * dw
    y_center = (box[1] + box[3]) / 2.0 * dh
    width = (box[2] - box[0]) * dw
    height = (box[3] - box[1]) * dh
    
    return x_center, y_center, width, height

def convert_annotation(voc_xml_path, output_txt_path, logger):
    """Convert single VOC XML annotation to YOLO format"""
    try:
        tree = ET.parse(voc_xml_path)
        root = tree.getroot()
        
        # Get image dimensions
        size = root.find('size')
        width = int(size.find('width').text)
        height = int(size.find('height').text)
        
        annotations = []
        
        # Process each object
        for obj in root.iter('object'):
            cls_name = obj.find('name').text
            
            if cls_name not in CLASS_TO_IDX:
                logger.warning(f"Unknown class '{cls_name}' in {voc_xml_path}")
                continue
            
            cls_id = CLASS_TO_IDX[cls_name]
            
            # Get bounding box
            xmlbox = obj.find('bndbox')
            xmin = float(xmlbox.find('xmin').text)
            ymin = float(xmlbox.find('ymin').text)
            xmax = float(xmlbox.find('xmax').text)
            ymax = float(xmlbox.find('ymax').text)
            
            # Convert to YOLO format
            bbox = convert_bbox((width, height), (xmin, ymin, xmax, ymax))
            
            # Format: class_id x_center y_center width height
            annotation = f"{cls_id} {' '.join(f'{x:.6f}' for x in bbox)}"
            annotations.append(annotation)
        
        # Write YOLO format annotation
        with open(output_txt_path, 'w') as f:
            f.write('\n'.join(annotations))
            if annotations:  # Add newline at end if file is not empty
                f.write('\n')
        
        return len(annotations) > 0
        
    except Exception as e:
        logger.error(f"Error processing {voc_xml_path}: {e}")
        return False

def convert_dataset(voc_root, output_root, splits, logger):
    """Convert VOC dataset to YOLO format"""
    voc_path = Path(voc_root)
    output_path = Path(output_root)
    
    # Verify VOC structure
    required_dirs = ['Annotations', 'JPEGImages', 'ImageSets/Main']
    for required_dir in required_dirs:
        if not (voc_path / required_dir).exists():
            logger.error(f"Required VOC directory not found: {voc_path / required_dir}")
            return False
    
    conversion_stats = {
        'total_images': 0,
        'converted_images': 0,
        'failed_conversions': 0,
        'splits': {}
    }
    
    # Process each split (train, val, etc.)
    for split in splits:
        logger.info(f"Processing {split} split...")
        
        # Create output directories
        images_dir = output_path / 'images' / split
        labels_dir = output_path / 'labels' / split
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)
        
        # Read image IDs for this split
        split_file = voc_path / 'ImageSets' / 'Main' / f'{split}.txt'
        if not split_file.exists():
            logger.warning(f"Split file not found: {split_file}")
            continue
        
        with open(split_file, 'r') as f:
            image_ids = [line.strip() for line in f.readlines() if line.strip()]
        
        split_stats = {
            'total': len(image_ids),
            'converted': 0,
            'failed': 0
        }
        
        # Process each image
        for image_id in image_ids:
            conversion_stats['total_images'] += 1
            
            # Copy image file
            src_img = voc_path / 'JPEGImages' / f'{image_id}.jpg'
            dst_img = images_dir / f'{image_id}.jpg'
            
            if not src_img.exists():
                logger.warning(f"Image file not found: {src_img}")
                split_stats['failed'] += 1
                conversion_stats['failed_conversions'] += 1
                continue
            
            try:
                copy2(src_img, dst_img)
            except Exception as e:
                logger.error(f"Failed to copy image {src_img}: {e}")
                split_stats['failed'] += 1
                conversion_stats['failed_conversions'] += 1
                continue
            
            # Convert annotation
            src_xml = voc_path / 'Annotations' / f'{image_id}.xml'
            dst_txt = labels_dir / f'{image_id}.txt'
            
            if not src_xml.exists():
                logger.warning(f"Annotation file not found: {src_xml}")
                # Create empty annotation file
                dst_txt.touch()
                split_stats['converted'] += 1
                conversion_stats['converted_images'] += 1
                continue
            
            if convert_annotation(src_xml, dst_txt, logger):
                split_stats['converted'] += 1
                conversion_stats['converted_images'] += 1
            else:
                split_stats['failed'] += 1
                conversion_stats['failed_conversions'] += 1
        
        conversion_stats['splits'][split] = split_stats
        logger.info(f"Completed {split} split: {split_stats['converted']}/{split_stats['total']} converted")
    
    return conversion_stats

def create_yolo_config(output_root, dataset_name="Pascal VOC 2012"):
    """Create YOLOv5 data configuration file"""
    config = {
        'path': str(Path(output_root).resolve()),
        'train': 'images/train',
        'val': 'images/val',
        'nc': len(VOC_CLASSES),
        'names': VOC_CLASSES
    }
    
    config_path = Path(output_root) / f'yolov5_voc_config.yaml'
    
    import yaml
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    return config_path

def save_conversion_report(output_root, stats, logger):
    """Save conversion statistics report"""
    report = {
        'dataset': 'Pascal VOC 2012',
        'conversion_timestamp': str(Path().resolve()),
        'statistics': stats,
        'classes': {
            'count': len(VOC_CLASSES),
            'names': VOC_CLASSES
        }
    }
    
    report_path = Path(output_root) / 'conversion_report.json'
    
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    logger.info(f"Conversion report saved to {report_path}")

def main():
    parser = argparse.ArgumentParser(description="Convert Pascal VOC dataset to YOLO format")
    parser.add_argument("--voc-root", required=True, help="Path to VOC dataset root directory")
    parser.add_argument("--output-root", required=True, help="Output directory for YOLO format dataset")
    parser.add_argument("--splits", nargs='+', default=['train', 'val'], 
                       help="Dataset splits to convert (default: train val)")
    parser.add_argument("--create-config", action='store_true', 
                       help="Create YOLOv5 data configuration file")
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging()
    logger.info(f"Starting VOC to YOLO conversion with parameters: {vars(args)}")
    
    # Create output directory
    os.makedirs(args.output_root, exist_ok=True)
    
    # Convert dataset
    stats = convert_dataset(args.voc_root, args.output_root, args.splits, logger)
    
    if not stats:
        logger.error("Conversion failed!")
        return 1
    
    # Create YOLOv5 config if requested
    if args.create_config:
        config_path = create_yolo_config(args.output_root)
        logger.info(f"YOLOv5 config created at {config_path}")
    
    # Save conversion report
    save_conversion_report(args.output_root, stats, logger)
    
    # Print summary
    logger.info("=== Conversion Summary ===")
    logger.info(f"Total images processed: {stats['total_images']}")
    logger.info(f"Successfully converted: {stats['converted_images']}")
    logger.info(f"Failed conversions: {stats['failed_conversions']}")
    
    for split, split_stats in stats['splits'].items():
        logger.info(f"{split}: {split_stats['converted']}/{split_stats['total']} images")
    
    logger.info("VOC to YOLO conversion completed!")
    return 0

if __name__ == "__main__":
    sys.exit(main())