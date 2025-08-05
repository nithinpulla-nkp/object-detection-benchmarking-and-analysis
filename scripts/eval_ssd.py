#!/usr/bin/env python
import sys
import os
import argparse
import json
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader
from tqdm import tqdm
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torchvision.models.detection import ssd300_vgg16, SSD300_VGG16_Weights
from torchvision.models.detection.ssd import SSDClassificationHead

# Add project root to PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from datasets.coco_voc import COCODetectionDataset

def get_transform():
    return T.Compose([
        T.Resize((300, 300)),
        T.ToTensor(),
        T.Normalize(mean=[0.48235, 0.45882, 0.40784], std=[0.229, 0.224, 0.225])
    ])

def collate_fn(batch):
    return tuple(zip(*batch))

def main():
    parser = argparse.ArgumentParser(description="Evaluate SSD300 model on Pascal VOC dataset")
    parser.add_argument('--coco-json', required=True, help='Path to COCO format JSON annotation file')
    parser.add_argument('--image-dir', required=True, help='Path to directory containing images')
    parser.add_argument('--weights', required=True, help='Path to trained model weights')
    parser.add_argument('--results', required=True, help='Path to save detection results JSON')
    parser.add_argument('--batch-size', type=int, default=4, help='Batch size for evaluation')
    parser.add_argument('--num-classes', type=int, default=21, help='Number of classes (including background)')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[1] Loading model on {device}...")

    model = ssd300_vgg16(weights=SSD300_VGG16_Weights.COCO_V1)
    model.head.classification_head = SSDClassificationHead(
        in_channels=[512, 1024, 512, 256, 256, 256],
        num_anchors=[4, 6, 6, 6, 4, 4],
        num_classes=args.num_classes
    )

    model.load_state_dict(torch.load(args.weights, map_location=device))
    model.to(device).eval()

    print("[2] Preparing dataset...")
    dataset = COCODetectionDataset(args.coco_json, args.image_dir, transform=get_transform())
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    print("[3] Running inference...")
    coco_gt = COCO(args.coco_json)
    if 'info' not in coco_gt.dataset:
        coco_gt.dataset['info'] = {}
    
    all_outputs = []
    with torch.no_grad():
        for images, targets in tqdm(dataloader, desc="Evaluating"):
            images = [img.to(device) for img in images]
            outputs = model(images)
            
            for output, target in zip(outputs, targets):
                img_id = target['image_id'].item()
                for box, score, label in zip(output['boxes'], output['scores'], output['labels']):
                    x1, y1, x2, y2 = box.cpu().tolist()
                    all_outputs.append({
                        "image_id": img_id,
                        "category_id": int(label) + 1,  # Convert 0-based to 1-based for COCO format
                        "bbox": [x1, y1, x2 - x1, y2 - y1],
                        "score": float(score)
                    })

    os.makedirs(os.path.dirname(args.results), exist_ok=True)
    print(f"[4] Saving {len(all_outputs)} detections to {args.results}")
    with open(args.results, 'w') as f:
        json.dump(all_outputs, f)

    print("[5] Computing COCO mAP...")
    coco_dt = coco_gt.loadRes(args.results)
    coco_eval = COCOeval(coco_gt, coco_dt, iouType='bbox')
    coco_eval.params.imgIds = sorted({d['image_id'] for d in all_outputs})
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

if __name__ == "__main__":
    main()