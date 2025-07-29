import sys
import os

# Add project root to PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torchvision
from torchvision.models.detection import SSD300_VGG16_Weights
from torchvision.models.detection.ssd import SSDClassificationHead
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from datasets.coco_voc import COCODetectionDataset
from torch.utils.data import DataLoader
from torchvision.transforms import transforms as T
from tqdm import tqdm
import json

# Config
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 21  # 20 Pascal VOC classes + background
COCO_JSON = "data/voc2012_coco.json"
IMAGE_DIR = "data/VOCdevkit/VOC2012/JPEGImages"
WEIGHTS_PATH = "outputs/ssd300_voc.pth"
RESULTS_PATH = "outputs/ssd_results.json"
BATCH_SIZE = 4

def main():
    print("[1] Loading model...")

    model = torchvision.models.detection.ssd300_vgg16(weights=SSD300_VGG16_Weights.COCO_V1)

    # Replace head with Pascal VOC head
    model.head.classification_head = SSDClassificationHead(
        in_channels=[512, 1024, 512, 256, 256, 256],
        num_anchors=[4, 6, 6, 6, 4, 4],
        num_classes=NUM_CLASSES
    )

    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    print("[2] Preparing dataset...")

    transform = T.Compose([
        T.Resize((300, 300)),
        T.ToTensor(),
        T.Normalize(mean=[0.48235, 0.45882, 0.40784], std=[0.229, 0.224, 0.225])  # matches SSD VGG16 backbone
    ])

    dataset = COCODetectionDataset(COCO_JSON, IMAGE_DIR, transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=lambda x: list(zip(*x)))

    print("[3] Running evaluation...")

    all_outputs = []
    coco = COCO(COCO_JSON)

    with torch.no_grad():
        for images, targets in tqdm(dataloader, desc="Evaluating"):
            images = [img.to(DEVICE) for img in images]
            outputs = model(images)

            for output, target in zip(outputs, targets):
                image_id = target["image_id"]
                boxes = output["boxes"].cpu().numpy()
                scores = output["scores"].cpu().numpy()
                labels = output["labels"].cpu().numpy()

                for box, score, label in zip(boxes, scores, labels):
                    x1, y1, x2, y2 = map(float, box)
                    all_outputs.append({
                        "image_id": int(image_id) if torch.is_tensor(image_id) else image_id,
                        "category_id": int(label + 1),  # if your model uses 0-based indexing
                        "bbox": [x1, y1, x2 - x1, y2 - y1],
                        "score": float(score)
                    })

    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(all_outputs, f)
    print(f"📝 Results saved to: {RESULTS_PATH}")

    print("[4] Running COCO mAP evaluation...")

    # Add missing fields for compatibility
    if 'info' not in coco.dataset:
        coco.dataset['info'] = {"description": "VOC to COCO converted dataset"}
    if 'licenses' not in coco.dataset:
        coco.dataset['licenses'] = [{"id": 1, "name": "Unknown", "url": ""}]
    coco.createIndex()

    coco_dt = coco.loadRes(RESULTS_PATH)
    coco_eval = COCOeval(coco, coco_dt, iouType='bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

if __name__ == "__main__":
    main()