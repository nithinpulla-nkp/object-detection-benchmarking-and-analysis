import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from datasets.coco_voc import COCODetectionDataset

COCO_JSON = "data/voc2012_coco.json"
IMG_DIR = "data/VOCdevkit/VOC2012/JPEGImages"

assert os.path.exists(COCO_JSON), f"COCO JSON file not found: {COCO_JSON}"

dataset = COCODetectionDataset(COCO_JSON, IMG_DIR)

print(f"Total images listed in JSON: {len(dataset)}")

missing = []
for i in range(len(dataset)):
    try:
        _ = dataset[i]
    except Exception as e:
        print(f"[{i}] Missing or broken image: {e}")
        missing.append(i)

print(f"\n Total valid images: {len(dataset) - len(missing)} / {len(dataset)}")