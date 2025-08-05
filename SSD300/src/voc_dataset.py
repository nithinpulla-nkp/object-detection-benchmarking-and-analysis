# voc_dataset.py

import os
import xml.etree.ElementTree as ET
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

VOC_CLASSES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]
CLASS_TO_IDX = {cls: idx for idx, cls in enumerate(VOC_CLASSES)}

class VOCDetectionDataset(Dataset):
    def __init__(self, root_dir, split="train", transforms=None):
        self.root = root_dir
        self.image_dir = os.path.join(root_dir, "JPEGImages")
        self.annotation_dir = os.path.join(root_dir, "Annotations")
        split_file = os.path.join(root_dir, "ImageSets", "Main", f"{split}.txt")
        with open(split_file) as f:
            self.ids = [line.strip() for line in f.readlines()]
        self.transforms = transforms

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        image_id = self.ids[idx]
        image_path = os.path.join(self.image_dir, f"{image_id}.jpg")
        ann_path = os.path.join(self.annotation_dir, f"{image_id}.xml")

        # Load image
        image = Image.open(image_path).convert("RGB")

        # Parse XML
        tree = ET.parse(ann_path)
        root = tree.getroot()

        boxes = []
        labels = []
        for obj in root.findall("object"):
            label = obj.find("name").text.lower().strip()
            if label not in CLASS_TO_IDX:
                continue
            labels.append(CLASS_TO_IDX[label])

            bbox = obj.find("bndbox")
            xmin = float(bbox.find("xmin").text)
            ymin = float(bbox.find("ymin").text)
            xmax = float(bbox.find("xmax").text)
            ymax = float(bbox.find("ymax").text)
            boxes.append([xmin, ymin, xmax, ymax])

        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([idx])
        }

        if self.transforms:
            image = self.transforms(image)

        return image, target