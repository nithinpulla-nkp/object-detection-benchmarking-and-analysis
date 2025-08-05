import os
import torch
from torch.utils.data import Dataset
from PIL import Image
from pycocotools.coco import COCO

class COCODetectionDataset(Dataset):
    def __init__(self, ann_file, img_folder, transform=None):
        print(f"🔍 Loading COCO file from: {ann_file}")
        self.coco = COCO(ann_file)
        self.img_folder = img_folder
        self.transform = transform
        self.ids = list(self.coco.imgs.keys())

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        img_info = self.coco.loadImgs(img_id)[0]
        
        path = os.path.join(self.img_folder, img_info['file_name'])
        img = Image.open(path).convert("RGB")

        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)

        boxes = []
        labels = []
        areas = []
        iscrowd = []

        for ann in anns:
            x, y, w, h = ann['bbox']
            boxes.append([x, y, x + w, y + h])
            labels.append(ann['category_id'] - 1)
            areas.append(ann['area'])
            iscrowd.append(ann.get('iscrowd', 0))

        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64),
            "image_id": torch.tensor([img_id], dtype=torch.int64),
            "area": torch.tensor(areas, dtype=torch.float32),
            "iscrowd": torch.tensor(iscrowd, dtype=torch.uint8),
        }

        if self.transform:
            img = self.transform(img)

        return img, target

    def __len__(self):
        return len(self.ids)