#!/usr/bin/env python
import sys
import os
import argparse
import torch
from torch.utils.data import DataLoader
from torchvision.models.detection import ssd300_vgg16, SSD300_VGG16_Weights
from torchvision.models.detection.ssd import SSDClassificationHead
import torchvision.transforms as T
from tqdm import tqdm
from torch.optim import Adam

# Add project root to PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
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
    parser = argparse.ArgumentParser(description="Train SSD300 model on Pascal VOC dataset")
    parser.add_argument('--coco-json', required=True, help='Path to COCO format JSON annotation file')
    parser.add_argument('--image-dir', required=True, help='Path to directory containing images')
    parser.add_argument('--output-path', required=True, help='Path to save trained model weights')
    parser.add_argument('--batch-size', type=int, default=4, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=5, help='Number of training epochs')
    parser.add_argument('--num-classes', type=int, default=21, help='Number of classes (including background)')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    print(f"[1] Using device: {device}")
    print("[2] Loading dataset...")
    dataset = COCODetectionDataset(args.coco_json, args.image_dir, transform=get_transform())
    print(f"    Dataset size: {len(dataset)}")
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)

    print("[3] Initializing SSD300 model...")
    model = ssd300_vgg16(weights=SSD300_VGG16_Weights.COCO_V1)
    model.head.classification_head = SSDClassificationHead(
        in_channels=[512, 1024, 512, 256, 256, 256],
        num_anchors=[4, 6, 6, 6, 4, 4],
        num_classes=args.num_classes
    )
    model.to(device)

    print("[4] Starting training...")
    model.train()
    optimizer = Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        total_loss = 0.0
        for images, targets in tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}"):
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            total_loss += losses.item()

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

        print(f"    Epoch {epoch+1}/{args.epochs}  loss={total_loss:.4f}")

    print(f"[5] Saving model to: {args.output_path}")
    torch.save(model.state_dict(), args.output_path)

if __name__ == "__main__":
    main()