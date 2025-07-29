import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision.models.detection import ssd300_vgg16
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.transforms import functional as F
from datasets.coco_voc import COCODetectionDataset
import torchvision.transforms as T
from tqdm import tqdm

# -------- Config --------
BATCH_SIZE = 4
EPOCHS = 10
NUM_CLASSES = 21  # 20 classes + background
COCO_JSON = "data/voc2012_coco.json"
IMG_DIR = "data/VOCdevkit/VOC2012/JPEGImages"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "outputs/ssd300_voc.pth"
# ------------------------

def get_transform():
    return T.Compose([
        T.Resize((300, 300)),
        T.ToTensor(),
    ])

def collate_fn(batch):
    return tuple(zip(*batch))

def main():
    os.makedirs("outputs", exist_ok=True)

    print("[1] Loading dataset...")
    dataset = COCODetectionDataset(IMG_DIR, COCO_JSON, transforms=get_transform())
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

    print("[2] Initializing SSD300 model...")
    model = ssd300_vgg16(pretrained=True)
    model.head.classification_head.num_classes = NUM_CLASSES
    model.to(DEVICE)

    from torchvision.models.detection.ssd import SSDClassificationHead

    model.head.classification_head = SSDClassificationHead(
        in_channels=[512, 1024, 512, 256, 256, 256],     # one per feature map
        num_anchors=[4, 6, 6, 6, 4, 4],                   # anchors per feature map
        num_classes=NUM_CLASSES                            # 21 for Pascal VOC
    )

    print("[3] Starting training...")
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9, weight_decay=5e-4)

    for epoch in range(EPOCHS):
        total_loss = 0.0
        for images, targets in tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            images = list(img.to(DEVICE) for img in images)
            targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            total_loss += losses.item()

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

        print(f"Epoch [{epoch+1}/{EPOCHS}] Loss: {total_loss:.4f}")

    print(f"[4] Saving model to: {MODEL_PATH}")
    torch.save(model.state_dict(), MODEL_PATH)

if __name__ == "__main__":
    main()