import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


from datasets.voc_dataset import VOCDetectionDataset
from torchvision.transforms import ToTensor

# Replace with the actual relative path from project root
dataset = VOCDetectionDataset(
    root_dir="data/VOCdevkit/VOC2012",  # ✅ Use relative path
    split="train",
    transforms=ToTensor()
)

image, target = dataset[0]
print(f"Image shape: {image.shape}")           # torch.Size([3, H, W])
print(f"Boxes shape: {target['boxes'].shape}") # [N, 4]
print(f"Labels: {target['labels']}")           # Tensor of label indices