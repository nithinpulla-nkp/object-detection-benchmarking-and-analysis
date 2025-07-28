import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torchvision.transforms import ToTensor
from datasets.voc_dataset import VOCDetectionDataset

VOC_CLASSES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]

def visualize_sample(dataset, idx=None):
    if idx is None:
        idx = random.randint(0, len(dataset) - 1)

    image, target = dataset[idx]
    boxes = target["boxes"]
    labels = target["labels"]

    image_np = image.permute(1, 2, 0).numpy()

    fig, ax = plt.subplots(1, figsize=(8, 6))
    ax.imshow(image_np)

    for box, label in zip(boxes, labels):
        xmin, ymin, xmax, ymax = box.tolist()
        class_name = VOC_CLASSES[label]
        rect = patches.Rectangle(
            (xmin, ymin), xmax - xmin, ymax - ymin,
            linewidth=2, edgecolor='red', facecolor='none'
        )
        ax.add_patch(rect)
        ax.text(xmin, ymin - 5, class_name, color='yellow',
                fontsize=9, weight='bold', backgroundcolor='black')

    ax.axis('off')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    dataset = VOCDetectionDataset(
        root_dir="data/VOCdevkit/VOC2012",
        split="train",
        transforms=ToTensor()
    )

    # Show 5 random samples
    for _ in range(5):
        visualize_sample(dataset)