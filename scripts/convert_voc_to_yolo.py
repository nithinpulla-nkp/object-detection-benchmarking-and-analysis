import os
import xml.etree.ElementTree as ET

VOC_CLASSES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]
CLASS_TO_IDX = {cls: idx for idx, cls in enumerate(VOC_CLASSES)}

def convert(size, box):
    dw = 1.0 / size[0]
    dh = 1.0 / size[1]
    x = (box[0] + box[2]) / 2.0 * dw
    y = (box[1] + box[3]) / 2.0 * dh
    w = (box[2] - box[0]) * dw
    h = (box[3] - box[1]) * dh
    return x, y, w, h

def convert_annotation(voc_dir, image_id, out_dir):
    in_file = os.path.join(voc_dir, 'Annotations', f'{image_id}.xml')
    out_file = os.path.join(out_dir, f'{image_id}.txt')

    tree = ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    with open(out_file, 'w') as f:
        for obj in root.iter('object'):
            cls = obj.find('name').text
            if cls not in CLASS_TO_IDX:
                continue
            cls_id = CLASS_TO_IDX[cls]
            xmlbox = obj.find('bndbox')
            b = (float(xmlbox.find('xmin').text), float(xmlbox.find('ymin').text),
                 float(xmlbox.find('xmax').text), float(xmlbox.find('ymax').text))
            bb = convert((w, h), b)
            f.write(f"{cls_id} {' '.join(map(str, bb))}\n")

if __name__ == "__main__":
    voc_root = "data/VOCdevkit/VOC2012"
    out_label_dir = "yolov5/VOCdevkit/VOC2012/labels"
    os.makedirs(out_label_dir, exist_ok=True)

    split_file = os.path.join(voc_root, "ImageSets", "Main", "trainval.txt")
    with open(split_file) as f:
        image_ids = [line.strip() for line in f.readlines()]

    for image_id in image_ids:
        convert_annotation(voc_root, image_id, out_label_dir)