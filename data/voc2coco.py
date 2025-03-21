'''
Copyright (C) 2025 co-pace GmbH (a subsidiary of Continental AG).
Licensed under the BSD-3-Clause License. 
@author: Georgii Mikriukov
'''

import sys
import os
module_path = os.path.abspath(os.path.join('../src/'))
if module_path not in sys.path:
    sys.path.append(module_path)

import argparse
import json

import numpy as np
from pycocotools import mask
from PIL import Image
from skimage import measure
from tqdm import tqdm


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, bytes):
            return obj.decode("utf-8")
        return super(NumpyEncoder, self).default(obj)


def get_label2id():
    """Create a dictionary mapping label names to IDs."""
    return {
        'aeroplane': 1,
        'bicycle': 2,
        'bird': 3,
        'boat': 4,
        'bottle': 5,
        'bus': 6,
        'car': 7,
        'cat': 8,
        'chair': 9,
        'cow': 10,
        'diningtable': 11,
        'dog': 12,
        'horse': 13,
        'motorbike': 14,
        'person': 15,
        'pottedplant': 16,
        'sheep': 17,
        'sofa': 18,
        'train': 19,
        'tvmonitor': 20
    }

def get_segmentation_from_mask(binary_mask, format="polygon", tolerance=2):
    """
    Convert binary mask to COCO segmentation format.

    Args:
        binary_mask (ndarray): 2D binary mask.
        format (str): 'polygon' or 'rle'.
        tolerance (int): Maximum distance for polygon approximation.

    Returns:
        Segmentation: Polygons or RLE based on the format.
    """
    if format == "polygon":
        padded_binary_mask = np.pad(binary_mask, pad_width=1, mode='constant', constant_values=0)
        contours = measure.find_contours(padded_binary_mask, 0.5)
        polygons = []
        for contour in contours:
            contour = np.flip(contour, axis=1)  # Convert to (x, y)
            contour = measure.approximate_polygon(contour, tolerance=tolerance)
            if len(contour) < 3:  # Skip invalid polygons
                continue
            polygons.append(contour.ravel().tolist())
        return polygons
    elif format == "rle":
        binary_mask = np.asfortranarray(binary_mask.astype(np.uint8))
        return mask.encode(binary_mask)
    else:
        raise ValueError(f"Unknown format: {format}")

def convert_voc_to_coco(image_list_path, image_dir, annotation_dir, seg_dir, output_json, segmentation_format="polygon"):
    label2id = get_label2id()

    output_dict = {
        "images": [],
        "annotations": [],
        "categories": []
    }

    # Add category info
    for label, label_id in label2id.items():
        output_dict["categories"].append({"id": label_id, "name": label, "supercategory": "none"})

    # Read image list
    with open(image_list_path, 'r') as f:
        image_list = f.read().splitlines()

    annotation_id = 1
    for image_id, image_name in enumerate(tqdm(image_list), 1):
        # Image info
        image_path = os.path.join(image_dir, f"{image_name}.jpg")
        img = Image.open(image_path)
        width, height = img.size
        output_dict["images"].append({
            "id": image_id,
            "file_name": f"{image_name}.jpg",
            "width": width,
            "height": height
        })

        # Segmentation mask
        seg_path = os.path.join(seg_dir, f"{image_name}.png")
        seg_mask = np.array(Image.open(seg_path))

        for class_id in np.unique(seg_mask):
            if class_id == 0:  # Background
                continue

            binary_mask = (seg_mask == class_id).astype(np.uint8)
            segmentation = get_segmentation_from_mask(binary_mask, format=segmentation_format)

            if not segmentation:
                continue

            rle = mask.encode(np.asfortranarray(binary_mask))
            bbox = mask.toBbox(rle).tolist()
            area = mask.area(rle).item()

            output_dict["annotations"].append({
                "id": annotation_id,
                "image_id": image_id,
                "category_id": class_id,
                "segmentation": segmentation,
                "area": area,
                "bbox": bbox,
                "iscrowd": 0
            })
            annotation_id += 1

    # Save COCO JSON
    with open(output_json, 'w') as f:
        json.dump(output_dict, f, indent=4, cls=NumpyEncoder)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Convert VOC dataset to COCO format.")
    parser.add_argument("--split", type=str, default="val", help="Dataset split (e.g., 'train', 'val').")
    parser.add_argument("--voc_root", type=str, default="./data/voc2012/VOCdevkit/VOC2012",
                        help="Root directory of the VOC dataset.")
    parser.add_argument("--segmentation_format", type=str, default="polygon",
                        choices=["polygon", "rle"], help="Segmentation format (polygon or rle).")
    parser.add_argument("--output_json", type=str, help="Output JSON file path.")

    args = parser.parse_args()

    image_list_path = os.path.join(args.voc_root, f"ImageSets/Segmentation/{args.split}.txt")
    image_dir = os.path.join(args.voc_root, "JPEGImages")
    annotation_dir = os.path.join(args.voc_root, "Annotations")
    seg_dir = os.path.join(args.voc_root, "SegmentationClass")
    output_json = os.path.join(args.voc_root, f"voc2012{args.split}_annotations.json")

    convert_voc_to_coco(
        image_list_path=image_list_path,
        image_dir=image_dir,
        annotation_dir=annotation_dir,
        seg_dir=seg_dir,
        output_json=output_json,
        segmentation_format=args.segmentation_format
    )
