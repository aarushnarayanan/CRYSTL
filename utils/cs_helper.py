import torch
import json
import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision

class BottleDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.image_dir = os.path.join(root_dir, "images")
        with open(os.path.join(root_dir, "_annotations.coco.json")) as f:
            self.data = json.load(f)
        
        self.images = [img for img in self.data['images']]
        self.annotations = self.data['annotations']
        self.categories = {cat['id']: cat['name'] for cat in self.data['categories']}

    def __getitem__(self, idx):
        img_info = self.images[idx]
        img_path = os.path.join(self.image_dir, img_info['file_name'])
        img = Image.open(img_path).convert("RGB")
        
        anns = [ann for ann in self.annotations if ann['image_id'] == img_info['id']]
        boxes = []
        labels = []
        for ann in anns:
            x, y, w, h = ann['bbox']
            boxes.append([x, y, x+w, y+h])
            labels.append(ann['category_id'])
        
        return (
            torchvision.transforms.functional.to_tensor(img),
            {
                'boxes': torch.tensor(boxes, dtype=torch.float32),
                'labels': torch.tensor(labels, dtype=torch.int64)
            }
        )

    def __len__(self):
        return len(self.images)

def collate_fn(batch):
    return tuple(zip(*batch))