from semantic_aug.few_shot_dataset import FewShotDataset
from semantic_aug.generative_augmentation import GenerativeAugmentation
from typing import Any, Tuple, Dict

import numpy as np
import torchvision.transforms as transforms
import torchvision
import torch
import glob
import os
import re

from PIL import Image
from collections import defaultdict

DEFAULT_ROOT_DIR = os.environ.get("DTD_ROOT", "datasets/dtd")
DEFAULT_IMAGE_DIR = os.environ.get(
    "DTD_IMAGE_DIR", os.path.join(DEFAULT_ROOT_DIR, "images"))

def split_values(s):
    # This regex matches the first sequence of digits and then any following text.
    match = re.match(r'(\d+)\s*(.*)', s)
    first_number = match.group(1)
    other = match.group(2)
    return first_number, other
    
class DTDDataset(FewShotDataset):

    class_names = \
        ["banded", "blotchy", "braided", "bubbly", "bumpy", "chequered", "cobwebbed", "cracked", "crosshatched", "crystalline",
        "dotted", "fibrous", "flecked", "freckled", "frilly", "gauzy", "grid", "grooved", "honeycombed", "interlaced", "knitted",
        "lacelike", "lined", "marbled", "matted", "meshed", "paisley", "perforated", "pitted", "pleated", "polka-dotted", "porous",
        "potholed", "scaly", "smeared", "spiralled", "sprinkled", "stained", "stratified", "striped", "studded", "swirly", "veined",
        "waffled", "woven", "wrinkled", "zigzagged"]

    
    num_classes = len(class_names)

    def __init__(self, *args, split: str = "train", seed: int = 0, 
                root_dir: str = DEFAULT_ROOT_DIR,
                image_dir: str = DEFAULT_IMAGE_DIR, 
                # num_way: int = 47,
                examples_per_class: int = None, 
                generative_aug: GenerativeAugmentation = None, 
                synthetic_probability: float = 0.5,
                use_randaugment: bool = False,
                image_size: Tuple[int] = (256, 256), **kwargs):

        super(DTDDataset, self).__init__(
            *args, examples_per_class=examples_per_class,
            synthetic_probability=synthetic_probability, 
            generative_aug=generative_aug, **kwargs)

        all_class_names = self.class_names.copy()
        rng = np.random.default_rng(seed)
        
        split_dirs = {
            "train": [f'labels/train{seed + 1}.txt', f'labels/val{seed + 1}.txt'],
            "val": [f'labels/val{seed + 1}.txt']}[split]

        class_to_images = defaultdict(list)
        for split_dir in split_dirs:
            with open(os.path.join(root_dir, split_dir)) as f:
                for line in f:
                    class_name, _ = line.split('/')
                    class_to_images[class_name].append(os.path.join(image_dir, line.strip()))
                    
        class_to_ids = {key: rng.permutation(
            len(class_to_images[key])) for key in self.class_names}

        if examples_per_class is not None:
            class_to_ids = {key: ids[:examples_per_class] 
                            for key, ids in class_to_ids.items()}

        self.class_to_images = {
            key: [class_to_images[key][i] for i in ids] 
            for key, ids in class_to_ids.items()}

        self.all_images = sum([
            self.class_to_images[key] 
            for key in self.class_names], [])

        self.all_labels = [i for i, key in enumerate(
            self.class_names) for _ in self.class_to_images[key]]

        self.class_dict = {class_name: f"class{i}" for i, class_name in enumerate(self.class_names)}


        if use_randaugment: train_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.RandAugment(),
            transforms.ToTensor(),
            transforms.ConvertImageDtype(torch.float),
            transforms.Lambda(lambda x: x.expand(3, *image_size)),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], 
                                  std=[0.5, 0.5, 0.5])
        ])

        else: train_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15.0),
            transforms.ToTensor(),
            transforms.ConvertImageDtype(torch.float),
            transforms.Lambda(lambda x: x.expand(3, *image_size)),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], 
                                  std=[0.5, 0.5, 0.5])
        ])

        val_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.ConvertImageDtype(torch.float),
            transforms.Lambda(lambda x: x.expand(3, *image_size)),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], 
                                  std=[0.5, 0.5, 0.5])
        ])

        self.transform = {"train": train_transform, "val": val_transform}[split]
        print("Num classes: ", self.num_classes)
        # print("Selected classes: ", self.class_names)
        print("Number of images: ", len(self.all_images))
        # print("Number of way: ", self.num_way)

    def __len__(self):
        
        return len(self.all_images)

    def get_image_by_idx(self, idx: int) -> Image.Image:

        return Image.open(self.all_images[idx]).convert('RGB')

    def get_label_by_idx(self, idx: int) -> int:

        return self.all_labels[idx]
    
    def get_metadata_by_idx(self, idx: int) -> dict:
        name = self.class_names[self.all_labels[idx]]
        label = self.class_dict[name]
        other_labels = self.get_other_labels(name)
        return dict(label=label, name=name, other_labels=other_labels)

    def get_other_labels(self, current_name: str) -> list:
        other_class_names = [name for name in self.class_names if name != current_name]
        other_labels = [self.class_dict[name] for name in other_class_names]
        return other_labels
    
