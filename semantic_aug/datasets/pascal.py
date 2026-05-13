from semantic_aug.few_shot_dataset import FewShotDataset
from semantic_aug.generative_augmentation import GenerativeAugmentation
from typing import Any, Tuple, Dict

import numpy as np
import torchvision.transforms as transforms
import torch
import os

from PIL import Image
from collections import defaultdict


PASCAL_DIR = os.environ.get("PASCAL_DIR", "datasets/pascal")

TRAIN_IMAGE_SET = os.path.join(
    PASCAL_DIR, "ImageSets/Segmentation/train.txt")
VAL_IMAGE_SET = os.path.join(
    PASCAL_DIR, "ImageSets/Segmentation/val.txt")

DEFAULT_IMAGE_DIR = os.path.join(PASCAL_DIR, "JPEGImages")
DEFAULT_LABEL_DIR = os.path.join(PASCAL_DIR, "SegmentationClass")
DEFAULT_INSTANCE_DIR = os.path.join(PASCAL_DIR, "SegmentationObject")


class PASCALDataset(FewShotDataset):

    class_names = ['airplane', 'bicycle', 'bird', 'boat', 'bottle', 
        'bus', 'car', 'cat', 'chair', 'cow', 'dining table', 'dog', 
        'horse', 'motorcycle', 'person', 'potted plant', 'sheep', 
        'sofa', 'train', 'television']

    num_classes: int = len(class_names)

    def __init__(self, *args, split: str = "train", seed: int = 0, 
                 train_image_set: str = TRAIN_IMAGE_SET, 
                 val_image_set: str = VAL_IMAGE_SET, 
                 image_dir: str = DEFAULT_IMAGE_DIR, 
                 label_dir: str = DEFAULT_LABEL_DIR, 
                 instance_dir: str = DEFAULT_INSTANCE_DIR, 
                #  num_way: int = 20,
                 examples_per_class: int = None, 
                 generative_aug: GenerativeAugmentation = None, 
                 synthetic_probability: float = 0.5,
                 use_randaugment: bool = False,
                 image_size: Tuple[int] = (256, 256), **kwargs):

        super(PASCALDataset, self).__init__(
            *args, examples_per_class=examples_per_class,
            synthetic_probability=synthetic_probability, 
            generative_aug=generative_aug, **kwargs)

        all_class_names = self.class_names.copy()
        rng = np.random.default_rng(seed)
        # if num_way < self.num_classes:
        #     self.class_names = rng.choice(all_class_names, size=num_way, replace=False)
        #     self.num_classes = len(self.class_names)

        image_set = {"train": train_image_set, "val": val_image_set}[split]

        with open(image_set, "r") as f:
            image_set_lines = [x.strip() for x in f.readlines()]

        class_to_images = defaultdict(list)
        class_to_annotations = defaultdict(list)

        for image_id in image_set_lines:

            labels = os.path.join(label_dir, image_id + ".png")
            instances = os.path.join(instance_dir, image_id + ".png")

            labels = np.asarray(Image.open(labels))
            instances = np.asarray(Image.open(instances))

            instance_ids, pixel_loc, counts = np.unique(
                instances, return_index=True, return_counts=True)

            counts[0] = counts[-1] = 0  # remove background

            argmax_index = counts.argmax()

            mask = np.equal(instances, instance_ids[argmax_index])
            class_name = all_class_names[
                labels.flat[pixel_loc[argmax_index]] - 1]

            if class_name in self.class_names:
                class_to_images[class_name].append(
                    os.path.join(image_dir, image_id + ".jpg"))
                class_to_annotations[class_name].append(dict(mask=mask))

        class_to_ids = {key: rng.permutation(
            len(class_to_images[key])) for key in self.class_names}

        if examples_per_class is not None:
            class_to_ids = {key: ids[:examples_per_class] 
                            for key, ids in class_to_ids.items()}

        self.class_to_images = {
            key: [class_to_images[key][i] for i in ids] 
            for key, ids in class_to_ids.items()}

        self.class_to_annotations = {
            key: [class_to_annotations[key][i] for i in ids] 
            for key, ids in class_to_ids.items()}

        self.all_images = sum([
            self.class_to_images[key] 
            for key in self.class_names], [])

        self.all_annotations = sum([
            self.class_to_annotations[key] 
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
