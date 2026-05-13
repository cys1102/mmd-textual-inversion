from semantic_aug.few_shot_dataset import FewShotDataset
from semantic_aug.generative_augmentation import GenerativeAugmentation
from typing import Any, Tuple, Dict

import numpy as np
import torchvision.transforms as transforms
import torchvision
import torch
import glob
import os

from PIL import Image
from collections import defaultdict


DEFAULT_IMAGE_DIR = os.environ.get("PETS_DIR", "datasets/pets")


class PetsDataset(FewShotDataset):

    class_names = [
        'abyssinian',
        'american bulldog',
        'american pit bull terrier',
        'basset hound',
        'beagle',
        'bengal',
        'birman',
        'bombay',
        'boxer',
        'british shorthair',
        'chihuahua',
        'egyptian mau',
        'english cocker spaniel',
        'english setter',
        'german shorthaired',
        'great pyrenees',
        'havanese',
        'japanese chin',
        'keeshond',
        'leonberger',
        'maine coon',
        'miniature pinscher',
        'newfoundland',
        'persian',
        'pomeranian',
        'pug',
        'ragdoll',
        'russian blue',
        'saint bernard',
        'samoyed',
        'scottish terrier',
        'shiba inu',
        'siamese',
        'sphynx',
        'staffordshire bull terrier',
        'wheaten terrier',
        'yorkshire terrier']

    num_classes: int = len(class_names)

    def __init__(self, *args, split: str = "train", seed: int = 0, 
                 image_dir: str = DEFAULT_IMAGE_DIR, 
                 examples_per_class: int = None, 
                 generative_aug: GenerativeAugmentation = None, 
                 num_way: int = None,
                 synthetic_probability: float = 0.5,
                 use_randaugment: bool = False,
                 image_size: Tuple[int] = (256, 256), 
                 train_transform: transforms = None,
                 val_transform: transforms = None, 
                 **kwargs):

        super(PetsDataset, self).__init__(
            *args, examples_per_class=examples_per_class,
            synthetic_probability=synthetic_probability, 
            generative_aug=generative_aug, **kwargs)
        
        all_class_names = self.class_names.copy()
        rng = np.random.default_rng(seed)

        if num_way is not None:
            self.class_names = self.class_names[:num_way]
            self.num_classes = len(self.class_names)
        # if num_way < self.num_classes:
        #     self.class_names = rng.choice(all_class_names, size=num_way, replace=False)
        #     self.num_classes = len(self.class_names)

        class_to_images = defaultdict(list)

        for image_path in glob.glob(os.path.join(image_dir, "*.jpg")):
            class_name = " ".join(image_path.split("/")[-1].lower().replace("_", " ").split(" ")[:-1])
            if class_name in self.class_names:
                class_to_images[class_name].append(image_path)

        class_to_ids = {key: rng.permutation(
            len(class_to_images[key])) for key in self.class_names}
        
        class_to_ids = {key: np.array_split(class_to_ids[key], 2)[0 if split == "train" else 1] for key in self.class_names}

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
        if train_transform is None:
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

        if val_transform is None:
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
    
