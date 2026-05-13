from semantic_aug.few_shot_dataset import FewShotDataset
from semantic_aug.generative_augmentation import GenerativeAugmentation
from typing import Any, Tuple, Dict

import numpy as np
import torchvision.transforms as transforms
import torchvision
import torch
import glob
import os

from scipy.io import loadmat
from PIL import Image
from collections import defaultdict


DEFAULT_IMAGE_DIR = os.environ.get("FLOWERS102_DIR", "datasets/flowers102")


class Flowers102Dataset(FewShotDataset):

    class_names = [
        'pink primrose',
        'hard-leaved pocket orchid',
        'canterbury bells',
        'sweet pea',
        'english marigold',
        'tiger lily',
        'moon orchid',
        'bird of paradise',
        'monkshood',
        'globe thistle',
        'snapdragon',
        "colt's foot",
        'king protea',
        'spear thistle',
        'yellow iris',
        'globe-flower',
        'purple coneflower',
        'peruvian lily',
        'balloon flower',
        'giant white arum lily',
        'fire lily',
        'pincushion flower',
        'fritillary',
        'red ginger',
        'grape hyacinth',
        'corn poppy',
        'prince of wales feathers',
        'stemless gentian',
        'artichoke',
        'sweet william',
        'carnation',
        'garden phlox',
        'love in the mist',
        'mexican aster',
        'alpine sea holly',
        'ruby-lipped cattleya',
        'cape flower',
        'great masterwort',
        'siam tulip',
        'lenten rose',
        'barbeton daisy',
        'daffodil',
        'sword lily',
        'poinsettia',
        'bolero deep blue',
        'wallflower',
        'marigold',
        'buttercup',
        'oxeye daisy',
        'common dandelion',
        'petunia',
        'wild pansy',
        'primula',
        'sunflower',
        'pelargonium',
        'bishop of llandaff',
        'gaura',
        'geranium',
        'orange dahlia',
        'pink-yellow dahlia',
        'cautleya spicata',
        'japanese anemone',
        'black-eyed susan',
        'silverbush',
        'californian poppy',
        'osteospermum',
        'spring crocus',
        'bearded iris',
        'windflower',
        'tree poppy',
        'gazania',
        'azalea',
        'water lily',
        'rose',
        'thorn apple',
        'morning glory',
        'passion flower',
        'lotus',
        'toad lily',
        'anthurium',
        'frangipani',
        'clematis',
        'hibiscus',
        'columbine',
        'desert-rose',
        'tree mallow',
        'magnolia',
        'cyclamen ',
        'watercress',
        'canna lily',
        'hippeastrum ',
        'bee balm',
        'ball moss',
        'foxglove',
        'bougainvillea',
        'camellia',
        'mallow',
        'mexican petunia',
        'bromelia',
        'blanket flower',
        'trumpet creeper',
        'blackberry lily']

    num_classes: int = len(class_names)

    def __init__(self, *args, split: str = "train", seed: int = 0, 
                 image_dir: str = DEFAULT_IMAGE_DIR, 
                #  num_way: int = 102,
                 examples_per_class: int = None, 
                 generative_aug: GenerativeAugmentation = None, 
                 num_way: int = None,
                 synthetic_probability: float = 0.5,
                 use_randaugment: bool = False,
                 image_size: Tuple[int] = (256, 256),
                 train_transform: transforms = None,
                 val_transform: transforms = None, 
                 **kwargs):

        super(Flowers102Dataset, self).__init__(
            *args, examples_per_class=examples_per_class,
            synthetic_probability=synthetic_probability, 
            generative_aug=generative_aug, **kwargs)
        
        all_class_names = self.class_names.copy()
        rng = np.random.default_rng(seed)
        if num_way is not None:
            self.class_names = self.class_names[:num_way]
            self.num_classes = len(self.class_names)

        imagelabels = loadmat(os.path.join(image_dir, "imagelabels.mat"))["labels"][0]
        image_files = sorted(list(glob.glob(os.path.join(image_dir, "jpg/*.jpg"))))

        class_to_images = defaultdict(list)

        for image_idx, image_path in enumerate(image_files):
            class_name = all_class_names[imagelabels[image_idx] - 1]
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
