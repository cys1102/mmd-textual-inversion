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

DEFAULT_ROOT_DIR = os.environ.get("CUB_ROOT", "datasets/CUB_200_2011/CUB_200_2011")
DEFAULT_IMAGE_DIR = os.environ.get(
    "CUB_IMAGE_DIR", os.path.join(DEFAULT_ROOT_DIR, "images"))

class CUBDataset(FewShotDataset):

    class_names = \
        ['a Black footed Albatross', 'a Laysan Albatross', 'a Sooty Albatross', 'a Groove billed Ani', 'a Crested Auklet', 'a Least Auklet',
        'a Parakeet Auklet', 'a Rhinoceros Auklet', 'a Brewer Blackbird', 'a Red winged Blackbird', 'a Rusty Blackbird', 'a Yellow headed Blackbird',
        'a Bobolink', 'a Indigo Bunting', 'a Lazuli Bunting', 'a Painted Bunting', 'a Cardinal', 'a Spotted Catbird', 'a Gray Catbird',
        'a Yellow breasted Chat', 'a Eastern Towhee', 'a Chuck will Widow', 'a Brandt Cormorant', 'a Red faced Cormorant', 'a Pelagic Cormorant',
        'a Bronzed Cowbird', 'a Shiny Cowbird', 'a Brown Creeper', 'a American Crow', 'a Fish Crow', 'a Black billed Cuckoo', 'a Mangrove Cuckoo',
        'a Yellow billed Cuckoo', 'a Gray crowned Rosy Finch', 'a Purple Finch', 'a Northern Flicker', 'a Acadian Flycatcher', 'a Great Crested Flycatcher',
        'a Least Flycatcher', 'a Olive sided Flycatcher', 'a Scissor tailed Flycatcher', 'a Vermilion Flycatcher', 'a Yellow bellied Flycatcher',
        'a Frigatebird', 'a Northern Fulmar', 'a Gadwall', 'a American Goldfinch', 'a European Goldfinch', 'a Boat tailed Grackle', 'a Eared Grebe',
        'a Horned Grebe', 'a Pied billed Grebe', 'a Western Grebe', 'a Blue Grosbeak', 'a Evening Grosbeak', 'a Pine Grosbeak', 'a Rose breasted Grosbeak',
        'a Pigeon Guillemot', 'a California Gull', 'a Glaucous winged Gull', 'a Heermann Gull', 'a Herring Gull', 'a Ivory Gull', 'a Ring billed Gull',
        'a Slaty backed Gull', 'a Western Gull', 'a Anna Hummingbird', 'a Ruby throated Hummingbird', 'a Rufous Hummingbird',
        'a Green Violetear', 'a Long tailed Jaeger', 'a Pomarine Jaeger', 'a Blue Jay', 'a Florida Jay', 'a Green Jay', 'a Dark eyed Junco',
        'a Tropical Kingbird', 'a Gray Kingbird', 'a Belted Kingfisher', 'a Green Kingfisher', 'a Pied Kingfisher', 'a Ringed Kingfisher',
        'a White breasted Kingfisher', 'a Red legged Kittiwake', 'a Horned Lark', 'a Pacific Loon', 'a Mallard', 'a Western Meadowlark',
        'a Hooded Merganser', 'a Red breasted Merganser', 'a Mockingbird', 'a Nighthawk', 'a Clark Nutcracker', 'a White breasted Nuthatch',
        'a Baltimore Oriole', 'a Hooded Oriole', 'a Orchard Oriole', 'a Scott Oriole', 'a Ovenbird', 'a Brown Pelican', 'a White Pelican',
        'a Western Wood Pewee', 'a Sayornis', 'a American Pipit', 'a Whip poor Will', 'a Horned Puffin', 'a Common Raven', 'a White necked Raven',
        'a American Redstart', 'a Geococcyx', 'a Loggerhead Shrike', 'a Great Grey Shrike', 'a Baird Sparrow', 'a Black throated Sparrow',
        'a Brewer Sparrow', 'a Chipping Sparrow', 'a Clay colored Sparrow', 'a House Sparrow', 'a Field Sparrow', 'a Fox Sparrow',
        'a Grasshopper Sparrow', 'a Harris Sparrow', 'a Henslow Sparrow', 'a Le Conte Sparrow', 'a Lincoln Sparrow', 'a Nelson Sharp tailed Sparrow',
        'a Savannah Sparrow', 'a Seaside Sparrow', 'a Song Sparrow', 'a Tree Sparrow', 'a Vesper Sparrow', 'a White crowned Sparrow',
        'a White throated Sparrow', 'a Cape Glossy Starling', 'a Bank Swallow', 'a Barn Swallow', 'a Cliff Swallow', 'a Tree Swallow',
        'a Scarlet Tanager', 'a Summer Tanager', 'a Artic Tern', 'a Black Tern', 'a Caspian Tern', 'a Common Tern', 'a Elegant Tern',
        'a Forsters Tern', 'a Least Tern', 'a Green tailed Towhee', 'a Brown Thrasher', 'a Sage Thrasher', 'a Black capped Vireo',
        'a Blue headed Vireo', 'a Philadelphia Vireo', 'a Red eyed Vireo', 'a Warbling Vireo', 'a White eyed Vireo',
        'a Yellow throated Vireo', 'a Bay breasted Warbler', 'a Black and white Warbler', 'a Black throated Blue Warbler',
        'a Blue winged Warbler', 'a Canada Warbler', 'a Cape May Warbler', 'a Cerulean Warbler', 'a Chestnut sided Warbler',
        'a Golden winged Warbler', 'a Hooded Warbler', 'a Kentucky Warbler', 'a Magnolia Warbler', 'a Mourning Warbler',
        'a Myrtle Warbler', 'a Nashville Warbler', 'a Orange crowned Warbler', 'a Palm Warbler', 'a Pine Warbler', 'a Prairie Warbler',
        'a Prothonotary Warbler', 'a Swainson Warbler', 'a Tennessee Warbler', 'a Wilson Warbler', 'a Worm eating Warbler',
        'a Yellow Warbler', 'a Northern Waterthrush', 'a Louisiana Waterthrush', 'a Bohemian Waxwing', 'a Cedar Waxwing',
        'a American Three toed Woodpecker', 'a Pileated Woodpecker', 'a Red bellied Woodpecker', 'a Red cockaded Woodpecker',
        'a Red headed Woodpecker', 'a Downy Woodpecker', 'a Bewick Wren', 'a Cactus Wren', 'a Carolina Wren', 'a House Wren',
        'a Marsh Wren', 'a Rock Wren', 'a Winter Wren', 'a Common Yellowthroat']
    
    num_classes = len(class_names)

    def __init__(self, *args, split: str = "train", seed: int = 0, 
                root_dir: str = DEFAULT_ROOT_DIR,
                image_dir: str = DEFAULT_IMAGE_DIR, 
                num_way: int = 5,
                examples_per_class: int = None, 
                generative_aug: GenerativeAugmentation = None, 
                synthetic_probability: float = 0.5,
                use_randaugment: bool = False,
                image_size: Tuple[int] = (256, 256), **kwargs):

        super(CUBDataset, self).__init__(
            *args, num_way=num_way, examples_per_class=examples_per_class,
            synthetic_probability=synthetic_probability, 
            generative_aug=generative_aug, **kwargs)

        all_class_names = self.class_names.copy()

        rng = np.random.default_rng(seed)
        self.class_names = rng.choice(all_class_names, size=num_way, replace=False)
        self.num_classes = len(self.class_names)

        self.is_train = split == "train"

        self.class_id_to_name = {}
        for id, name in enumerate(all_class_names):
            if name in self.class_names:
                self.class_id_to_name[id] = name
        # name_to_class_id = {v: k for k, v in class_id_to_name.items() if v in self.class_names}

        images_path = {}
        with open(os.path.join(root_dir, 'images.txt')) as f:
            for line in f:
                image_id, path = line.split()
                images_path[image_id] = os.path.join(image_dir, path)

        class_ids = {}
        with open(os.path.join(root_dir, 'image_class_labels.txt')) as f:
            for line in f:
                image_id, class_id = line.split()
                class_ids[image_id] = class_id

        # class_to_ids = {v: k for k, v in class_ids.items()}

        data_id = []
        with open(os.path.join(root_dir, 'train_test_split.txt')) as f:
            for line in f:
                image_id, is_train = line.split()
                if self.is_train == int(is_train) and int(class_ids[image_id]) in self.class_id_to_name.keys():
                    data_id.append(image_id)

        self.class_to_images = defaultdict(list)
        for image_id, path in images_path.items():
            if image_id in data_id:
                class_id = class_ids[image_id]
                class_name = self.class_id_to_name[int(class_id)]
                self.class_to_images[class_name].append(path)

        if examples_per_class:
            for class_name in self.class_to_images:
                self.class_to_images[class_name] = rng.choice(self.class_to_images[class_name], size=examples_per_class, replace=False)
        
        self.all_images = [img for imgs in self.class_to_images.values() for img in imgs]
        # self.class_to_images = {
        #     key: [class_to_images[key][i] for i in ids] 
        #     for key, ids in class_to_ids.items()}

        # self.all_images = sum([
        #     self.class_to_images[key] 
        #     for key in self.class_names], [])

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
        print("Selected classes: ", self.class_names)
        print("Number of images: ", len(self.all_images))

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
    
