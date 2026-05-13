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


DEFAULT_IMAGE_DIR = "/projects/rsalakhugroup/datasets/flowers102"


class CarsDataset(FewShotDataset):

    cars_names = ['a FIAT 500 Convertible 2012', 'a Ferrari FF Coupe 2012', 'a Ferrari California Convertible 2012', 'a Ferrari 458 Italia Convertible 2012', 'a Ferrari 458 Italia Coupe 2012', 'a Fisker Karma Sedan 2012', 'a Ford F-450 Super Duty Crew Cab 2012', 'a Ford Mustang Convertible 2007', 'a Ford Freestar Minivan 2007', 'a Ford Expedition EL SUV 2009', 'a Aston Martin Virage Convertible 2012', 'a Ford Edge SUV 2012', 'a Ford Ranger SuperCab 2011', 'a Ford GT Coupe 2006', 'a Ford F-150 Regular Cab 2012', 'a Ford F-150 Regular Cab 2007', 'a Ford Focus Sedan 2007', 'a Ford E-Series Wagon Van 2012', 'a Ford Fiesta Sedan 2012', 'a GMC Terrain SUV 2012', 'a GMC Savana Van 2012', 'a Aston Martin Virage Coupe 2012', 'a GMC Yukon Hybrid SUV 2012', 'a GMC Acadia SUV 2012', 'a GMC Canyon Extended Cab 2012', 'a Geo Metro Convertible 1993', 'a HUMMER H3T Crew Cab 2010', 'a HUMMER H2 SUT Crew Cab 2009', 'a Honda Odyssey Minivan 2012', 'a Honda Odyssey Minivan 2007', 'a Honda Accord Coupe 2012', 'a Honda Accord Sedan 2012', 'a Audi RS 4 Convertible 2008', 'a Hyundai Veloster Hatchback 2012', 'a Hyundai Santa Fe SUV 2012', 'a Hyundai Tucson SUV 2012', 'a Hyundai Veracruz SUV 2012', 'a Hyundai Sonata Hybrid Sedan 2012', 'a Hyundai Elantra Sedan 2007', 'a Hyundai Accent Sedan 2012', 'a Hyundai Genesis Sedan 2012', 'a Hyundai Sonata Sedan 2012', 'a Hyundai Elantra Touring Hatchback 2012', 'a Audi A5 Coupe 2012', 'a Hyundai Azera Sedan 2012', 'a Infiniti G Coupe IPL 2012', 'a Infiniti QX56 SUV 2011', 'a Isuzu Ascender SUV 2008', 'a Jaguar XK XKR 2012', 'a Jeep Patriot SUV 2012', 'a Jeep Wrangler SUV 2012', 'a Jeep Liberty SUV 2012', 'a Jeep Grand Cherokee SUV 2012', 'a Jeep Compass SUV 2012', 'a Audi TTS Coupe 2012', 'a Lamborghini Reventon Coupe 2008', 'a Lamborghini Aventador Coupe 2012', 'a Lamborghini Gallardo LP 570-4 Superleggera 2012', 'a Lamborghini Diablo Coupe 2001', 'a Land Rover Range Rover SUV 2012', 'a Land Rover LR2 SUV 2012', 'a Lincoln Town Car Sedan 2011', 'a MINI Cooper Roadster Convertible 2012', 'a Maybach Landaulet Convertible 2012', 'a Mazda Tribute SUV 2011', 'a Audi R8 Coupe 2012', 'a McLaren MP4-12C Coupe 2012', 'a Mercedes-Benz 300-Class Convertible 1993', 'a Mercedes-Benz C-Class Sedan 2012', 'a Mercedes-Benz SL-Class Coupe 2009', 'a Mercedes-Benz E-Class Sedan 2012', 'a Mercedes-Benz S-Class Sedan 2012', 'a Mercedes-Benz Sprinter Van 2012', 'a Mitsubishi Lancer Sedan 2012', 'a Nissan Leaf Hatchback 2012', 'a Nissan NV Passenger Van 2012', 'a Audi V8 Sedan 1994', 'a Nissan Juke Hatchback 2012', 'a Nissan 240SX Coupe 1998', 'a Plymouth Neon Coupe 1999', 'a Porsche Panamera Sedan 2012', 'a Ram C/V Cargo Van Minivan 2012', 'a Rolls-Royce Phantom Drophead Coupe Convertible 2012', 'a Rolls-Royce Ghost Sedan 2012', 'a Rolls-Royce Phantom Sedan 2012', 'a Scion xD Hatchback 2012', 'a Spyker C8 Convertible 2009', 'a Audi 100 Sedan 1994', 'a Spyker C8 Coupe 2009', 'a Suzuki Aerio Sedan 2007', 'a Suzuki Kizashi Sedan 2012', 'a Suzuki SX4 Hatchback 2012', 'a Suzuki SX4 Sedan 2012', 'a Tesla Model S Sedan 2012', 'a Toyota Sequoia SUV 2012', 'a Toyota Camry Sedan 2012', 'a Toyota Corolla Sedan 2012', 'a Toyota 4Runner SUV 2012', 'a Audi 100 Wagon 1994', 'a Volkswagen Golf Hatchback 2012', 'a Volkswagen Golf Hatchback 1991', 'a Volkswagen Beetle Hatchback 2012', 'a Volvo C30 Hatchback 2012', 'a Volvo 240 Sedan 1993', 'a Volvo XC90 SUV 2007', 'a smart fortwo Convertible 2012', 'a Audi TT Hatchback 2011', 'a AM General Hummer SUV 2000', 'a Audi S6 Sedan 2011', 'a Audi S5 Convertible 2012', 'a Audi S5 Coupe 2012', 'a Audi S4 Sedan 2012', 'a Audi S4 Sedan 2007', 'a Audi TT RS Coupe 2012', 'a BMW ActiveHybrid 5 Sedan 2012', 'a BMW 1 Series Convertible 2012', 'a BMW 1 Series Coupe 2012', 'a BMW 3 Series Sedan 2012', 'a Acura RL Sedan 2012', 'a BMW 3 Series Wagon 2012', 'a BMW 6 Series Convertible 2007', 'a BMW X5 SUV 2007', 'a BMW X6 SUV 2012', 'a BMW M3 Coupe 2012', 'a BMW M5 Sedan 2010', 'a BMW M6 Convertible 2010', 'a BMW X3 SUV 2012', 'a BMW Z4 Convertible 2012', 'a Bentley Continental Supersports Conv. Convertible 2012', 'a Acura TL Sedan 2012', 'a Bentley Arnage Sedan 2009', 'a Bentley Mulsanne Sedan 2011', 'a Bentley Continental GT Coupe 2012', 'a Bentley Continental GT Coupe 2007', 'a Bentley Continental Flying Spur Sedan 2007', 'a Bugatti Veyron 16.4 Convertible 2009', 'a Bugatti Veyron 16.4 Coupe 2009', 'a Buick Regal GS 2012', 'a Buick Rainier SUV 2007', 'a Buick Verano Sedan 2012', 'a Acura TL Type-S 2008', 'a Buick Enclave SUV 2012', 'a Cadillac CTS-V Sedan 2012', 'a Cadillac SRX SUV 2012', 'a Cadillac Escalade EXT Crew Cab 2007', 'a Chevrolet Silverado 1500 Hybrid Crew Cab 2012', 'a Chevrolet Corvette Convertible 2012', 'a Chevrolet Corvette ZR1 2012', 'a Chevrolet Corvette Ron Fellows Edition Z06 2007', 'a Chevrolet Traverse SUV 2012', 'a Chevrolet Camaro Convertible 2012', 'a Acura TSX Sedan 2012', 'a Chevrolet HHR SS 2010', 'a Chevrolet Impala Sedan 2007', 'a Chevrolet Tahoe Hybrid SUV 2012', 'a Chevrolet Sonic Sedan 2012', 'a Chevrolet Express Cargo Van 2007', 'a Chevrolet Avalanche Crew Cab 2012', 'a Chevrolet Cobalt SS 2010', 'a Chevrolet Malibu Hybrid Sedan 2010', 'a Chevrolet TrailBlazer SS 2009', 'a Chevrolet Silverado 2500HD Regular Cab 2012', 'a Acura Integra Type R 2001', 'a Chevrolet Silverado 1500 Classic Extended Cab 2007', 'a Chevrolet Express Van 2007', 'a Chevrolet Monte Carlo Coupe 2007', 'a Chevrolet Malibu Sedan 2007', 'a Chevrolet Silverado 1500 Extended Cab 2012', 'a Chevrolet Silverado 1500 Regular Cab 2012', 'a Chrysler Aspen SUV 2009', 'a Chrysler Sebring Convertible 2010', 'a Chrysler Town and Country Minivan 2012', 'a Chrysler 300 SRT-8 2010', 'a Acura ZDX Hatchback 2012', 'a Chrysler Crossfire Convertible 2008', 'a Chrysler PT Cruiser Convertible 2008', 'a Daewoo Nubira Wagon 2002', 'a Dodge Caliber Wagon 2012', 'a Dodge Caliber Wagon 2007', 'a Dodge Caravan Minivan 1997', 'a Dodge Ram Pickup 3500 Crew Cab 2010', 'a Dodge Ram Pickup 3500 Quad Cab 2009', 'a Dodge Sprinter Cargo Van 2009', 'a Dodge Journey SUV 2012', 'a Aston Martin V8 Vantage Convertible 2012', 'a Dodge Dakota Crew Cab 2010', 'a Dodge Dakota Club Cab 2007', 'a Dodge Magnum Wagon 2008', 'a Dodge Challenger SRT8 2011', 'a Dodge Durango SUV 2012', 'a Dodge Durango SUV 2007', 'a Dodge Charger Sedan 2012', 'a Dodge Charger SRT-8 2009', 'a Eagle Talon Hatchback 1998', 'a FIAT 500 Abarth 2012', 'a Aston Martin V8 Vantage Coupe 2012']

class Cars:
    '''
    For datasets organized in train/val folders
    '''
    def __init__(self, preprocess,
                 location=os.path.expanduser('~/data'),
                 batch_size=128,
                 num_workers=16,
                 classnames=None):
        train_path = os.path.join(location, 'train')
        test_path = '/opt/tiger/filter_transfer/data/cars_train_val_test/test'
        if not os.path.exists(test_path):
            test_path = os.path.join(location, 'test')
        if not os.path.exists(test_path):
            raise ValueError("Test data must be stored in dataset/test or {0}".format(test_path))

        self.train_dataset = folder.ImageFolder(root=train_path, transform=preprocess)
        self.test_dataset = folder.ImageFolder(root=test_path, transform=preprocess)

        self.train_loader = DataLoader(self.train_dataset, batch_size=batch_size,
            shuffle=True, num_workers=num_workers, pin_memory=True)

        self.test_loader = DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        # self.classnames = [v[2:] for v in cars_names]
        self.classnames = [v for v in cars_names_coop]


    num_classes: int = len(class_names)

    def __init__(self, *args, split: str = "train", seed: int = 0, 
                 image_dir: str = DEFAULT_IMAGE_DIR, 
                 examples_per_class: int = None, 
                 generative_aug: GenerativeAugmentation = None, 
                 synthetic_probability: float = 0.5,
                 use_randaugment: bool = False,
                 image_size: Tuple[int] = (256, 256), **kwargs):

        super(Flowers102Dataset, self).__init__(
            *args, examples_per_class=examples_per_class,
            synthetic_probability=synthetic_probability, 
            generative_aug=generative_aug, **kwargs)

        imagelabels = loadmat(os.path.join(image_dir, "imagelabels.mat"))["labels"][0]
        image_files = sorted(list(glob.glob(os.path.join(image_dir, "jpg/*.jpg"))))

        class_to_images = defaultdict(list)

        for image_idx, image_path in enumerate(image_files):
            class_name = self.class_names[imagelabels[image_idx] - 1]
            class_to_images[class_name].append(image_path)

        rng = np.random.default_rng(seed)
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

    def __len__(self):
        
        return len(self.all_images)

    def get_image_by_idx(self, idx: int) -> Image.Image:

        return Image.open(self.all_images[idx]).convert('RGB')

    def get_label_by_idx(self, idx: int) -> int:

        return self.all_labels[idx]
    
    def get_metadata_by_idx(self, idx: int) -> dict:

        return dict(name=self.class_names[self.all_labels[idx]])