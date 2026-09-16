import io
import os
import json
import random
import csv
from PIL import Image, ImageFile

import torch
from torch.utils.data import Dataset
from torchvision import transforms

import numpy as np

import warnings

from tqdm import tqdm

import torchvision.transforms as transforms
import torchvision.transforms.functional as F

ImageFile.LOAD_TRUNCATED_IMAGES = True

class CSVDataset(Dataset):

    def __init__(self, csv_path: str, is_train: bool, resolution: int = 224, override_category_label: int = None):
        self.csv_path = csv_path
        self.is_train = is_train
        self.resolution = resolution
        self.override_category_label = override_category_label
        self.data_list = []
        self._init_transforms()
        self._load_data()
        if not self.data_list:
            raise RuntimeError(f"No data found in CSV: {self.csv_path}")

    def _init_transforms(self):
        self.train_transform = transforms.Compose([
            transforms.Resize([224, 224]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]),
        ])

        self.eval_transform = transforms.Compose([
            transforms.Resize([224, 224]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
        ])

    def _load_data(self):
        print(f"Loading CSV dataset: {self.csv_path}")
        if self.override_category_label is not None:
            print(f"  Using override category_label={self.override_category_label} for all samples")
        with open(self.csv_path, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                raise RuntimeError(f"CSV has no header: {self.csv_path}")
            if "img_path" not in reader.fieldnames or "label" not in reader.fieldnames:
                raise KeyError(f"CSV must contain columns img_path,label. Got: {reader.fieldnames}")

            for row in reader:
                img_path = (row.get("img_path") or "").strip()
                if not img_path:
                    continue
                try:
                    label = int(row.get("label"))
                except Exception:
                    continue


                if self.override_category_label is not None:
                    category_label = self.override_category_label
                else:
                    try:
                        category_label = int(row.get("intersec_label", -1))
                    except Exception:
                        category_label = -1


                ismale = row.get("ismale", "")
                iswhite = row.get("iswhite", "")
                isblack = row.get("isblack", "")

                self.data_list.append({
                    "image_path": img_path,
                    "label": label,
                    "category_label": category_label,
                    "ismale": ismale,
                    "iswhite": iswhite,
                    "isblack": isblack,
                })

    def _load_rgb(self, file_path: str) -> Image.Image:
        try:
            pil_img = Image.open(file_path).convert('RGB')
            return pil_img
        except Exception as e:
            raise IOError(f"Failed to load image '{file_path}' PIL.") from e

    def __len__(self):
        return len(self.data_list)

    def get_sample_info(self, index):

        sample = self.data_list[index]
        return {
            "img_path": sample["image_path"],
            "label": sample["label"],
            "ismale": sample.get("ismale", ""),
            "iswhite": sample.get("iswhite", ""),
            "isblack": sample.get("isblack", ""),
        }

    def __getitem__(self, index):
        sample = self.data_list[index]
        image_path = sample["image_path"]
        target = sample["label"]
        category_label = sample.get("category_label", -1)

        try:
            image = self._load_rgb(image_path)
            if self.is_train:
                image_tensor = self.train_transform(image)
            else:
                image_tensor = self.eval_transform(image)

            return image_tensor, torch.tensor(int(target), dtype=torch.long), image_path, torch.tensor(int(category_label), dtype=torch.long)
        except Exception as e:
            warnings.warn(f"Error loading image '{image_path}': {e}. Skipping and loading a random sample.")
            return self.__getitem__(random.randint(0, len(self.data_list) - 1))


class SimpleImageFolderDataset(Dataset):

    def __init__(self, folder_path: str, is_train: bool = True, resolution: int = 224):
        self.folder_path = folder_path
        self.is_train = is_train
        self.resolution = resolution
        self.data_list = []
        self._init_transforms()
        self._load_data()
        if not self.data_list:
            raise RuntimeError(f"No images found in: {self.folder_path}")

    def _init_transforms(self):
        self.train_transform = transforms.Compose([
            transforms.Resize([224, 224]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]),
        ])
        self.eval_transform = transforms.Compose([
            transforms.Resize([224, 224]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
        ])

    def _load_data(self):
        print(f"Loading images from folder: {self.folder_path}")
        real_count, fake_count = 0, 0
        for root, dirs, files in os.walk(self.folder_path):
            for f in files:
                if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):

                    if '_fake' in f.lower():
                        label = 1
                        fake_count += 1
                    elif '_real' in f.lower():
                        label = 0
                        real_count += 1
                    else:

                        continue
                    self.data_list.append({
                        'path': os.path.join(root, f),
                        'label': label
                    })
        self.data_list.sort(key=lambda x: x['path'])
        print(f"  Found {len(self.data_list)} images ({real_count} real, {fake_count} fake)")

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        sample = self.data_list[index]
        image_path = sample['path']
        label = sample['label']
        try:
            image = Image.open(image_path).convert('RGB')
            if self.is_train:
                image_tensor = self.train_transform(image)
            else:
                image_tensor = self.eval_transform(image)
            return image_tensor, torch.tensor(label, dtype=torch.long), image_path, torch.tensor(-1, dtype=torch.long)
        except Exception as e:
            warnings.warn(f"Error loading image '{image_path}': {e}. Skipping.")
            return self.__getitem__(random.randint(0, len(self.data_list) - 1))


class GenerativeImageDataset(Dataset):

    def __init__(self, root: str, is_train: bool,
                 label: int = None, category2label: int = None,
                 resolution: int = 224,
                 real_folder_name: str = '0_real', fake_folder_name: str = '1_fake'):

        self.root = root
        self.is_train = is_train
        self.resolution = resolution
        self.data_list = []

        self.explicit_label = label
        self.category2label = category2label

        self.real_folder = real_folder_name
        self.fake_folder = fake_folder_name

        self._init_transforms()
        self._load_data()

        if not self.data_list:
            raise RuntimeError(f"No data found for the specified parameters at root: {self.root}")


    def _init_transforms(self):
        self.train_transform = transforms.Compose([
            transforms.Resize([224, 224]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]),
        ])

        self.eval_transform = transforms.Compose([
            transforms.Resize([224, 224]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
        ])


    def _load_data(self):
        print(f"Recursively scanning for data in: {self.root}")
        print(f" - Real folder: '{self.real_folder}', Fake folder: '{self.fake_folder}'")


        for dirpath, dirnames, filenames in os.walk(self.root):
            dirnames.sort()
            filenames.sort()

            relative_path = os.path.relpath(dirpath, self.root)
            path_parts = relative_path.split(os.sep)

            label_for_this_dir = None

            if self.explicit_label is not None:

                label_for_this_dir = self.explicit_label

            elif self.real_folder and self.real_folder in path_parts:
                label_for_this_dir = 0
            elif self.fake_folder and self.fake_folder in path_parts:
                label_for_this_dir = 1


            if label_for_this_dir is not None:

                category = path_parts[0] if path_parts and path_parts[0] != '.' else 'unknown'

                for img_name in filenames:
                    if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                        full_path = os.path.join(dirpath, img_name)
                        self.data_list.append({
                            "image_path": full_path,
                            "label": label_for_this_dir,
                            "category": category
                        })

        if self.data_list:
            real_count = sum(1 for item in self.data_list if item['label'] == 0)
            fake_count = sum(1 for item in self.data_list if item['label'] == 1)
            total_count = len(self.data_list)

            print(f"Found {total_count} images:")
            print(f" - Real images: {real_count}")
            print(f" - Fake images: {fake_count}")
            if total_count > 0:
                print(f" - Real/Fake ratio: {real_count/total_count:.2%}/{fake_count/total_count:.2%}")
        else:
            print("Warning: No images found. Check your root path and folder names.")


    def _load_rgb(self, file_path: str) -> Image.Image:
        try:
            pil_img = Image.open(file_path).convert('RGB')
            return pil_img
        except Exception as e:
            raise IOError(f"Failed to load image '{file_path}' PIL.") from e


    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):

        sample = self.data_list[index]
        image_path = sample['image_path']
        target = sample['label']

        category_label = -1

        if 'category' in sample:
            if self.category2label is not None and sample['category'] in self.category2label:
                category_label = self.category2label[sample['category']]

        try:
            image = self._load_rgb(image_path)
            if self.is_train:
                image_tensor = self.train_transform(image)
            else:
                image_tensor = self.eval_transform(image)

            return image_tensor, torch.tensor(int(target), dtype=torch.long), image_path, torch.tensor(int(category_label), dtype=torch.long)

        except Exception as e:
            warnings.warn(f"Error loading image '{image_path}': {e}. Skipping and loading a random sample.")
            return self.__getitem__(random.randint(0, len(self.data_list) - 1))
