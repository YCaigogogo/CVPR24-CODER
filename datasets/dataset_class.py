# *_*coding: utf-8 *_*
# author --liming--
 
import torch
import torchvision
import pickle
# import dataset_config
import dataset_config
from torchvision import datasets, transforms
import os
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from pathlib import Path
import scipy.io as sio
import torch.utils.data as data
import pandas as pd
from PIL import Image
from collections import OrderedDict
from tqdm import tqdm
import tarfile
import requests
import shutil

import os.path
import pathlib
from typing import Any, Callable, Optional, Union, Tuple
from typing import Sequence

import random

from torchvision.datasets.utils import verify_str_arg
from torchvision.datasets import VisionDataset
from torchvision.datasets.folder import default_loader
from torchvision.datasets.utils import download_url
from torchvision.datasets.utils import extract_archive

# from utils.get_dataset import *

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

URLS = {"matched-frequency" : "https://huggingface.co/datasets/vaishaal/ImageNetV2/resolve/main/imagenetv2-matched-frequency.tar.gz",
        "threshold-0.7" : "https://huggingface.co/datasets/vaishaal/ImageNetV2/resolve/main/imagenetv2-threshold0.7.tar.gz",
        "top-images": "https://huggingface.co/datasets/vaishaal/ImageNetV2/resolve/main/imagenetv2-top-images.tar.gz",
        "val": "https://imagenet2val.s3.amazonaws.com/imagenet_validation.tar.gz"}

FNAMES = {"matched-frequency" : "imagenetv2-matched-frequency-format-val",
        "threshold-0.7" : "imagenetv2-threshold0.7-format-val",
        "top-images": "imagenetv2-top-images-format-val",
        "val": "imagenet_validation"}

V2_DATASET_SIZE = 10000

def load_pickle(path):
    f = open(path, 'rb')
    result = pickle.load(f)
    return result

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')
    

class CUB200:
    def __init__(self, batch_size=16, train_shuffle=True, img_size=224):
        self.train_transform = self.get_transform(img_size=img_size)
        self.test_transform = self.get_transform(img_size=img_size)

        self.root_train = dataset_config.Dataset_path['CUB']['root_train']
        self.root_test = dataset_config.Dataset_path['CUB']['root_test']

        self.batch_size = batch_size
        self.train_shuffle = train_shuffle
        self.class_num = 200


    def get_transform(self, img_size=224):
        transform = transforms.Compose([
            transforms.Resize(img_size, interpolation=BICUBIC),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
            ])
        
        return transform
 
    def train_data_load(self):
        train_dataset = torchvision.datasets.ImageFolder(self.root_train,
                                                        transform=self.train_transform)
        
        class_ins_idx = [[] for _ in range(200)]

        for idx, (img_path, label) in enumerate(train_dataset.samples):
            class_ins_idx[label].append(idx)

        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                batch_size=self.batch_size,
                                                shuffle=self.train_shuffle,
                                                num_workers=4)

        return train_dataset, train_loader, 
    
    def test_data_load(self):
        test_dataset = torchvision.datasets.ImageFolder(self.root_test,
                                                    transform=self.test_transform)
   
            
        test_loader = torch.utils.data.DataLoader(test_dataset,
                                                batch_size=self.batch_size,
                                                shuffle=False,
                                                num_workers=4)
        
        return test_dataset, test_loader
        

class Caltech101(Dataset):
    def __init__(self, root, train, transform=None, target_transform=None, get_fs=False, k=5, seed=10):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.split = 'trainval' if train else 'test'
        data = load_pickle(os.path.join(self.root, 'annotations.pkl'))[self.split]
        self.samples = [(os.path.join(self.root, "101_ObjectCategories", i[0]), i[1]) for i in data]

        self.get_fs = get_fs

        if self.get_fs and train:
            class_ins_idx = [[] for _ in range(100)]
            train_class_ins_idx = []
            val_class_ins_idx = []
            for idx, (img_path, label) in enumerate(self.samples):
                class_ins_idx[label].append(idx)

            random.seed(seed)

            for i in range(len(class_ins_idx)):
                part1 = random.sample(class_ins_idx[i], k)
                part2 = [x for x in class_ins_idx[i] if x not in part1]
                train_class_ins_idx.extend(part2)
                val_class_ins_idx.extend(part1)
            
            self.train_sampler = torch.utils.data.SubsetRandomSampler(train_class_ins_idx)
            self.val_sampler = torch.utils.data.SubsetRandomSampler(val_class_ins_idx)

    def get_sampler(self):
        return self.train_sampler, self.val_sampler

    def __getitem__(self, i):
        image, label = self.samples[i]
        image = pil_loader(image)
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

    def __len__(self):
        return len(self.samples)

class EuroSAT(Dataset):
    def __init__(self, root, train, transform=None, target_transform=None, get_fs=False, k=5, seed=10):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.split = 'trainval' if train else 'test'
        data = load_pickle(os.path.join(self.root, 'annotations.pkl'))[self.split]
        self.samples = [(os.path.join(self.root, '2750', i[0]), i[1]) for i in data]

        class_ins_idx = [[] for _ in range(10)]
        train_class_ins_idx = []
        val_class_ins_idx = []
        for idx, (img_path, label) in enumerate(self.samples):
            class_ins_idx[label].append(idx)

        self.get_fs = get_fs
        random.seed(seed)

        if self.get_fs and train:
            for i in range(len(class_ins_idx)):
                part1 = random.sample(class_ins_idx[i], k)
                part2 = [x for x in class_ins_idx[i] if x not in part1]
                train_class_ins_idx.extend(part2)
                val_class_ins_idx.extend(part1)
            
            self.train_sampler = torch.utils.data.SubsetRandomSampler(train_class_ins_idx)
            self.val_sampler = torch.utils.data.SubsetRandomSampler(val_class_ins_idx)
        
        
    def get_sampler(self):
        return self.train_sampler, self.val_sampler

    def __getitem__(self, i):
        image, label = self.samples[i]
        image = Image.open(image).convert('RGB')
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

    def __len__(self):
        return len(self.samples)
    

class Flowers(Dataset):
    def __init__(self, root, train, mode='all', transform=None, target_transform=None, get_fs=False, k=5, seed=10):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.split = 'trainval' if train else 'test'
        self.load_data(mode)

        class_ins_idx = [[] for _ in range(102)]
        train_class_ins_idx = []
        val_class_ins_idx = []
        for idx, (img_path, label) in enumerate(self.samples):
            class_ins_idx[label].append(idx)

        self.get_fs = get_fs
        random.seed(seed)

        if self.get_fs and train:
            for i in range(len(class_ins_idx)):
                part1 = random.sample(class_ins_idx[i], k)
                part2 = [x for x in class_ins_idx[i] if x not in part1]
                train_class_ins_idx.extend(part2)
                val_class_ins_idx.extend(part1)
            
            self.train_sampler = torch.utils.data.SubsetRandomSampler(train_class_ins_idx)
            self.val_sampler = torch.utils.data.SubsetRandomSampler(val_class_ins_idx)
        
        
    def get_sampler(self):
        return self.train_sampler, self.val_sampler

    def _check_exists(self):
        return os.path.exists(self.root)

    def load_data(self, mode):
        data = load_pickle(os.path.join(self.root, 'annotations.pkl'))[self.split]
        self.samples = [(os.path.join(self.root, 'jpg', i[0]), i[1]) for i in data]

    def __getitem__(self, i):
        image, label = self.samples[i]
        image = Image.open(image).convert('RGB')
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

    def __len__(self):
        return len(self.samples)


class OxfordIIITPet(VisionDataset):
    """`Oxford-IIIT Pet Dataset   <https://www.robots.ox.ac.uk/~vgg/data/pets/>`_.

    Args:
        root (string): Root directory of the dataset.
        split (string, optional): The dataset split, supports ``"trainval"`` (default) or ``"test"``.
        target_types (string, sequence of strings, optional): Types of target to use. Can be ``category`` (default) or
            ``segmentation``. Can also be a list to output a tuple with all specified target types. The types represent:

                - ``category`` (int): Label for one of the 37 pet categories.
                - ``segmentation`` (PIL image): Segmentation trimap of the image.

            If empty, ``None`` will be returned as target.

        transform (callable, optional): A function/transform that  takes in a PIL image and returns a transformed
            version. E.g, ``transforms.RandomCrop``.
        target_transform (callable, optional): A function/transform that takes in the target and transforms it.
        download (bool, optional): If True, downloads the dataset from the internet and puts it into
            ``root/oxford-iiit-pet``. If dataset is already downloaded, it is not downloaded again.
    """

    _RESOURCES = (
        ("https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz", "5c4f3ee8e5d25df40f4fd59a7f44e54c"),
        ("https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz", "95a8c909bbe2e81eed6a22bccdf3f68f"),
    )
    _VALID_TARGET_TYPES = ("category", "segmentation")

    def __init__(
        self,
        root: str,
        split: str = "trainval",
        target_types: Union[Sequence[str], str] = "category",
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
        get_fs = False,
        k = 5,
        seed=10
    ):
        self._split = verify_str_arg(split, "split", ("trainval", "test"))
        if isinstance(target_types, str):
            target_types = [target_types]
        self._target_types = [
            verify_str_arg(target_type, "target_types", self._VALID_TARGET_TYPES) for target_type in target_types
        ]

        super().__init__(root, transforms=transforms, transform=transform, target_transform=target_transform)
        # self._base_folder = pathlib.Path(self.root) / "oxford-iiit-pet"
        self._base_folder = pathlib.Path(self.root) 
        self._images_folder = self._base_folder / "images"
        self._anns_folder = self._base_folder / "annotations"
        self._segs_folder = self._anns_folder / "trimaps"

        if not self._check_exists():
            raise RuntimeError("Dataset not found. You can use download=True to download it")

        image_ids = []
        self._labels = []
        with open(self._anns_folder / f"{self._split}.txt") as file:
            for line in file:
                image_id, label, *_ = line.strip().split()
                image_ids.append(image_id)
                self._labels.append(int(label) - 1)

        self.classes = [
            " ".join(part.title() for part in raw_cls.split("_"))
            for raw_cls, _ in sorted(
                {(image_id.rsplit("_", 1)[0], label) for image_id, label in zip(image_ids, self._labels)},
                key=lambda image_id_and_label: image_id_and_label[1],
            )
        ]
        self.class_to_idx = dict(zip(self.classes, range(len(self.classes))))

        self._images = [self._images_folder / f"{image_id}.jpg" for image_id in image_ids]
        self._segs = [self._segs_folder / f"{image_id}.png" for image_id in image_ids]

        class_ins_idx = [[] for _ in range(37)]
        train_class_ins_idx = []
        val_class_ins_idx = []
        for idx, label in enumerate(self._labels):
            class_ins_idx[label].append(idx)

        self.get_fs = get_fs
        random.seed(seed)

        if self.get_fs and split == 'trainval':
            for i in range(len(class_ins_idx)):
                part1 = random.sample(class_ins_idx[i], k)
                part2 = [x for x in class_ins_idx[i] if x not in part1]
                train_class_ins_idx.extend(part2)
                val_class_ins_idx.extend(part1)
            
            self.train_sampler = torch.utils.data.SubsetRandomSampler(train_class_ins_idx)
            self.val_sampler = torch.utils.data.SubsetRandomSampler(val_class_ins_idx)

        self.samples = []

        for idx in range(len(self._images)):
            self.samples.append((self._images[idx], self._labels[idx]))


    def get_sampler(self):
        return self.train_sampler, self.val_sampler
    
    def get_images(self):
        return self._images

    def __len__(self) -> int:
        return len(self._images)

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        image = Image.open(self._images[idx]).convert("RGB")

        target: Any = []
        for target_type in self._target_types:
            if target_type == "category":
                target.append(self._labels[idx])
            else:  # target_type == "segmentation"
                target.append(Image.open(self._segs[idx]))

        if not target:
            target = None
        elif len(target) == 1:
            target = target[0]
        else:
            target = tuple(target)

        if self.transforms:
            image, target = self.transforms(image, target)

        return image, target

    def _check_exists(self) -> bool:
        for folder in (self._images_folder, self._anns_folder):
            if not (os.path.exists(folder) and os.path.isdir(folder)):
                return False
        else:
            return True


class SUN397(Dataset):
    def __init__(self, root, train, transform=None, target_transform=None):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.split = 'trainval' if train else 'test'
        data = load_pickle(os.path.join(self.root, 'annotations.pkl'))[self.split]
        self.samples = [(os.path.join(self.root, 'SUN397', i[0]), i[1]) for i in data]

    def __getitem__(self, i):
        image, label = self.samples[i]
        image = Image.open(image).convert('RGB')
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

    def __len__(self):
        return len(self.samples)
    
class Aircraft(VisionDataset):
    """`FGVC-Aircraft <http://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/>`_ Dataset.
        Aircraft models are organized in a four-levels hierarchy. The four levels, from finer to coarser, are:

        Model, e.g. Boeing 737-76J. Since certain models are nearly visually indistinguishable, this level is not used in the evaluation.
        Variant, e.g. Boeing 737-700. A variant collapses all the models that are visually indistinguishable into one class. The dataset comprises 102 different variants.
        Family, e.g. Boeing 737. The dataset comprises 70 different families.
        Manufacturer, e.g. Boeing. The dataset comprises 41 different manufacturers.
    """
    url = 'http://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/archives/fgvc-aircraft-2013b.tar.gz'
    class_types = ('variant', 'family', 'manufacturer')
    splits = ('train', 'val', 'trainval', 'test')
    img_folder = os.path.join('fgvc-aircraft-2013b', 'data', 'images')

    def __init__(self, root, train, class_type='variant', transform=None,
                 target_transform=None, download=False, get_fs=False, k=5, seed=10):
        super(Aircraft, self).__init__(root, transform=transform, target_transform=target_transform)
        split = 'trainval' if train else 'test'

        self.class_type = class_type
        self.split = split
        self.classes_file = os.path.join(self.root, 'fgvc-aircraft-2013b', 'data',
                                         'images_%s_%s.txt' % (self.class_type, self.split))

        if download:
            self.download()

        (image_ids, targets, classes, class_to_idx) = self.find_classes()
        samples = self.make_dataset(image_ids, targets)

        self.loader = default_loader

        self.samples = samples
        self.classes = classes
        self.class_to_idx = class_to_idx

        class_ins_idx = [[] for _ in range(70)]
        train_class_ins_idx = []
        val_class_ins_idx = []
        for idx, (img_path, label) in enumerate(self.samples):
            class_ins_idx[label].append(idx)

        self.get_fs = get_fs
        random.seed(seed)

        if self.get_fs and train:
            for i in range(len(class_ins_idx)):
                part1 = random.sample(class_ins_idx[i], k)
                part2 = [x for x in class_ins_idx[i] if x not in part1]
                train_class_ins_idx.extend(part2)
                val_class_ins_idx.extend(part1)
            
            self.train_sampler = torch.utils.data.SubsetRandomSampler(train_class_ins_idx)
            self.val_sampler = torch.utils.data.SubsetRandomSampler(val_class_ins_idx)
        
        
    def get_sampler(self):
        return self.train_sampler, self.val_sampler

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, target

    def __len__(self):
        return len(self.samples)

    def _check_exists(self):
        return os.path.exists(os.path.join(self.root, self.img_folder)) and \
               os.path.exists(self.classes_file)

    def download(self):
        if self._check_exists():
            return

        # prepare to download data to PARENT_DIR/fgvc-aircraft-2013.tar.gz
        print('Downloading %s...' % self.url)
        tar_name = self.url.rpartition('/')[-1]
        download_url(self.url, root=self.root, filename=tar_name)
        tar_path = os.path.join(self.root, tar_name)
        print('Extracting %s...' % tar_path)
        extract_archive(tar_path)
        print('Done!')

    def find_classes(self):
        # read classes file, separating out image IDs and class names
        image_ids = []
        targets = []
        with open(self.classes_file, 'r') as f:
            for line in f:
                split_line = line.split(' ')
                image_ids.append(split_line[0])
                targets.append(' '.join(split_line[1:]))

        # index class names
        classes = np.unique(targets)
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        targets = [class_to_idx[c] for c in targets]

        return image_ids, targets, classes, class_to_idx

    def make_dataset(self, image_ids, targets):
        assert (len(image_ids) == len(targets))
        images = []
        for i in range(len(image_ids)):
            item = (os.path.join(self.root, self.img_folder,
                                 '%s.jpg' % image_ids[i]), targets[i])
            images.append(item)
        return images
    
class ImageNetV2Dataset(Dataset):
    def __init__(self, variant="matched-frequency", transform=None, location="."):
        self.dataset_root = pathlib.Path(f"{location}/imagenetv2-{variant}/")
        self.tar_root = pathlib.Path(f"{location}/imagenetv2-{variant}.tar.gz")
        self.fnames = list(self.dataset_root.glob("**/*.jpeg"))
        self.transform = transform
        assert variant in URLS, f"unknown V2 Variant: {variant}"
        if not self.dataset_root.exists() or len(self.fnames) != V2_DATASET_SIZE:
            if not self.tar_root.exists():
                print(f"Dataset {variant} not found on disk, downloading....")
                response = requests.get(URLS[variant], stream=True)
                total_size_in_bytes= int(response.headers.get('content-length', 0))
                block_size = 1024 #1 Kibibyte
                progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
                with open(self.tar_root, 'wb') as f:
                    for data in response.iter_content(block_size):
                        progress_bar.update(len(data))
                        f.write(data)
                progress_bar.close()
                if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
                    assert False, f"Downloading from {URLS[variant]} failed"
            print("Extracting....")
            tarfile.open(self.tar_root).extractall(f"{location}")
            shutil.move(f"{location}/{FNAMES[variant]}", self.dataset_root)
            self.fnames = list(self.dataset_root.glob("**/*.jpeg"))

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, i):
        img, label = Image.open(self.fnames[i]), int(self.fnames[i].parent.name)
        if self.transform is not None:
            img = self.transform(img)
        return img, label

class Cars(VisionDataset):
    """`Stanford Cars <https://ai.stanford.edu/~jkrause/cars/car_dataset.html>`_ Dataset.
    """
    file_list = {
        'imgs': ('http://imagenet.stanford.edu/internal/car196/car_ims.tgz', 'car_ims.tgz'),
        'annos': ('http://imagenet.stanford.edu/internal/car196/cars_annos.mat', 'cars_annos.mat')
    }

    def __init__(self, root, train, transform=None, target_transform=None, download=False, get_fs=False, k=5, seed=10):
        super(Cars, self).__init__(root, transform=transform, target_transform=target_transform)

        self.loader = default_loader
        self.train = train

        if self._check_exists():
            print('Files already downloaded and verified.')
        elif download:
            self._download()
        else:
            raise RuntimeError(
                'Dataset not found. You can use download=True to download it.')

        loaded_mat = sio.loadmat(os.path.join(self.root, self.file_list['annos'][1]))
        loaded_mat = loaded_mat['annotations'][0]
        self.samples = []
        for item in loaded_mat:
            if self.train != bool(item[-1][0]):
                path = str(item[0][0])
                path = os.path.join("datasets/data/cars", path)
                label = int(item[-2][0]) - 1
                self.samples.append((path, label))
        
        class_ins_idx = [[] for _ in range(196)]
        train_class_ins_idx = []
        val_class_ins_idx = []
        for idx, (img_path, label) in enumerate(self.samples):
            class_ins_idx[label].append(idx)

        self.get_fs = get_fs
        random.seed(seed)

        if self.get_fs and train:
            for i in range(len(class_ins_idx)):
                part1 = random.sample(class_ins_idx[i], k)
                part2 = [x for x in class_ins_idx[i] if x not in part1]
                train_class_ins_idx.extend(part2)
                val_class_ins_idx.extend(part1)
            
            self.train_sampler = torch.utils.data.SubsetRandomSampler(train_class_ins_idx)
            self.val_sampler = torch.utils.data.SubsetRandomSampler(val_class_ins_idx)
        
    def get_sampler(self):
        return self.train_sampler, self.val_sampler

    def __getitem__(self, index):
        path, target = self.samples[index]
        # path = os.path.join(self.root, path)

        image = self.loader(path)
        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return image, target

    def __len__(self):
        return len(self.samples)

    def _check_exists(self):
        return (os.path.exists(os.path.join(self.root, self.file_list['imgs'][1]))
                and os.path.exists(os.path.join(self.root, self.file_list['annos'][1])))

    def _download(self):
        print('Downloading...')
        for url, filename in self.file_list.values():
            download_url(url, root=self.root, filename=filename)
        print('Extracting...')
        archive = os.path.join(self.root, self.file_list['imgs'][1])
        extract_archive(archive)


class DTD(Dataset):
    def __init__(self, root, train, partition=1, transform=None, target_transform=None, get_fs=False, k=5, seed=10):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self._split = ['train', 'val'] if train else ['test']

        self._image_files = []
        classes = []

        self._base_folder = Path(self.root)
        self._data_folder = self._base_folder / "dtd"
        self._meta_folder = self._data_folder / "labels"
        self._images_folder = self._data_folder / "images"
        for split in self._split:
            with open(self._meta_folder / f"{split}{partition}.txt") as file:
                for line in file:
                    cls, name = line.strip().split("/")
                    self._image_files.append(self._images_folder.joinpath(cls, name))
                    classes.append(cls)

        self.classes = sorted(set(classes))
        self.class_to_idx = dict(zip(self.classes, range(len(self.classes))))
        self._labels = [self.class_to_idx[cls] for cls in classes]

        self.samples = [(i, j) for i, j in zip(self._image_files, self._labels)]

        class_ins_idx = [[] for _ in range(47)]
        train_class_ins_idx = []
        val_class_ins_idx = []
        for idx, (img_path, label) in enumerate(self.samples):
            class_ins_idx[label].append(idx)

        self.get_fs = get_fs
        random.seed(seed)

        if self.get_fs and train:
            for i in range(len(class_ins_idx)):
                part1 = random.sample(class_ins_idx[i], k)
                part2 = [x for x in class_ins_idx[i] if x not in part1]
                train_class_ins_idx.extend(part2)
                val_class_ins_idx.extend(part1)
            
            self.train_sampler = torch.utils.data.SubsetRandomSampler(train_class_ins_idx)
            self.val_sampler = torch.utils.data.SubsetRandomSampler(val_class_ins_idx)

    def get_sampler(self):
        return self.train_sampler, self.val_sampler

    def __getitem__(self, idx):
        image_file, label = self.samples[idx]
        image = Image.open(image_file).convert("RGB")

        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            label = self.target_transform(label)

        return image, label

    def __len__(self):
        return len(self.samples)
    

class CUB200_Caps(Dataset):
    def __init__(self, image_caption_list):
        # self.root_train = dataset_config.Dataset_path['CUB_Caps']['root_train']
        self.samples = image_caption_list
 
    def __getitem__(self, i):
        image, caption = self.samples[i]

        return image, caption

    def __len__(self):
        return len(self.samples)
    

class Flicker30k(Dataset):
    def __init__(self, average=False):
        self.image_path = dataset_config.Dataset_path['Flicker30k']['image_path']
        self.caption_path = dataset_config.Dataset_path['Flicker30k']['caption_path']
        info = self.read_image_info(self.caption_path)
        image_path_list = list(info['image_name'])
        comment_list = list(info['comment'])
        self.average = average

        if average:
            total_dict = {}
            self.comment_list = []
            for i in range(len(image_path_list)):
                if image_path_list[i] not in total_dict.keys():
                    total_dict[image_path_list[i]] = []
                total_dict[image_path_list[i]].append(comment_list[i])
            
            unique_ordered_dict = OrderedDict.fromkeys(image_path_list)
            image_path_list = list(unique_ordered_dict.keys())

            for i in range(len(image_path_list)):
                self.comment_list.append('|'.join(total_dict[image_path_list[i]]))
            
            self.image_path_list = image_path_list
        
        else:
            self.image_path_list = image_path_list
            self.comment_list = comment_list

        self.transform = transforms.Compose([
            transforms.Resize(224, interpolation=BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ])


    def read_image_info(self, filepath):
        info=pd.read_csv(filepath,sep='|')
        info=info.rename(columns=lambda x: x.strip())
        info.drop(info[info.comment.isnull()].index,axis=0,inplace=True)
        return info
    

    def __getitem__(self, i):
        image_path = os.path.join(self.image_path, self.image_path_list[i])
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)

        caption = self.comment_list[i]
        
        return image, caption


    def __len__(self):
        return len(self.image_path_list)


if __name__ == '__main__':
    dataset = Flicker30k()