import os
import pickle
from scipy.io import loadmat
import scipy.io as sio
import torch
import random
from .oxford_pets import OxfordPets
from torchvision.datasets.utils import verify_str_arg
from torchvision.datasets import VisionDataset
from torchvision.datasets.folder import default_loader
from torchvision.datasets.utils import download_url
from torchvision.datasets.utils import extract_archive
template = ['a photo of a {}.']
class StanfordCars(VisionDataset):
    """`Stanford Cars <https://ai.stanford.edu/~jkrause/cars/car_dataset.html>`_ Dataset.
    """
    file_list = {
        'imgs': ('http://imagenet.stanford.edu/internal/car196/car_ims.tgz', 'car_ims.tgz'),
        'annos': ('http://imagenet.stanford.edu/internal/car196/cars_annos.mat', 'cars_annos.mat')
    }

    def __init__(self, root, train, transform=None, target_transform=None, download=False, get_fs=False, k=5, seed=10):
        super(StanfordCars, self).__init__(root, transform=transform, target_transform=target_transform)

        self.loader = default_loader
        self.train = train
        self.template = template
        if self._check_exists():
            print('Files already downloaded and verified.')
        elif download:
            self._download()
        else:
            raise RuntimeError(
                'Dataset not found. You can use download=True to download it.')

        loaded_mat = sio.loadmat(os.path.join(self.root, self.file_list['annos'][1]))
        # self.classnames = loaded_mat['class_names'][0]
        loaded_mat = loaded_mat['annotations'][0]
        
        with open('data/class_name/Cars.txt', 'r') as f:
            lines = f.readlines()
        self.classnames = [line.strip() for line in lines]
        self.samples = []
        for item in loaded_mat:
            if self.train != bool(item[-1][0]):
                path = str(item[0][0])
                path = os.path.join(root, path)
                label = int(item[-2][0]) - 1
                self.samples.append((path, label))

        class_ins_idx = [[] for _ in range(196)]
        train_class_ins_idx = []
        val_class_ins_idx = []
        for idx, (img_path, label) in enumerate(self.samples):
            class_ins_idx[label].append(idx)

        self.get_fs = get_fs
        # 固定随机种子为10
        random.seed(seed)

        if self.get_fs and train:
            for i in range(len(class_ins_idx)):
                part1 = random.sample(class_ins_idx[i], k)
                part2 = [x for x in class_ins_idx[i] if x not in part1]
                train_class_ins_idx.extend(part1)
                val_class_ins_idx.extend(part2)
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