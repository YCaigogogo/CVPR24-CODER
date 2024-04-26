# *_*coding: utf-8 *_*
# author --liming--
 
import torch
import torchvision
import os 
import sys
sys.path.append(os.path.split(os.path.realpath(__file__))[0])
import dataset_config
from torchvision import datasets, transforms
from datasets.dataset_class import *
import torch.utils.data as data
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

def get_transform(img_size=224):
    transform = transforms.Compose([
        transforms.Resize(img_size, interpolation=BICUBIC),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
    ])
    
    return transform


def get_dataset(dataset_name, batch_size=32, img_size=224, train_shuffle=True):
    if dataset_name == 'CUB200':
        dataset = CUB200(batch_size, train_shuffle, img_size=img_size)
        test_dataset, testLoader = dataset.test_data_load()
        num_class = 200  
             
    elif dataset_name == 'places-365':
        test_transform = get_transform(img_size=img_size)
        test_dataset, num_class = make_places_365_dataset("datasets/data/places-365", test_transform)
        testLoader = data.DataLoader(test_dataset, batch_size=batch_size, num_workers=4)

    elif dataset_name == 'ImageNet-V2':
        test_transform = get_transform(img_size=img_size)
        imagenet_dataset = ImageNetV2Dataset("matched-frequency", location="datasets/data/imagenet_v2", transform=test_transform) # supports matched-frequency, threshold-0.7, top-images variants
        testLoader = torch.utils.data.DataLoader(imagenet_dataset, batch_size=32, num_workers=4, pin_memory=True) # use whatever batch size you wish
        num_class = 1000 
    else:
        dataset_path = dataset_config.Dataset_path[dataset_name]
        trainLoader, testLoader, num_class = prepare_dataloader(dataset_name, dataset_path, batch_size, train_shuffle, img_size=img_size)
    
    return testLoader, num_class
 

def prepare_dataloader(dataset_type, dataset_path, batch_size, train_shuffle=True, return_dataset=False, img_size=224):
    train_transform = get_transform(img_size=img_size)
    test_transform = get_transform(img_size=img_size)

    train_dataset, test_dataset, num_classes = make_dataset(dataset_type, dataset_path, train_transform, test_transform)
    
    data_loader_train = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=4,
        pin_memory=False,
        shuffle=train_shuffle
    )

    data_loader_test = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        num_workers=4,
        pin_memory=False
    )
    
    if return_dataset:
        return train_dataset, test_dataset, num_classes
    else:
        return data_loader_train, data_loader_test, num_classes

def make_places_365_dataset(data_path, val_transform):
    # print(data_path)
    # assert 0
    val_dataset = torchvision.datasets.Places365(root=data_path, split='val', download=False, small=True, transform=val_transform)
    num_classes = 365
    print(f'Dataset: places365 - [train 0] [test {len(val_dataset)}] [num_classes 365]')
    return val_dataset, num_classes

def make_dataset(name, data_path, train_transform, test_transform):
    def imagefolder_dataset(train_prefix, test_prefix):
        return torchvision.datasets.ImageFolder(os.path.join(data_path, train_prefix), transform=train_transform), torchvision.datasets.ImageFolder(os.path.join(data_path, test_prefix), transform=test_transform)
    if name == 'CIFAR10':
        train_dataset = torchvision.datasets.CIFAR10(root=data_path, train=True, download=False, transform=train_transform)
        val_dataset = torchvision.datasets.CIFAR10(root=data_path, train=False, download=False, transform=test_transform)
        num_classes = 10
    elif name == 'CIFAR100':
        train_dataset = torchvision.datasets.CIFAR100(root=data_path, train=True, download=False, transform=train_transform)
        val_dataset = torchvision.datasets.CIFAR100(root=data_path, train=False, download=False, transform=test_transform)
        num_classes = 100
    elif name == 'ImageNet':
        train_dataset, val_dataset = imagefolder_dataset('train', 'val')
        num_classes = 1000
    elif name == 'Aircraft':
        train_dataset = Aircraft(data_path, class_type='family', transform=train_transform, train=True, download=False)
        val_dataset = Aircraft(data_path, class_type='family', transform=test_transform, train=False, download=False)
        num_classes = 70
    elif name == 'caltech-101':
        train_dataset = Caltech101(data_path, transform=train_transform, train=True)
        val_dataset = Caltech101(data_path, transform=test_transform, train=False)
        # num_classes = 101
        num_classes = 100
    elif name == 'Cars':
        train_dataset = Cars(data_path, transform=train_transform, train=True, download=False)
        val_dataset = Cars(data_path, transform=test_transform, train=False, download=False)
        num_classes = 196
    elif name == 'DTD':
        train_dataset = DTD(data_path, transform=train_transform, train=True)
        val_dataset = DTD(data_path, transform=test_transform, train=False)
        num_classes = 47
    elif name == 'eurosat':
        train_dataset = EuroSAT(data_path, transform=train_transform, train=True)
        val_dataset = EuroSAT(data_path, transform=test_transform, train=False)
        num_classes = 10
    elif name == 'flowers':
        train_dataset = Flowers(data_path, transform=train_transform, train=True)
        val_dataset = Flowers(data_path, transform=test_transform, train=False)
        num_classes = 102
    elif name == 'food-101':
        train_dataset, val_dataset = imagefolder_dataset('train', 'test')
        num_classes = 101
    elif name == 'oxford-pet':
        train_dataset = OxfordIIITPet(root=data_path, split='trainval', download=False, transform=train_transform)
        val_dataset = OxfordIIITPet(root=data_path, split='test', download=False, transform=test_transform)
        num_classes = 37
    elif name == 'STL10':
        train_dataset = torchvision.datasets.STL10(root=data_path, split='train', download=False, transform=train_transform)
        val_dataset = torchvision.datasets.STL10(root=data_path, split='test', download=False, transform=test_transform)
        num_classes = 10
    elif name == 'SVHN':
        train_dataset = torchvision.datasets.SVHN(root=data_path, split='train', download=False, transform=train_transform)
        val_dataset = torchvision.datasets.SVHN(root=data_path, split='test', download=False, transform=test_transform)
        num_classes = 10
    elif name == 'sun397':
        train_dataset = SUN397(data_path, transform=train_transform, train=True)
        val_dataset = SUN397(data_path, transform=test_transform, train=False)
        num_classes = 397
    else:
        raise NotImplementedError
    
    print(f'Dataset: {name} - [train {len(train_dataset)}] [test {len(val_dataset)}] [num_classes {num_classes}]')
    return train_dataset, val_dataset, num_classes
