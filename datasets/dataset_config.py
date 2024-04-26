import os

def get_abs_path(relative_path):
    parent_path = os.path.abspath(os.getcwd())
    abs_path = os.path.join(parent_path, relative_path)

    return abs_path

Dataset_path = {}

Dataset_path['CUB'] = {}
Dataset_path['CUB']['path'] = get_abs_path("datasets/data/CUB200")
 
Dataset_path['CUB']['root_train']= os.path.join(Dataset_path['CUB']['path'], 'train')
Dataset_path['CUB']['root_test'] = os.path.join(Dataset_path['CUB']['path'], 'test')

Dataset_path['caltech-101'] = 'datasets/data/caltech-101'

Dataset_path['eurosat'] = 'datasets/data/eurosat'

Dataset_path['flowers'] = 'datasets/data/oxford_flowers'

Dataset_path['food-101'] = 'datasets/data/food-101'

Dataset_path['oxford-pet'] = 'datasets/data/oxford-pet'

Dataset_path['sun397'] = 'datasets/data/sun397'

Dataset_path['Aircraft'] = 'datasets/data/aircraft'

Dataset_path['Cars'] = 'datasets/data/cars'

Dataset_path['DTD'] = 'datasets/data/dtd'

Dataset_path['ImageNet'] = 'datasets/data/imagenet'