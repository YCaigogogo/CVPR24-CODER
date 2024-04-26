Here's how you should organize the dataset. The processing methods for some of these datasets are sourced from Tip-Adapter.

### FGVCAircraft
- Create a folder named `aircraft/` under `datasets/data`.
- Download the data from https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/archives/fgvc-aircraft-2013b.tar.gz.
- Extract `fgvc-aircraft-2013b.tar.gz` and keep only `data/`.
- Rename the folder to `fgvc_aircraft/`.

The directory structure should look like
```
datasets/data/aircraft/fgvc-aircraft-2013b/fgvc_aircraft/
|–– images/
|–– ... # a bunch of .txt files
```

### Caltech101
- Create a folder named `caltech-101/` under `datasets/data`.
- Download `101_ObjectCategories.tar.gz` from http://www.vision.caltech.edu/Image_Datasets/Caltech101/101_ObjectCategories.tar.gz and extract the file under `datasets/data/caltech-101`.
- Download `split_zhou_Caltech101.json` from this [link](https://drive.google.com/file/d/1hyarUivQE36mY6jSomru6Fjd-JzwcCzN/view?usp=sharing) and put it under `datasets/data/caltech-101`. 

The directory structure should look like
```
datasets/data/caltech-101/
|–– 101_ObjectCategories/
|–– split_zhou_Caltech101.json
```

### StanfordCars
- Create a folder named `stanford_cars/` under `datasets/data`.
- Download the train images http://ai.stanford.edu/~jkrause/car196/cars_train.tgz.
- Download the test images http://ai.stanford.edu/~jkrause/car196/cars_test.tgz.
- Download the train labels https://ai.stanford.edu/~jkrause/cars/car_devkit.tgz.
- Download the test labels http://ai.stanford.edu/~jkrause/car196/cars_test_annos_withlabels.mat.
- Download `split_zhou_StanfordCars.json` from this [link](https://drive.google.com/file/d/1ObCFbaAgVu0I-k_Au-gIUcefirdAuizT/view?usp=sharing).

The directory structure should look like
```
datasets/data/stanford_cars/
|–– cars_test\
|–– cars_test_annos_withlabels.mat
|–– cars_train\
|–– devkit\
|–– split_zhou_StanfordCars.json
```

### CUB200
- Create a folder named `CUB200/` under `datasets/data`.
- Download `CUB200.zip` from https://drive.google.com/file/d/1Z05uNNbe9y4l3a9mYrpv_KRN6tO1Y1cy/view?usp=drive_link and extract the file under `datasets/data/CUB200`.

The directory structure should look like
```
datasets/data/CUB200/
|–– train\
|–– test\
```

### DTD
- Create a folder named `dtd/` under `datasets/data`.
- Download the dataset from https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz and extract it to `datasets/data/dtd`. This should lead to `datasets/data/dtd/dtd`.
- Download `split_zhou_DescribableTextures.json` from this [link](https://drive.google.com/file/d/1u3_QfB467jqHgNXC00UIzbLZRQCg2S7x/view?usp=sharing).

The directory structure should look like
```
datasets/data/dtd/dtd/
|–– images/
|–– imdb/
|–– labels/
|–– split_zhou_DescribableTextures.json
```

### EuroSAT
- Create a folder named `eurosat/` under `datasets/data`.
- Download the dataset from http://madm.dfki.de/files/sentinel/EuroSAT.zip and extract it to `datasets/data/eurosat/`.
- Download `split_zhou_EuroSAT.json` from [here](https://drive.google.com/file/d/1Ip7yaCWFi0eaOFUGga0lUdVi_DDQth1o/view?usp=sharing).

The directory structure should look like
```
datasets/data/eurosat/
|–– 2750/
|–– split_zhou_EuroSAT.json
```

### Food101
- Create a folder named `food-101/` under `datasets/data`.
- Download the dataset from https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/ and extract the file `food-101.tar.gz` under `datasets/data/food-101`, resulting in a folder named `datasets/data/food-101/`.
- Download the dataset from https://drive.google.com/file/d/1qbqmbEqUMyRrh_eRHmbHJp98J6AR2LCG/view?usp=drive_link and extract the file under `datasets/data/food-101`.
- Download `split_zhou_Food101.json` from [here](https://drive.google.com/file/d/1QK0tGi096I0Ba6kggatX1ee6dJFIcEJl/view?usp=sharing).

The directory structure should look like
```
datasets/data/food-101/
|–– images/
|–– license_agreement.txt
|–– meta/
|–– train/
|–– test/
|–– README.txt
|–– split_zhou_Food101.json
```

### ImageNet
- Create a folder named `imagenet/` under `datasets/data`.
- Download the dataset from the [official website](https://image-net.org/index.php) and extract the training and validation sets to `datasets/data/imagenet/images`. The directory structure should look like
```
datasets/data/imagenet/
|–– train/ # contains 1,000 folders like n01440764, n01443537, etc.
|–– val/
```

### ImageNet-V2
- Create a folder named `imagenet_v2/` under `datasets/data`.
- Download the dataset from https://huggingface.co/datasets/vaishaal/ImageNetV2/blob/main/imagenetv2-matched-frequency.tar.gz and extract the folder `imagenetv2-matched-frequency` under `datasets/data/imagenet_v2`, resulting in a folder named `datasets/data/imagenet_v2/imagenetv2-matched-frequency`.
```
datasets/data/imagenet_v2/
|–– imagenetv2-matched-frequency/
```

### Flowers102
- Create a folder named `oxford_flowers/` under `datasets/data`.
- Download the images and labels from https://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz and https://www.robots.ox.ac.uk/~vgg/data/flowers/102/imagelabels.mat respectively.
- Download `cat_to_name.json` from [here](https://drive.google.com/file/d/1AkcxCXeK_RCGCEC_GvmWxjcjaNhu-at0/view?usp=sharing). 
- Download `split_zhou_OxfordFlowers.json` from [here](https://drive.google.com/file/d/1Pp0sRXzZFZq15zVOzKjKBu4A9i01nozT/view?usp=sharing).

The directory structure should look like
```
datasets/data/oxford_flowers/
|–– cat_to_name.json
|–– imagelabels.mat
|–– jpg/
|–– split_zhou_OxfordFlowers.json
```

### OxfordPets
- Create a folder named `oxford-pet/` under `datasets/data`.
- Download the images from https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz.
- Download the annotations from https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz.
- Download `split_zhou_OxfordPets.json` from this [link](https://drive.google.com/file/d/1501r8Ber4nNKvmlFVQZ8SeUHTcdTTEqs/view?usp=sharing). 

The directory structure should look like
```
datasets/data/oxford-pet/
|–– images/
|–– annotations/
|–– split_zhou_OxfordPets.json
```

### Places365
- Create a folder named `places-365/` under `datasets/data`.
- Download the dataset from https://drive.google.com/file/d/1NEOgA3axZJ19IjgaQh8NILLrtmTLSfEU/view?usp=drive_link and extract it under `datasets/data/places-365`.
The directory structure should look like
```
datasets/data/oxford-pet/
|–– val256/
...
```

### SUN397
- Create a folder named  `sun397/` under `datasets/data`.
- Download the images http://vision.princeton.edu/projects/2010/SUN/SUN397.tar.gz.
- Download the partitions https://vision.princeton.edu/projects/2010/SUN/download/Partitions.zip.
- Extract these files under `datasets/data/sun397/`.
- Download `split_zhou_SUN397.json` from this [link](https://drive.google.com/file/d/1y2RD81BYuiyvebdN-JymPfyWYcd8_MUq/view?usp=sharing).

The directory structure should look like
```
datasets/data/sun397/
|–– SUN397/
|–– split_zhou_SUN397.json
|–– ... # a bunch of .txt files
```

### UCF101
- Create a folder named `ucf101/` under `datasets/data`.
- Download the zip file `UCF-101-midframes.zip` from [here](https://drive.google.com/file/d/10Jqome3vtUA2keJkNanAiFpgbyC9Hc2O/view?usp=sharing) and extract it to `datasets/data/ucf101/`. This zip file contains the extracted middle video frames.
- Download `split_zhou_UCF101.json` from this [link](https://drive.google.com/file/d/1I0S0q91hJfsV9Gf4xDIjgDq4AqBNJb1y/view?usp=sharing).

The directory structure should look like
```
datasets/data/ucf101/
|–– UCF-101-midframes/
|–– split_zhou_UCF101.json
```