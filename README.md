# Leveraging Cross-Modal Neighbor Representation for Improved CLIP Classification

🎉The code repository for "Leveraging Cross-Modal Neighbor Representation for Improved CLIP Classification"  in PyTorch. If you use any content of this repo for your work, please cite the following bib entry:

```
  @inproceedings{yic2024coder,
    title={Leveraging Cross-Modal Neighbor Representation for Improved CLIP Classification},
    author={Yi, Chao and Ren, Lu and Zhan, De-Chuan and Ye, Han-Jia},
    booktitle={CVPR},
    year={2024}
  }
```

## Requirements
### 🗂️ Environment
1. [torch 1.13.0](https://github.com/pytorch/pytorch)
2. [torchvision 0.14.0](https://github.com/pytorch/vision)
3. [openai 0.28.0](https://github.com/openai/openai-cookbook?tab=readme-ov-file)

### 🔎 Model
We use the five CLIP models provided officially by OpenAI, namely CLIP RN50, CLIP ViT-B/32, CLIP ViT-B/16, CLIP ViT-L/14 and CLIP ViT-L/14@336px. The download links for these models are [CLIP RN50](https://openaipublic.azureedge.net/clip/models/afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762/RN50.pt), [CLIP ViT-B/32](https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt), [CLIP ViT-B/16](https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt), [CLIP ViT-L/14](https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt), [ViT-L/14@336px](https://openaipublic.azureedge.net/clip/models/3035c92b350959924f9f00213499208652fc7ea050643e8b385c2dac08641f02/ViT-L-14-336px.pt). 
The downloaded pre-trained models should be placed in the `./CLIP/models/ckp/clip`.

### 🔎 Datasets

Please refer to the **DATASET.md** file for specific details on downloading and processing the relevant datasets.

### 👨‍🔬👩‍🔬 Expert Knowledge

We provide external expert knowledge related to the object classes used in our experiments, including object attributes, analogous classes, synonyms, and one-to-one attributes. This knowledge is generated through calls to the [ChatGPT API](https://github.com/openai/openai-cookbook?tab=readme-ov-file) or [WordNet](https://wordnet.princeton.edu/) and is stored in text format. Download link: [link](https://drive.google.com/file/d/1qOjqfWgJyUWgxhVhafW7G6qDCmahamVs/view?usp=drive_link)

After download the expert_knowledge.zip file, you need to unzip the zip file to get the folder `./expert_knowledge`.


## 🔑 Running scripts

To run the zero-shot image classification experiments:

```
bash exp_zero_shot.sh
```
To run the few-shot image classification experiments:

```
bash exp_few_shot.sh
```
You need to first modify the `root_path` attribute in each YAML file (e.g. caltech101.yaml) under the `./configs` folder, setting it to the absolute path of the root folder where your data is located.


## 👨‍🏫 Acknowledgment

We would like to express our gratitude to the following repositories for offering valuable components and functions that contributed to our code.

- [Tip-Adapter](https://github.com/gaopengcuhk/Tip-Adapter)
- [Visual Classfication via Description from Large Language Models](https://github.com/sachit-menon/classify_by_description_release)