# We optimize the prompt, so the experimental results may slightly differ from those in the CVPR paper.
datasets=('caltech-101' 'dtd' 'eurosat' 'fgvc' 'food101' 'imagenet' 'oxford_flowers' 'oxford-pet' 'stanford_cars' 'sun397' 'ucf101')
shots=(1 2 4 8 16)

# 使用嵌套循环对每种参数组合进行实验
for dataset in "${datasets[@]}"; do
    for shot in "${shots[@]}"; do
        echo "Running few-shot image classification with dataset $dataset using $shot-shot image per class."
        python classify_few_shot.py --config configs/$dataset.yaml --shots $shot
    done
done