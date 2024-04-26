# We optimize the prompt, so the experimental results may slightly differ from those in the CVPR paper.
datasets=('eurosat' 'CUB200' 'DTD' 'food-101' 'caltech-101' 'oxford-pet' 'places-365' 'ImageNet' 'ImageNet-V2')

for dataset in "${datasets[@]}"; do
    echo "Running zero-shot image classification with dataset $dataset using CODER"
    python classify_zero_shot.py --dataset $dataset --text_types name att ana syno
    echo "Running zero-shot image classification with dataset $dataset using CODER*"
    python classify_zero_shot.py --dataset $dataset --text_types name att ana syno ovo --ovo_modify
done
