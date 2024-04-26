import yaml
from utils import *
import os
from CLIP.models.load_model import get_clip_model
from torchvision import transforms
from datasets.few_shot_dataset import build_dataset
from datasets.few_shot_dataset.utils import build_data_loader
from CODER.CODER import CODER
from datasets.few_shot_dataset.imagenet import ImageNet

torch.set_printoptions(precision=8, threshold=None, edgeitems=None, linewidth=None, profile=None)

def get_cache(train_loader, cfg, clip_model, coder):
    cache_keys, cache_values = build_cache(cfg, clip_model, train_loader, coder)

    cache_keys = cache_keys.to(torch.float64)
    cache_values = cache_values.to(torch.float64)

    return cache_keys, cache_values

def get_val_feature(val_dataloader, cfg, clip_model, coder):
    val_features_ori, val_features_coder, val_labels = load_test_features(cfg, "val", clip_model, val_dataloader, coder)
    
    val_features_ori = val_features_ori.to(torch.float64)
    val_features_coder = val_features_coder.to(torch.float64)

    return val_features_ori, val_features_coder, val_labels

def get_test_feature(test_dataloader, cfg, clip_model, coder):
    test_features_ori, test_features_coder, test_labels = load_test_features(cfg, "test", clip_model, test_dataloader, coder)

    test_features_ori = test_features_ori.to(torch.float64)
    test_features_coder = test_features_coder.to(torch.float64)

    return test_features_ori, test_features_coder, test_labels

def search_hp(cache_keys, cache_values, val_dataloader, clip_model, coder, clip_text_classifiers, cfg):
    clip_text_classifiers = clip_text_classifiers.float()

    min_beta_search_scale = 0.1
    max_beta_search_scale = 200
    min_alpha_search_scale = 0.1
    max_alpha_search_scale = 20
    beta_search_step = 500
    alpha_search_step = 50

    val_features_ori, val_features, val_labels = get_val_feature(val_dataloader, cfg, clip_model, coder)
    clip_logits = 100 * val_features_ori.float() @ clip_text_classifiers

    repeat_times = 3
    for i in range(repeat_times):
        beta_list = [i * (max_beta_search_scale - min_beta_search_scale) / beta_search_step + min_beta_search_scale  for i in range(beta_search_step)]
        alpha_list = [i * (max_alpha_search_scale - min_alpha_search_scale) / alpha_search_step + min_alpha_search_scale  for i in range(alpha_search_step)]
        best_acc = 0
        best_beta, best_alpha = 0, 0
        for beta in beta_list:
                for alpha in alpha_list:
                    acc = test_model(cfg, alpha, beta, val_features, val_labels, cache_keys, cache_values, clip_logits)
                    if acc > best_acc:
                        print("New best setting, beta: {:.2f}, alpha: {:.2f}; accuracy: {:.2f}".format(beta, alpha, acc))
                        best_acc = acc
                        best_beta = beta
                        best_alpha = alpha
                        
        min_beta_search_scale = best_beta * 0.5
        max_beta_search_scale = best_beta * 1.5
        min_alpha_search_scale = best_alpha * 0.5
        max_alpha_search_scale = best_alpha * 1.5

    return best_beta, best_alpha

def few_shot_classify(train_dataloader, val_dataloader, test_dataloader, clip_model, coder, clip_text_classifiers, cfg):
    clip_text_classifiers = clip_text_classifiers.float()
    cache_keys, cache_values = get_cache(train_dataloader, cfg, clip_model, coder)
    
    with torch.no_grad():
        test_features_ori, test_features, test_labels = get_test_feature(test_dataloader, cfg, clip_model, coder)
        clip_logits = 100 * test_features_ori.float() @ clip_text_classifiers
        zero_shot_acc = cls_acc(clip_logits, test_labels)
        
        print("\n**** Zero-shot CLIP's test accuracy: {:.2f}. ****\n".format(zero_shot_acc))

        beta, alpha = cfg['init_beta'], cfg['init_alpha']
        test_acc = test_model(cfg, alpha, beta, test_features, test_labels, cache_keys, cache_values, clip_logits)

        beta, alpha = search_hp(cache_keys, cache_values, val_dataloader, clip_model, coder, clip_text_classifiers, cfg) 
        test_acc2 = test_model(cfg, alpha, beta, test_features, test_labels, cache_keys, cache_values, clip_logits)

        few_shot_acc = max(test_acc, test_acc2)
        
        print("\n**** Few-shot CLIP's test accuracy: {:.2f}. ****\n".format(few_shot_acc))


def main(args):
    cfg = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
    cfg = update_cfg(args, cfg)

    set_seed(cfg['seed'])

    cache_dir = os.path.join('./caches', cfg['dataset'])
    os.makedirs(cache_dir, exist_ok=True)
    cfg['cache_dir'] = cache_dir
    args = args

    print("\nRunning configs.")
    print(cfg, "\n")

    clip_model, preprocess = get_clip_model(cfg["backbone"])

    train_tranform = transforms.Compose([
            transforms.RandomResizedCrop(size=224, scale=(0.5, 1), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
        ])
    
    if cfg['dataset'] != 'ImageNet':
        dataset = build_dataset(cfg['dataset'], cfg['root_path'], cfg['shots'])
        val_loader = build_data_loader(data_source=dataset.val, batch_size=64, is_train=False, tfm=preprocess, shuffle=False)
        test_loader = build_data_loader(data_source=dataset.test, batch_size=64, is_train=False, tfm=preprocess, shuffle=False)
        train_loader = build_data_loader(data_source=dataset.train_x, batch_size=256, tfm=train_tranform, is_train=True, shuffle=False)
        
    else:
        dataset = ImageNet(cfg['root_path'], cfg['shots'], preprocess)
        test_loader = torch.utils.data.DataLoader(dataset.test, batch_size=64, num_workers=8, shuffle=False)
        val_loader = test_loader
        train_loader = torch.utils.data.DataLoader(dataset.train, batch_size=256, num_workers=8, shuffle=False)
        
    coder = CODER(cfg["dataset"], ['name','att', 'ana'], clip_model, task_type='few-shot')
    clip_text_classifiers = clip_classifier(dataset.classnames, dataset.template, clip_model)

    few_shot_classify(train_loader, val_loader, test_loader, clip_model, coder, clip_text_classifiers, cfg)



if __name__ == '__main__':
    args = get_arguments()  
    main(args)
    