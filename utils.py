import numpy as np
import torch
import random
import argparse
from CLIP import clip
import torch.nn.functional as F
from tqdm import tqdm


def update_cfg(args, cfg):
    if args.config is not None:
        cfg['config'] = args.config
    if args.shots is not None:
        cfg['shots'] = args.shots
    if args.T is not None:
        cfg['T'] = args.T
    if args.backbone is not None:
        cfg['backbone'] = args.backbone
        
    return cfg

def set_seed(seed):
    torch.manual_seed(seed)            
    torch.cuda.manual_seed(seed)       
    torch.cuda.manual_seed_all(seed)  

    np.random.seed(seed)               
    random.seed(seed)                  

    torch.backends.cudnn.deterministic = True  
    torch.backends.cudnn.benchmark = False 

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', dest='config', help='settings of Tip-Adapter in yaml format')
    parser.add_argument('--gpu', type= str ,default=None)
    parser.add_argument('--beta', type= float ,default=None)
    parser.add_argument('--alpha', type= float ,default=None)
    parser.add_argument('--shots', type= int,default=None)
    parser.add_argument('--T', type=int,default=1)
    parser.add_argument('--backbone', type=str,default='RN50')

    args = parser.parse_args()

    return args

def cls_acc(output, target, topk=1):
    pred = output.topk(topk, 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    acc = float(correct[: topk].reshape(-1).float().sum(0, keepdim=True).cpu().numpy())
    acc = 100 * acc / target.shape[0]

    return acc

def minmax_normalization(metric):
    row_max = torch.max(metric, dim=1).values
    row_min = torch.min(metric, dim=1).values
    range_values = row_max - row_min
    normalized_tensor = (metric - row_min.view(-1, 1)) / range_values.view(-1, 1)

    return normalized_tensor

def test_model(cfg, alpha, beta, test_features, test_labels, cache_keys, cache_values, clip_logits):
    total = 0
    correct = 0

    if cfg['dataset'] == 'ImageNet':
        test_features = test_features.detach().cpu().float()
        cache_keys = cache_keys.detach().cpu().float()
        
    with torch.no_grad():
        affinity = test_features.float() @ cache_keys.float() 
        affinity = minmax_normalization(affinity)
        
        affinity = affinity / cfg['T']
        metric = ((-1) * (beta - beta * affinity)).exp()

        cache_logits = metric.float().cuda() @ cache_values.float().cuda()
        tip_logits =  clip_logits + cache_logits * alpha
        pred = tip_logits.topk(1, 1, True, True)[1].t()
        correct_ = pred.eq(test_labels.view(1, -1).expand_as(pred))
        correct += float(correct_[: 1].reshape(-1).float().sum(0, keepdim=True).cpu().numpy())
        total += len(test_labels)
        acc = 100 * correct / total

    return acc

def clip_classifier(classnames, template, clip_model):
    with torch.no_grad():
        clip_weights = []

        for classname in classnames:
            # Tokenize the prompts
            classname = classname.replace('_', ' ')
            texts = [t.format(classname) for t in template]
            texts = clip.tokenize(texts, truncate=True).cuda()
            class_embeddings = clip_model.encode_text(texts)
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            clip_weights.append(class_embedding)

        clip_weights = torch.stack(clip_weights, dim=1).cuda()

    return clip_weights

def get_thres(dataset):
    thres_dict = {
        'ImageNet': 0.1,
        'ImageNet-V2': 0.05,
        'places-365': 0.1,
        'caltech-101': 0.3,
        'DTD': 0.6,
        'CUB200': 0.2,
        'eurosat': 0.3, 
        'food-101': 0.1,
        'oxford-pet': 0.1
    }

    return thres_dict[dataset]

def build_cache_model(cfg, clip_model, train_loader_cache):
    if cfg['load_cache'] == False:    
        cache_keys = []
        cache_values = []

        with torch.no_grad():
            # Data augmentation for the cache model
            for augment_idx in range(cfg['augment_epoch']):
                train_features = []

                print('Augment Epoch: {:} / {:}'.format(augment_idx, cfg['augment_epoch']))
                for i, (images, target) in enumerate(tqdm(train_loader_cache)):
                    images = images.cuda()
                    image_features = clip_model.encode_image(images)
                    train_features.append(image_features.cpu())
                    if augment_idx == 0:
                        target = target.cuda()
                        cache_values.append(target)
                cache_keys.append(torch.cat(train_features, dim=0).unsqueeze(0))
            
        cache_keys = torch.cat(cache_keys, dim=0).mean(dim=0).cuda()
        cache_keys /= cache_keys.norm(dim=-1, keepdim=True)

        cache_keys = cache_keys.permute(1, 0)
        cache_values = F.one_hot(torch.cat(cache_values, dim=0)).half()

        torch.save(cache_keys, cfg['cache_dir'] + '/keys_' + str(cfg['shots']) + "shots.pt")
        torch.save(cache_values, cfg['cache_dir'] + '/values_' + str(cfg['shots']) + "shots.pt")

    else:
        cache_keys = torch.load(cfg['cache_dir'] + '/keys_' + str(cfg['shots']) + "shots.pt")
        cache_values = torch.load(cfg['cache_dir'] + '/values_' + str(cfg['shots']) + "shots.pt")

    return cache_keys, cache_values

def build_cache(cfg, clip_model, train_loader, coder):
    if cfg['load_cache'] == False:    
        cache_keys = []
        cache_values = []

        with torch.no_grad():
            # Data augmentation for the cache model
            for augment_idx in range(cfg['augment_epoch']):
                train_features = []
                print('Augment Epoch: {:} / {:}'.format(augment_idx, cfg['augment_epoch']))
                for i, (images, target) in enumerate(tqdm(train_loader)):
                    images = images.cuda()
                    image_features = clip_model.encode_image(images)
                    train_features.append(image_features.cpu())
                    if augment_idx == 0:
                        target = target.cuda()
                        cache_values.append(target)
                cache_keys.append(torch.cat(train_features, dim=0).unsqueeze(0))

        cache_keys = torch.cat(cache_keys, dim=0).mean(dim=0)
        cache_keys /= cache_keys.norm(dim=-1, keepdim=True)

        cache_keys, _ = coder.get_general_CODER(cache_keys)

        cache_keys /= cache_keys.norm(dim=-1, keepdim=True)
        cache_keys = cache_keys.permute(1, 0).cuda()
        cache_values = F.one_hot(torch.cat(cache_values, dim=0)).half()

        torch.save(cache_keys, cfg['cache_dir'] + '/keys_' + str(cfg['shots']) + "shots.pt")
        torch.save(cache_values, cfg['cache_dir'] + '/values_' + str(cfg['shots']) + "shots.pt")  
    else:
        cache_keys = torch.load(cfg['cache_dir'] + '/keys_' + str(cfg['shots']) + "shots.pt")
        cache_values = torch.load(cfg['cache_dir'] + '/values_' + str(cfg['shots']) + "shots.pt")

    return cache_keys, cache_values

def get_clip_logits(clip_tc, feat):
    clip_tc = F.normalize(clip_tc, p=2, dim=1)
    feat = F.normalize(feat, p=2, dim=1)
    clip_logits =  (feat.float() @ clip_tc.T.float())

    return clip_logits

def load_test_features(cfg, split, clip_model, loader, coder):
    if cfg['load_pre_feat'] == False:
        ori_features, features, labels = [], [], []

        with torch.no_grad():
            for i, (images, target) in enumerate(tqdm(loader)):
                images, target = images.cuda(), target.cuda()
                image_features = clip_model.encode_image(images)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                ori_features.append(image_features.cpu())
                image_features = get_clip_logits(coder.total_general_texts, image_features)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                features.append(image_features.cpu())
                labels.append(target)

        ori_features, features, labels = torch.cat(ori_features).cuda(), torch.cat(features).cuda(), torch.cat(labels)

        torch.save(ori_features, cfg['cache_dir'] + "/" + split + "_f_ori.pt")
        torch.save(features, cfg['cache_dir'] + "/" + split + "_f.pt")
        torch.save(labels, cfg['cache_dir'] + "/" + split + "_l.pt")
   
    else:
        ori_features = torch.load(cfg['cache_dir'] + "/" + split + "_f_ori.pt")
        features = torch.load(cfg['cache_dir'] + "/" + split + "_f.pt")
        labels = torch.load(cfg['cache_dir'] + "/" + split + "_l.pt")
    
    return ori_features, features, labels