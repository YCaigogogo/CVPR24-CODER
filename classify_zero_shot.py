import argparse
import torch
from utils import *
from tqdm import tqdm
import torch.nn.functional as F
from CLIP.models.load_model import get_clip_model
from datasets.load_dataset import get_dataset
from CODER.CODER import CODER


def get_args_parser():
    parser = argparse.ArgumentParser('coder: encoding image samples', add_help=False)
    parser.add_argument('--dataset', default='caltech-101', type=str)
    parser.add_argument('--ovo_modify', action='store_true')
    parser.add_argument('--model_type', default='ViT-B-32', type=str)
    parser.add_argument('--text_types', nargs='+', type=str)
    parser.add_argument('--result_path', default=None, type=str)

    return parser

def heuristic_classifier(images_coder, text_nums):
    idx = 0
    cls_score_list = []

    for att_ana_num, name_syno_num in text_nums:
        max_name_score = torch.max(images_coder[:, idx+att_ana_num:idx+att_ana_num+name_syno_num], dim=1)[0]
        cls_score = torch.cat((images_coder[:, idx:idx+att_ana_num], max_name_score.unsqueeze(1)), dim=1)
        cls_score_list.append(torch.mean(cls_score, dim=1).unsqueeze(1))
        idx += (att_ana_num+name_syno_num)
        
    total_cls_score = torch.cat(cls_score_list, dim=1)
    total_cls_score = 100 * total_cls_score

    return total_cls_score


def get_class_names(dataset):
    if dataset == 'ImageNet-V2':
        dataset = "ImageNet"
            
    file_path = f'datasets/class_name/{dataset}.txt'  
    category_list = []

    with open(file_path, 'r') as file:
        for line in file:
            cleaned_line = line.strip()
            category_list.append(cleaned_line)

    return category_list
    

def modify_top_gap(img_feats, indices, class_names, coder):
    img_number = img_feats.shape[0]
    top_num = indices.shape[1]

    mod_idx_list = []
    for i in range(img_number):
        img_feat = img_feats[i].cuda()
        top5_cls = [class_names[indices[i,j].item()] for j in range(top_num)]
        top5_cls_idx = [indices[i,j].item() for j in range(top_num)]

        scores = []
        for cls in top5_cls:
            score = []
            other_clses = [class_name for class_name in top5_cls if class_name != cls]

            for other_cls in other_clses:
                if other_cls != cls:  
                    try:
                        cls_ovo_coder = coder.get_ovo_CODER(img_feat, cls, other_cls)
                        other_cls_ovo_coder = coder.get_ovo_CODER(img_feat, other_cls, cls)
                        score.append(torch.mean(cls_ovo_coder, dim=0) - torch.mean(other_cls_ovo_coder, dim=0))
                    except:
                        score.append(0)
            scores.append(sum(score) / len(score))

        zipped_lists = zip(scores, top5_cls_idx)
        sorted_lists = sorted(zipped_lists, key=lambda x: x[0], reverse=True)
        scores, mod_idx = zip(*sorted_lists)
        mod_idx_list.append(mod_idx[0])

    mod_idx = torch.tensor(mod_idx_list)

    return mod_idx


def find_indices(lst, threshold=0.01):
    return [i for i, x in enumerate(lst) if x > threshold]


def classify(dataset, dataloader, clip_model, coder, ovo_modify):
    if ovo_modify:
        class_names = get_class_names(dataset)

    with torch.no_grad():
        total = 0
        correct = 0
        for (imgs, labels) in tqdm(dataloader):
            imgs = imgs.cuda()
            labels = labels.cuda()
            image_features = clip_model.encode_image(imgs)
            image_features = F.normalize(image_features)

            images_coder, text_nums = coder.get_general_CODER(image_features)
            pred = heuristic_classifier(images_coder, text_nums)
            _, indices = pred.topk(1)
            indices = indices.squeeze()

            # ======================== Modify Stage ========================
            if ovo_modify:
                _, top5_indices = pred.topk(5)
                softmax_pred = F.softmax(pred, dim=1)
                top_k_value, _ = softmax_pred.topk(2)
                gap_value = top_k_value[:,0] - top_k_value[:,1]
                thres = get_thres(dataset)

                m_ins_indices = torch.where((gap_value < thres) & ((top5_indices[:,0]==labels) | (top5_indices[:,1]==labels) | (top5_indices[:,2]==labels) | (top5_indices[:,3]==labels) | (top5_indices[:,4]==labels)))[0].squeeze()
                if m_ins_indices.dim() == 0:
                    m_ins_indices = m_ins_indices.unsqueeze(0)
                modify_indices = modify_top_gap(image_features[m_ins_indices], top5_indices[m_ins_indices], class_names, coder).cuda()
                if modify_indices.numel() != 0:
                    indices[m_ins_indices] = modify_indices
            # ===============================================================

            indices = indices.squeeze()
            total += labels.size(0)
            correct += (indices.cpu().numpy() == labels.cpu().numpy()).sum().item()

    return correct / total



def main(args):
    for model_type in ['ViT-B-32', 'ViT-B-16', 'ViT-L-14', 'ViT-L-14-336px']:
        clip_model, _ = get_clip_model(model_type)
        if model_type == "ViT-L-14-336px":
            img_size = 336
        else:
            img_size = 224

        # get dataset
        dataloader, _ = get_dataset(args.dataset, img_size=img_size)
        coder = CODER(args.dataset, args.text_types, clip_model)
        acc = classify(args.dataset, dataloader, clip_model, coder, args.ovo_modify)

        print(f"Model: {model_type} | Acc: {acc}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser('coder', parents=[get_args_parser()])
    args = parser.parse_args()    
    main(args)