import torch
import pickle
from CLIP import clip
import torch.nn.functional as F
from CODER.template import get_templates
import json
from tqdm import tqdm


class CODER():
    def __init__(self, dataset:str, text_type_list:list, clip_model, task_type='zero-shot') -> None:
        self.dataset = dataset
        self.class_name_lists = self.get_class_names()
        self.clip_model = clip_model
        self.text_type_list = text_type_list
        self.task_type = task_type

        if task_type == 'few-shot':
            self.update_class_names()

        if "name" in text_type_list:
            self.name_texts = self.get_name_texts()
        if "att" in text_type_list:
            self.att_texts = self.get_att_texts()
        if "ana" in text_type_list:
            self.ana_texts = self.get_ana_texts()
        if "syno" in text_type_list:
            self.syno_texts = self.get_syno_texts()
        if "ovo" in text_type_list:
            self.ovo_texts = self.get_ovo_texts()

        
        self.concate_general_texts()

    def get_class_names(self):
        if self.dataset == 'ImageNet-V2':
            dataset = "ImageNet"
        else:
            dataset = self.dataset
                
        file_path = f'datasets/class_name/{dataset}.txt'  
        category_list = []

        with open(file_path, 'r') as file:
            for line in file:
                cleaned_line = line.strip()
                category_list.append(cleaned_line)
        return category_list

    def get_name_texts(self):
        print("============================= get name texts =============================")
        name_texts = []
        with torch.no_grad():
            for class_name in tqdm(self.class_name_lists):
                class_name = class_name.replace('_', ' ')
                text_inputs = clip.tokenize(get_templates(self.dataset, 'name', class_name=class_name), truncate=True).cuda()
                text_features = self.clip_model.encode_text(text_inputs)
                text_features /= text_features.norm(dim=-1, keepdim=True)
                name_texts.append(text_features)

        name_texts = torch.cat(name_texts).cuda()
            
        return name_texts
    
    def get_att_texts(self):
        print("============================= get attribute texts =============================")
        if self.dataset == 'ImageNet-V2':
            dataset = "ImageNet"
        else:
            dataset = self.dataset

        data_path = f'expert_knowledge/attribute/{dataset}.pkl'
        f = open(data_path, 'rb')
        attribute = pickle.load(f)

        attribute_texts = [[] for _ in range(len(attribute))]
        cls_attribute_num = []

        device = "cuda" if torch.cuda.is_available() else "cpu"

        with torch.no_grad():
            for i in tqdm(range(len(attribute))):
                class_name = self.class_name_lists[i]
                cls_attribute_num.append(len(attribute_texts[i]))
                input_list = []
                for j in range(len(attribute[i])):
                    input_list = input_list + get_templates(self.dataset, "att", class_name=class_name, att=attribute[i][j], task_type=self.task_type)
                for input in input_list:
                    text_feautre = clip.tokenize(input, truncate=True).to(device)
                    text_feature = self.clip_model.encode_text(text_feautre)
                    text_features = F.normalize(text_feature)
                    attribute_texts[i].append(text_features)
                if len(attribute_texts[i]):
                    attribute_texts[i] = torch.cat(attribute_texts[i])
                else:
                    attribute_texts[i] = None


        return attribute_texts

    def get_ana_texts(self):
        print("============================= get analogous texts =============================")
        if self.dataset == 'ImageNet-V2':
            f = open(f"expert_knowledge/simile_class/ImageNet.pkl", 'rb')
        else:
            f = open(f"expert_knowledge/simile_class/{self.dataset}.pkl", 'rb')
        analogous_classes = pickle.load(f)

        analgous_texts = [[] for _ in range(len(analogous_classes))]
        cls_analogous_num = []

        device = "cuda" if torch.cuda.is_available() else "cpu"

        with torch.no_grad():
            for i in tqdm(range(len(analogous_classes))):
                class_name = self.class_name_lists[i]
                cls_analogous_num.append(len(analogous_classes[i]))
                input_list = []
                for j in range(len(analogous_classes[i])):
                    input_list = input_list + get_templates(self.dataset, "ana", class_name=class_name, ana=analogous_classes[i][j], task_type=self.task_type)
                for input in input_list:
                    text_inputs = clip.tokenize(input, truncate=True).to(device)
                    text_features = self.clip_model.encode_text(text_inputs)
                    text_features = F.normalize(text_features)
                    analgous_texts[i].append(text_features)
                if len(analgous_texts[i]):
                    analgous_texts[i] = torch.cat(analgous_texts[i])
                else:
                    analgous_texts[i] = None

        return analgous_texts
    
    def get_syno_texts(self):
        print("============================= get synonym texts =============================")
        if self.dataset == 'ImageNet-V2':
            dataset = "ImageNet"
        else:
            dataset = self.dataset

        f = open(f"expert_knowledge/synonym/{dataset}.json", 'r')
        synonyms = json.load(f)

        synonym_texts = [[] for _ in range(len(synonyms))]

        with torch.no_grad():
            for i in tqdm(range(len(synonyms))):
                if len(synonyms[i]["syno"]) == 0:
                    synonym_texts[i] = None
                
                else:
                    input_list = []
                    for synonym in synonyms[i]["syno"]:
                        input_list = input_list + get_templates(dataset, 'syno', synonym)
                    text_inputs = clip.tokenize(input_list, truncate=True).cuda()
                    text_features = self.clip_model.encode_text(text_inputs)
                    text_features = F.normalize(text_features)
                
                    synonym_texts[i] = text_features
        
        return synonym_texts

    def get_ovo_texts(self):
        print("============================= get ovo texts =============================")
        if self.dataset == 'ImageNet-V2':
            dataset = "ImageNet"
        else:
            dataset = self.dataset

        f = open(f"expert_knowledge/ovo_attribute/{dataset}.json", 'r')
        ovo_dict = json.load(f)
        ovo_texts = {}

        with torch.no_grad():
            for cls1, hard_cls_dict in tqdm(ovo_dict.items()):
                if cls1 not in ovo_texts.keys():
                    ovo_texts[cls1] = {}
                
                for cls2, ovos in hard_cls_dict.items():
                    ovo_texts[cls1][cls2] = []

                    input_list = []
                    for ovo in ovos:
                        template_ovo = get_templates(dataset, 'ovo', ovo=ovo, class_name=cls1, other_class_name=cls2)
                        self.ovo_template_num = len(template_ovo)
                        input_list = input_list + template_ovo

                    text_inputs = clip.tokenize(input_list, truncate=True).cuda()
                    text_features = self.clip_model.encode_text(text_inputs)
                    text_features = F.normalize(text_features)

                    ovo_texts[cls1][cls2] = text_features.cpu()

        return ovo_texts

    def concate_general_texts(self):
        total_texts_list = []
        total_general_texts_num_list = []
        
        for i in range(len(self.class_name_lists)):
            att_num, ana_num, name_num, syno_num = 0, 0, 0, 0
            text_list = []
            if "att" in self.text_type_list:
                if self.att_texts[i] is not None:
                    text_list.append(self.att_texts[i])
                    att_num = self.att_texts[i].shape[0]
            if "ana" in self.text_type_list:
                if self.ana_texts[i] is not None:
                    text_list.append(self.ana_texts[i])
                    ana_num = self.ana_texts[i].shape[0]
            if "name" in self.text_type_list:
                text_list.append(self.name_texts[i].unsqueeze(0))
                name_num = 1
            if "syno" in self.text_type_list:
                if self.syno_texts[i] is not None:
                    text_list.append(self.syno_texts[i])
                    syno_num = self.syno_texts[i].shape[0]
            
            total_texts_list.append(torch.cat(text_list, dim=0))
            total_general_texts_num_list.append((att_num+ana_num, name_num+syno_num))
        
        self.total_general_texts = torch.cat(total_texts_list, dim=0)
        self.total_general_texts_num_list = total_general_texts_num_list

    def get_general_CODER(self, images):
        images = images.cuda()
        if images.dtype == torch.float:
            images = F.normalize(images, p=2, dim=1)
        elif images.dtype == torch.float16:
            images /= images.norm(dim=-1, keepdim=True)

        self.total_general_texts = F.normalize(self.total_general_texts, p=2, dim=1)

        images_coder = images.float() @ self.total_general_texts.T.float()
 
        return images_coder, self.total_general_texts_num_list

    def get_ovo_CODER(self, images, cls1, cls2):
        images = images.cuda()
        if images.dtype == torch.float:
            images = F.normalize(images, p=2, dim=1)
        elif images.dtype == torch.float16:
            images /= images.norm(dim=-1, keepdim=True)

        images_coder = images @ self.ovo_texts[cls1][cls2].T.cuda()

        return images_coder
    
    def get_ovo_template_num(self):
        return self.ovo_template_num
    
    def update_class_names(self):
        if self.dataset == 'eurosat':
            file_path = f'datasets/class_name/{self.dataset}1.txt'  
            category_list = []

            with open(file_path, 'r') as file:
                for line in file:
                    cleaned_line = line.strip()
                    category_list.append(cleaned_line)
            
            self.class_name_lists = category_list
    
    def update_total_general_texts(self, new_texts):
        self.total_general_texts = F.normalize(new_texts, p=2, dim=1)


def main():
    text_type_list = ["proto"]
    CODER(text_type_list)

if __name__ == "__main__":
    main()