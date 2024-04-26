import torch
import sys
import os
current_dir = os.path.dirname(__file__)   
parent_dir = os.path.dirname(current_dir) 
sys.path.insert(0, parent_dir)            
from CLIP import clip
import torch.nn.functional as F
from CODER.template import get_templates
import json
from tqdm import tqdm
import openai
import math
import time
from nltk.corpus import wordnet


class ATG():
    def __init__(self, api_key, model) -> None:
        self.api_key = api_key
        openai.api_key = api_key
        self.model = model
        # Used for filtering analogous texts
        self.class_text_feats = None
        self.clip_model = self.get_clip_model()

    def generate_text(self, text_type, cls=None, dataset=None, syno_expert="wordnet"):
        cls = [s.replace("_", " ") for s in cls]

        if text_type == "att":
            text_list = self.generate_att(cls)
        if text_type == "ana":
            text_list = self.generate_ana(cls, dataset)
        if text_type == "syno":
            text_list = self.generate_syno(cls, syno_expert)
        if text_type == "ovo":
            text_list = self.generate_ovo(cls)

        if text_list is not None:
            if text_type != "ovo":
                text_list = [item for item in text_list if item.strip()]
                text_list = [text.strip() for text in text_list]
            else:
                text_list[0] = [item for item in text_list[0] if item.strip()]
                text_list[1] = [item for item in text_list[1] if item.strip()]
                text_list[0] = [text.strip() for text in text_list[0]]
                text_list[1] = [text.strip() for text in text_list[1]]

        return text_list
    
    def generate_att(self, cls):
        if len(cls) != 1:
            assert 0, "invalid object class number"
        else:
            prompt = f"Q: What are useful visual features for distinguishing a lemur in a photo? \nA: There are several useful visual features to tell there is a lemur in a photo:\n \
_four-limbed primate\n\
_black, grey, white, brown, or red-brown\n\
_wet and hairless nose with curved nostrils\n\
_long tail\n\
_large eyes\n\
_furry bodies\n\
_clawed hands and feet\n\
Q: What are useful visual features for distinguishing a {cls[0]} in a photo?\n\
A: There are several useful visual features to tell there is a {cls[0]} in a photo:\n\
_"

            response = openai.ChatCompletion.create(model=self.model, messages=[{"role": "user", "content": prompt}])
            response = response["choices"][0]["message"]["content"]
            response = response.replace('\n', '')

            attribute_list = response.split('_')
            attribute_list = list(set(attribute_list))

            return attribute_list
        
        
    def generate_ana(self, cls, dataset=None):
        if len(cls) != 1:
            assert 0, "invalid object class number"
        else:
            prompt = f"Q:What other objects are cats visually similar to?\n\
A:\n\
_Tiger\n\
_Lion\n\
_Leopard\n\
_Garfield Cat\n\
_Cat-themed Products\n\
_Cat Toys\n\
Q:What other objects are {cls[0]} visually similar to?\n\
A:\n\
_"          
            if not self.class_text_feats:
                self.cls_name_feats, self.class_names = self.get_total_text_feat(dataset)

            analogous_class_list = []

            response = openai.ChatCompletion.create(model=self.model, messages=[{"role": "user", "content": prompt}])
            response = response["choices"][0]["message"]["content"]
            response = response.replace('\n', '')


            index = self.class_names.index(cls[0])
            
            other_category_feat = torch.cat((self.cls_name_feats[:index], self.cls_name_feats[index+1:]), dim=0).cuda()

            # find repeat class
            repeat_class_list, satisfy_class_list = self.get_repeat_class(response, other_category_feat, max_thres=0.9)
            analogous_class_list = analogous_class_list + satisfy_class_list
            if len(repeat_class_list) != 0:
                repeat_flag = 1
            else:
                repeat_flag = 0

            repeat_time = 0

            while (repeat_flag==1 and repeat_time<=5):
                time.sleep(1)
                repeat_time += 1

                repeat_class_name = ",".join(repeat_class_list)[:-1]
                prompt = f"Q: Helicopter has already appeared in the class set. Please give one more other classes which are visually similar to airplanes. Give me the class name without other words.\n\
A:\n\
_Drone\n\
Q: {repeat_class_name} has already appeared in the class set. Please give {len(repeat_class_list)} more other classes which are visually similar to {cls[0]}. Give me the class name without other words.\n\
A:\n\
_"
                response = openai.ChatCompletion.create(model=self.model, messages=[{"role": "user", "content": prompt}])
                response = response["choices"][0]["message"]["content"]
                response = response.replace('\n', '')
                repeat_class_list, satisfy_class_list = self.get_repeat_class(response, other_category_feat, max_thres=0.9)
                analogous_class_list = analogous_class_list + satisfy_class_list

                if len(repeat_class_list) == 0:
                    repeat_flag = 1

            return analogous_class_list


    def generate_syno(self, cls, syno_expert):
        if len(cls) != 1:
            assert 0, "invalid object class number"

        if syno_expert == "wordnet":
            synonym = self.generate_syno_wordnet(cls[0])
        elif syno_expert == "chatgpt":
            synonym = self.generate_syno_llm(cls[0])

        return synonym

    
    def generate_syno_wordnet(self, class_name):
        # try:
        noun_list = []
        for syn in wordnet.synsets(class_name):
            if syn.pos() == 'n':
                freq = 0
                for lemma in syn.lemmas():
                    freq+=lemma.count()
                noun_list.append((syn, freq))
        
        count_filter_list = sorted(noun_list, key=lambda x: x[1], reverse=True)

        sim_score_list = []
        proto_feat = self.get_text_feat(class_name)
        for syn, count in count_filter_list:
            definition = syn.definition()
            def_feat = self.get_text_feat(definition, use_template=False)
            sim_score_list.append(proto_feat@def_feat.T)
        max_index = sim_score_list.index(max(sim_score_list))
        choose_syn = count_filter_list[max_index][0]

        if choose_syn is not None:
            synonyms_list = []
            for lemma in choose_syn.lemmas():
                synonyms_list.append(lemma.name())

            element_to_remove = class_name
            if element_to_remove in synonyms_list:
                synonyms_list.remove(element_to_remove)

        else:
            synonyms_list = []
        
        return synonyms_list


    def generate_syno_llm(self, class_name):
        prompt = f"Q:What are the synonyms for forest?\n\
A:\n\
_Woodland\n\
_Woods\n\
_Thicket\n\
_Jungle\n\
_Timberland\n\
_Rainforest\n\
Q:What are the synonyms for {class_name}?\n\
A:\n\
_"          

        response = openai.ChatCompletion.create(model=self.model, messages=[{"role": "user", "content": prompt}])
        response = response["choices"][0]["message"]["content"]
        response = response.replace('\n', '')

        synonyms_list = response.split('_')
        synonyms_list = list(set(synonyms_list))

        return synonyms_list


    def generate_ovo(self, cls):
        if len(cls) != 2:
            assert 0, "invalid object class number"
        else:
            prompt = f"Q: What are different visual features between the river and the lake in a photo? Focus on their key differences.\n\
A: For River:\n\
_Flowing water\n\
_Narrow and elongated body of water\n\
For Lake:\n\
_Still or calm water\n\
_Larger and more circular or irregular in shape\n\
Q: What are different visual features between the {cls[0]} and the {cls[1]} in a photo? Focus on their key differences. You should decide the number of attributes to output based on the actual situation, not just 2. You should follow the format of the examples given above.\n\
A: For {cls[0]}:\n\
_"
            response = openai.ChatCompletion.create(model=self.model, messages=[{"role": "user", "content": prompt}])
            response = response["choices"][0]["message"]["content"]

            # postprocess
            response = response.replace('\n', '')

            del_idx1 = response.find("For")
            del_idx2 = response.find(":")

            cat1_response = response[:del_idx1]
            cat2_response = response[del_idx2+1:]

            cls1_att = cat1_response.split('_')
            cls2_att = cat2_response.split('_')[1:]
            
            return [cls1_att, cls2_att]
    
    # =========================== following are tool function ===========================
        
    def get_class_names(self, dataset):
        if dataset == 'ImageNet-V2':
            dataset = "ImageNet"
                
        file_path = f'datasets/class_name/{dataset}.txt'  
        category_list = []

        with open(file_path, 'r') as file:
            for line in file:
                cleaned_line = line.strip()
                category_list.append(cleaned_line)

        return category_list

        
    def get_clip_model(self, clip_model_name='ViT-L-14'):
        clip_model_path = f"CLIP/models/ckp/clip/{clip_model_name}.pt"
        clip_model, _ = clip.load(clip_model_name, clip_model_path, 'cuda')
        clip_model.eval()
        clip_model.cuda()
        return clip_model
    
    def get_total_text_feat(self, dataset):
        class_names = self.get_class_names(dataset)
        class_names = [s.replace("_", " ") for s in class_names]
        texts = [f"a photo of {s}" for s in class_names]

        text_features_list = []
        bs = 32

        bs_num = math.ceil(len(texts) / bs)

        for i in range(bs_num):
            with torch.no_grad():
                text_inputs = clip.tokenize(texts[i*bs:(i+1)*bs]).cuda()
                text_features = self.clip_model.encode_text(text_inputs)
                text_features = F.normalize(text_features)
                text_features_list.append(text_features)
        
        total_text_feats = torch.cat(text_features_list, dim=0)
        
        return total_text_feats, class_names
    
    def get_text_feat(self, text, use_template=True):
        with torch.no_grad():
            if use_template:
                text_inputs = clip.tokenize(f"a photo of {text}").cuda()
            else:
                text_inputs = clip.tokenize(text).cuda()
            text_features = self.clip_model.encode_text(text_inputs)
            text_features = F.normalize(text_features)
        return text_features
    
    def get_repeat_class(self, response, other_category_feat, max_thres=0.9):
        simile_class_list = response.split('_')
        simile_class_list = list(set(simile_class_list))

        repeat_class_list = []
        satisfy_class_list = []
        feat_list = []

        for simile_class in simile_class_list:
            simile_class_feat = self.get_text_feat(simile_class)
            feat_list.append(simile_class_feat)

        simile_feat = torch.cat(feat_list).cuda()
        cosine_sim = self.cal_cosine_sim(simile_feat, other_category_feat)
        max_values, _ = torch.max(cosine_sim, dim=1)
        max_values_list = max_values.tolist()
        for j, max_value in enumerate(max_values_list):
            if max_value > max_thres:
                repeat_class_list.append(simile_class_list[j])
                continue
            else:
                satisfy_class_list.append(simile_class_list[j])

        return repeat_class_list, satisfy_class_list
    
    def cal_cosine_sim(self, A, B):
        num_rows_A = A.size(0)
        num_rows_B = B.size(0)

        cosine_similarities = torch.zeros(num_rows_A, num_rows_B)

        for i in range(num_rows_A):
            for j in range(num_rows_B):
                cosine_similarities[i, j] = F.cosine_similarity(A[i], B[j], dim=0)
        
        return cosine_similarities

