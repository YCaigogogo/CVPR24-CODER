def get_templates(dataset, text_type, class_name=None, att=None, ana=None, syno=None, ovo=None, other_class_name=None, task_type='zero-shot'):
    if text_type == "name":
        templates = get_name_templates(dataset, class_name)
    elif text_type == 'syno':
        templates = get_name_templates(dataset, syno)
    elif text_type == "att":
        templates = get_att_templates(dataset, class_name, att, task_type)
    elif text_type == "ana":
        templates = get_ana_templates(dataset, class_name, ana, task_type)
    elif text_type == "ovo":
        templates = get_ovo_templates(dataset, class_name, ovo, other_class_name)

    return templates

def get_name_templates(dataset, class_name):
    if dataset == 'eurosat':
        templates = [
            f'a photo of {class_name}, satellite domain.'
        ]
    elif dataset == 'food-101':
        templates = [
            f'a photo of {class_name}, a type of food.'
        ]
    elif dataset == 'oxford_flowers':
        templates = [
            f'a photo of a {class_name}, a type of flower.'
        ]
    elif dataset == 'Aircraft':
        templates = [
            f'a photo of a {class_name}, a type of aircraft.'
        ]
    elif dataset == 'oxford_pets' or dataset == 'oxford-pet':
        templates = [
            f'a photo of a {class_name}, a type of pet.'
        ]
    elif dataset == 'ucf101':
        templates = [
            f'a photo of a person doing {class_name}.'
        ]
    else:
        templates = [
            f'a photo of {class_name}'
        ]   
    return templates


def get_att_templates(dataset, class_name, att, task_type):
    if dataset == 'eurosat':
        templates = [
            f"A satellite photo of " + class_name + " which has " + att + "."
        ]
    elif dataset == 'oxford-pet' and task_type == 'zero-shot':
        templates = [
            "A pet photo of " + class_name + " which has " + att + "."
        ]
    elif dataset == 'food-101':
        templates = [
            "A food photo of " + class_name + " which has " + att + "."
        ]
    elif dataset == 'DTD' and task_type == 'zero-shot':
        templates = [
            'The texture of ' + class_name + ' is characterized by its ' + att + ' feature.'
        ]
    else:
        templates = [
            class_name + " which has " + att
        ]
        
    return templates

def get_ana_templates(dataset, class_name, analogous_class, task_type):
    if dataset == 'eurosat':
        templates = [
            f"A satellite photo of " + f"a {class_name} similar to {analogous_class}."
        ]
    elif dataset == 'oxford-pet' and task_type == 'zero-shot':
        templates = [
            "A pet photo of " + f"a {class_name} similar to {analogous_class}."
        ]
    elif dataset == 'food-101':
        templates = [
            "A food photo of " + f"a {class_name} similar to {analogous_class}."
        ]
    elif dataset == 'DTD' and task_type == 'zero-shot':
        templates = [
            f'The texture of {class_name} bears a likeness to that of {analogous_class} .'
        ]
    else:
        templates = [
            f"a {class_name} similar to {analogous_class}."
        ]

    return templates

def get_ovo_templates(dataset, class_name1, ovo_text, class_name2):
    if dataset == 'eurosat':
        templates = [
            f"The unique utilization of {ovo_text} in satellite photos makes {class_name1} different from {class_name2}.",
            f"Differences in {ovo_text} representation separate {class_name1} in satellite photos from {class_name2}.",
            f"Contrasting {class_name1} and {class_name2} in satellite photos reveals notable differences in the representation of {ovo_text}.",
            f"{class_name1} and {class_name2} showcase distinct visual features when it comes to {ovo_text} in satellite imagery.",
            f"The comparison of {class_name1} and {class_name2} in satellite photos underscores variations in the depiction of {ovo_text}.",
            f"In the context of satellite imagery, differences emerge between {class_name1} and {class_name2} in how they capture {ovo_text}.",
            f"Analyzing satellite photos highlights disparities in the portrayal of {ovo_text} between {class_name1} and {class_name2}.",
            f"{class_name1} and {class_name2} exhibit contrasting visual interpretations when representing {ovo_text} in satellite imagery."
        ]

    elif dataset == 'oxford-pet':
        templates = [
            f'A {class_name1}, a species of pet, can be distinguished from a {class_name2}, a species of pet, by the characteristics of {ovo_text}', 
            f'Because of {ovo_text}, a {class_name1}, a kind of pet, is different from a {class_name2}, a kind of pet.', 
            f'Due to their {ovo_text}, a {class_name1}, a pet type, displays a different demeanor compared to a {class_name2}, another pet type.', 
            f'Because of their {ovo_text}, a {class_name1}, a kind of pet, differs significantly from a {class_name2}, a kind of pet.'
        ]

    elif dataset == 'food-101':
        templates = [
            f'While both are popular dishes, {class_name1} is notably different from {class_name2} in that {ovo_text}.',
            f'While both are popular snacks, {class_name1} is notably different from {class_name2} in that {ovo_text}.',
            f'While both are popular dessert, {class_name1} is notably different from {class_name2} in that {ovo_text}.',
            f'In terms of food aspects, a food photo of {class_name1}, which has {ovo_text}, differs from {class_name2}.'
        ]

    elif dataset == 'places-365':
        templates = [
            f'{class_name1}, a type of places, can be distinguished from {class_name2}, a type of places, by the characteristics of {ovo_text}',
            f'Because of {ovo_text}, {class_name1}, a type of places, is different from {class_name2}, a type of places.'
        ]

    elif dataset == 'CUB200':
        templates = [
            f'{class_name1}, a type of bird, can be distinguished from {class_name2} by the characteristics of {ovo_text}', 
            f'Because of {ovo_text}, {class_name1}, a type of bird, is different from {class_name2}.', 
            f"{class_name1}, a type of bird, exhibits a unique {ovo_text} that sets it apart from {class_name2}.",
            f"{class_name1}, a type of bird, uses {ovo_text} in a way that differs from {class_name2}, making it distinctive.", 
            f"The unique expression of {ovo_text} is what separates {class_name1}, a type of bird, from {class_name2}.", 
            f"In handling {ovo_text}, {class_name1}, a type of bird, differs distinctly from {class_name2}.",
            f"{class_name1}, a type of bird, stands out in {ovo_text} compared to {class_name2}.", 
            f"The way {class_name1}, a type of bird, approaches {ovo_text} distinguishes it from {class_name2}.", 
            f"In {ovo_text} expression, {class_name1}, a type of bird, deviates from {class_name2}.", 
            f"Unique {ovo_text} utilization makes {class_name1}, a type of bird, different from {class_name2}.",
            f"Differences in {ovo_text} usage separate {class_name1}, a type of bird, from {class_name2}." 
        ]
    
    elif dataset == 'DTD':
        templates = [
            f"{class_name1}, a type of texture, exhibits a unique {ovo_text} that sets it apart from {class_name2}." 
            f"Because of {ovo_text}, {class_name1} is different from {class_name2}.",  
            f"In handling {ovo_text}, {class_name1} differs distinctly from {class_name2}.", 
            f"In {ovo_text} expression, {class_name1} deviates from {class_name2}.", 
        ]
        
    else:
        templates = [
            f"Because of {ovo_text}, {class_name1} is different from {class_name2}.", 
            f"{class_name1} is characterized by a distinct {ovo_text}, while {class_name2} isn't.", 
            f"The distinctive way in which {class_name1} handles {ovo_text} separates it significantly from {class_name2}.", 
            f"{class_name1} stands out in {ovo_text} compared to {class_name2}.", 
            f"The way {class_name1} approaches {ovo_text} distinguishes it from {class_name2}.", 
            f"In {ovo_text} expression, {class_name1} deviates from {class_name2}.",
            f"Differences in {ovo_text} usage separate {class_name1} from {class_name2}." 
        ]

    return templates

