from CLIP import clip
import os


def get_clip_model(clip_model_name='ViT-L-14'):
    current_file_path = os.path.abspath(__file__)
    current_dir = os.path.dirname(current_file_path)
    clip_model_path = f"ckp/clip/{clip_model_name}.pt"
    clip_model_path = os.path.join(current_dir, clip_model_path)

    clip_model, preprocess = clip.load(clip_model_name, clip_model_path, 'cuda')
    clip_model.eval()
    clip_model.cuda()

    return clip_model, preprocess