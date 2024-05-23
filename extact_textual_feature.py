# *_*coding:utf-8 *_*
import os
import cv2
import math
import argparse
import numpy as np
from PIL import Image
import json 
import torch 
import timm # pip install timm==0.9.7
from transformers import AutoModel, AutoFeatureExtractor, AutoImageProcessor
from torchvision.io import read_image, ImageReadMode

# import config
import sys
sys.path.append('../../')
import config

##################### Pretrained models #####################
CLIP_VIT_BASE = 'clip-vit-base-patch32'  # https://huggingface.co/openai/clip-vit-base-patch32
CLIP_VIT_LARGE = 'clip-vit-large-patch14' # https://huggingface.co/openai/clip-vit-large-patch14
EVACLIP_VIT = 'eva02_base_patch14_224.mim_in22k' # https://huggingface.co/timm/eva02_base_patch14_224.mim_in22k
DATA2VEC_VISUAL = 'data2vec-vision-base-ft1k' # https://huggingface.co/facebook/data2vec-vision-base-ft1k
VIDEOMAE_BASE = 'videomae-base' # https://huggingface.co/MCG-NJU/videomae-base
VIDEOMAE_LARGE = 'videomae-large' # https://huggingface.co/MCG-NJU/videomae-large
DINO2_LARGE = 'dinov2-large' # https://huggingface.co/facebook/dinov2-large
DINO2_GIANT = 'dinov2-giant' # https://huggingface.co/facebook/dinov2-giant


def func_opencv_to_image(img):
    img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    return img

def func_opencv_to_numpy(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def func_read_frames(face_dir, vid):
    npy_path  = os.path.join(face_dir, vid, f'{vid}.npy')
    assert os.path.exists(npy_path), f'Error: {vid} does not have frames.npy!'
    frames = np.load(npy_path)
    return frames

# 策略3：相比于上面采样更加均匀 [将videomae替换并重新测试]
def resample_frames_uniform(frames, nframe=16):
    vlen = len(frames)
    start, end = 0, vlen
    
    n_frms_update = min(nframe, vlen) # for vlen < n_frms, only read vlen
    indices = np.arange(start, end, vlen / n_frms_update).astype(int).tolist()
    
    # whether compress into 'n_frms'
    while len(indices) < nframe:
        indices.append(indices[-1])
    indices = indices[:nframe]
    assert len(indices) == nframe, f'{indices}, {vlen}, {nframe}'
    return frames[indices]
    
def split_into_batch(inputs, bsize=32):
    batches = []
    for ii in range(math.ceil(len(inputs)/bsize)):
        batch = inputs[ii*bsize:(ii+1)*bsize]
        batches.append(batch)
    return batches


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run.')
    parser.add_argument('--dataset', type=str, default='MER2023', required=False, help='input dataset')
    parser.add_argument('--model_name', type=str, default=None, required=False, help='name of pretrained model')
    parser.add_argument('--model_dir', type=str, default=None,  required=False,help='model_dir of pretrained model')
    parser.add_argument('--feature_level', type=str,  required=False,default='UTTERANCE', help='feature level [FRAME or UTTERANCE]')
    parser.add_argument('--gpu', type=int, default=0, required=False,help='gpu id')
    params = parser.parse_args()

    print(f'==> Extracting {params.model_name} embeddings...')
    model_name = CLIP_VIT_LARGE
    model_dir = '' 

    # gain save_dir
    save_dir = os.path.join('/home/giveaway-6/arabic-nlp/task2/' , f'data')
    if not os.path.exists(save_dir): os.makedirs(save_dir)

    # load model
    if params.model_name in [CLIP_VIT_BASE, CLIP_VIT_LARGE, DATA2VEC_VISUAL, VIDEOMAE_BASE, VIDEOMAE_LARGE]: # from huggingface
        model = AutoModel.from_pretrained(  model_dir)
        processor  = AutoFeatureExtractor.from_pretrained( model_dir)
    elif params.model_name in [DINO2_LARGE, DINO2_GIANT]:
        
        model = AutoModel.from_pretrained(params.model_dir)
        processor  = AutoImageProcessor.from_pretrained(params.model_dir)
    elif params.model_name in [EVACLIP_VIT]: # from timm
         
        model = timm.create_model(params.model_name, pretrained=True, num_classes=0,pretrained_cfg_overlay=dict(file=params.model_dir))
        data_config = timm.data.resolve_model_data_config(model)
        transforms = timm.data.create_transform(**data_config, is_training=False)

    # 有 gpu 才会放在cuda上
    if params.gpu != -1:
        torch.cuda.set_device(params.gpu)
        model.cuda()
    model.eval()
    json_file = [obj for obj in json.load(
        open('/home/giveaway-6/arabic-nlp/task2/data/data/arabic_memes_propaganda_araieval_24_train.json' ))]
 
 
    print(f'Find total "{len(json_file)}" images.')
    for  row in json_file:
        name = row['id'] 
        # ------------------ Text --------------------------------
        sentence = row['text'] 
        # -------------------- Image ------------------------------
        img = read_image(os.path.join('/home/giveaway-6/arabic-nlp/task2/data/', row["img_path"]), ImageReadMode.RGB)
        print(f"Processing image '{row['id'] }' ({0}/{len(json_file)})...")
        
        # forward process [different model has its unique mode, it is hard to unify them as one process]
        # => split into batch to reduce memory usage
        with torch.no_grad():
             
            if params.model_name in [CLIP_VIT_BASE, CLIP_VIT_LARGE]:
                #frames = [func_opencv_to_image(frame) for frame in frames]
                inputs = processor(image=img, return_tensors="pt")['pixel_values']
                if params.gpu != -1: inputs = inputs.to("cuda")
                batches = split_into_batch(inputs, bsize=32)
                embeddings = []
                for batch in batches:
                    embeddings.append(model.get_image_features(batch)) # [58, 768]
                embeddings = torch.cat(embeddings, axis=0) # [frames_num, 768]
        embeddings = embeddings.detach().squeeze().cpu().numpy()
        EMBEDDING_DIM = max(EMBEDDING_DIM, np.shape(embeddings)[-1])

        # save into npy
        save_file = os.path.join(save_dir, f'{img}_img.npy')
         
        embeddings = np.array(embeddings).squeeze()
        if len(embeddings) == 0:
            embeddings = np.zeros((1, EMBEDDING_DIM))
        elif len(embeddings.shape) == 1:
            embeddings = embeddings[np.newaxis, :]
        np.save(save_file, embeddings)