import os

from tqdm import tqdm
import json
import argparse
import time
 
from os.path import dirname
import logging
import numpy as np
import pandas as pd

import torch
 
#from transformers import AutoModel, AutoTokenizer
from TweetNormalizer import normalizeTweet
 

from transformers import AutoModel, BertTokenizer, AutoTokenizer # version: 4.5.1, pip install transformers
from transformers import GPT2Tokenizer, GPT2Model, AutoModelForCausalLM

#from aranizer import aranizer_sp32k  #  pip install aranizer
from util import write_feature_to_csv, load_word2vec, load_glove, strip_accent

##################### Arabic #####################
ARABERT_BASE = 'bert-base-cased'
ARABERT_LARGE = 'bert-large-cased'
MARBERT_BASE = ' '
MARBERT_LARGE = ' '

##################### Arabic GPT2 ARANIZER #####################
ARABIANGPT03  = 'b ' # https://huggingface.co/riotu-lab/ArabianGPT-03B
ARABIANGPT08 = 'b ' # https://huggingface.co/riotu-lab/ArabianGPT-08B

##################### Arabic LLM #####################
JAIS13  = 'llama-7b-hf' # https://huggingface.co/ 
JAIS40 = 'llama-13b-hf' # https://huggingface.co/ 


ROOT_DIR = dirname(dirname(__file__))
logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.INFO)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def find_start_end_pos(tokenizer):
    sentence = 'الطقس جميل جدا اليوم' # 句子中没有空格
    input_ids = tokenizer(sentence, return_tensors='pt')['input_ids'][0]
    start, end = None, None

    # find start, must in range [0, 1, 2]
    for start in range(0, 3, 1):
        # 因为decode有时会出现空格，因此我们显示的时候把这部分信息去掉看看
        outputs = tokenizer.decode(input_ids[start:]).replace('', '')
        if outputs == sentence:
            print (f'start: {start};  end: {end}')
            return start, None

        if outputs.startswith(sentence):
            break
   
    # find end, must in range [-1, -2]
    for end in range(-1, -3, -1):
        outputs = tokenizer.decode(input_ids[start:end]).replace('', '')
        if outputs == sentence:
            break
    
    assert tokenizer.decode(input_ids[start:end]).replace('', '') == sentence
    print (f'start: {start};  end: {end}')
    return start, end


# 找到 batch_pos and feature_dim
def find_batchpos_embdim(tokenizer, model, gpu):
    sentence = 'الطقس جميل جدا اليوم'
    inputs = tokenizer(sentence, return_tensors='pt')
    if gpu != -1: inputs = inputs.to('cuda')

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True).hidden_states # for new version 4.5.1
        outputs = torch.stack(outputs)[[-1]].sum(dim=0) # sum => [batch, T, D=768]
        outputs = outputs.cpu().numpy() # (B, T, D) or (T, B, D)
        batch_pos = None
        if outputs.shape[0] == 1:
            batch_pos = 0
        if outputs.shape[1] == 1:
            batch_pos = 1
        assert batch_pos in [0, 1]
        feature_dim = outputs.shape[2]
    print (f'batch_pos:{batch_pos}, feature_dim:{feature_dim}')
    return batch_pos, feature_dim



def extract_embedding(model_name, text_file,   gpu=-1, language='arabic', model_dir=None):

    print('='*30 + f' Extracting "{model_name}" ' + '='*30)
    start_time = time.time()

    # save last four layers
    layer_ids = [-4, -3, -2, -1]

    # save_dir
    save_dir = os.path.join("/home/slasher/araieval_arabicnlp24/task2", f'dat')
    if not os.path.exists(save_dir): os.makedirs(save_dir)

    # load model and tokenizer: offline mode (load cached files) 
    print('Loading pre-trained tokenizer and model...')
    if model_name in [ARABIANGPT03,ARABIANGPT08]:
        model = GPT2Model.from_pretrained(model_dir)
        tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=False)
    elif model_name in [JAIS13, JAIS40]:
        model = AutoModel.from_pretrained(model_dir)
        tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=False)
    else:
        model = AutoModel.from_pretrained(model_dir)
        tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=False)
    
     
    if gpu != -1 and model_name in [JAIS13,JAIS40]:
        model = model.half()

     
    if gpu != -1:
        torch.cuda.set_device(gpu)
        model.cuda()
    model.eval()
    feature_dim = -1
    json_file = [obj for obj in json.load(open(text_file))]
    print('Calculate embeddings...')
    start, end = find_start_end_pos(tokenizer) # only preserve [start:end+1] tokens
    batch_pos, feature_dim = find_batchpos_embdim(tokenizer, model, gpu) # find batch pos
    idx = 0
    #print( start, end, batch_pos, feature_dim )
    for  row in json_file:
        name = row['id'] 
        # ------------------ Text --------------------------------
        sentence = row['text'] 
        # -------------------- Image ------------------------------
        image_path = row['img_path']
        idx= idx+1
        print(f'Processing {name} ({idx}/{len(json_file)})...')

        # extract embedding from sentences
        embeddings = []
        if pd.isna(sentence) == False and len(sentence) > 0:
            inputs = tokenizer(sentence, return_tensors='pt')
            if gpu != -1: inputs = inputs.to('cuda')
            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True).hidden_states # for new version 4.5.1
                outputs = torch.stack(outputs)[layer_ids].sum(dim=0) # sum => [batch, T, D=768]
                outputs = outputs.cpu().numpy() # (B, T, D)
                #print(f'batch : {batch_pos}')
                #print(f'output shape: {outputs.shape}')
                if batch_pos == 0:
                    embeddings = outputs[0, start:end]
                    #print(f'embeddings shape: {embeddings.shape}')
                elif batch_pos == 1:
                    embeddings = outputs[start:end, 0]
                    #print(f'embeddings shape: {embeddings.shape}')

         
        # align with label timestamp and write csv file
        #print (f'feature dimension: {feature_dim }')

        csv_file = os.path.join(save_dir, f'{name}_txt.npy')
        #img = os.path.join(save_dir, f'{name}_img.npy')
        #imgT = np.load(img) 
        #print(f'image embeddings shape: {imgT.shape}')
        embeddings = np.array(embeddings).squeeze()
        if len(embeddings) == 0:
            embeddings = np.zeros((1, feature_dim))
        elif len(embeddings.shape) == 1:
            embeddings = embeddings[np.newaxis, :]
        np.save(csv_file, embeddings)
        print('saved')
 

    end_time = time.time()
    print(f'Total {len(json_file)} files done! Time used ({model_name}): {end_time - start_time:.1f}s.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", "-d", required=False, type=str,
                        default="~/araieval_arabicnlp24/task2/",
                        help="The absolute path to the training data")
    parser.add_argument("--file-name", "-f", required=False, type=str,
                        default="arabic_memes_propaganda_araieval_24_train.json",
                        help="Input file name, exptects jsonl")
    parser.add_argument("--out-file-name", "-o", required=False, type=str,
                        default="train_feats.json", help="Output feature file name")
    args = parser.parse_args()
    ## Load tweets and get features
    data_dir = args.data_dir
    data_file = args.file_name
    out_path = os.path.join(data_dir, "features")
    print("------------------------ Processing ids ------------------------")
    extract_embedding(model_name='MARBERT_BASE', 
                      text_file='/home/slasher/araieval_arabicnlp24/task2/data/arabic_memes_propaganda_araieval_24_test.json',
                      gpu = 0,
                      model_dir='/home/slasher/MER2023-Baseline/tools/transformers/MARBERTv2/' )#  MARBERTv2


