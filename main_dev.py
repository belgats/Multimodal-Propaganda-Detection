
import json
import random
import logging
import argparse
import os
from os.path import join, dirname, basename
from sklearn.dummy import DummyClassifier
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pandas as pd 
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset


from model import MLP,Attention
from mult import MULT
import sys
sys.path.append('.')

from scorer.task2 import evaluate
from format_checker.task2 import check_format

random.seed(10)
ROOT_DIR = dirname(dirname(__file__))

logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.INFO)

 

 

def read_one(file_path=None, dir_path=None):

    text_file_path = join(dir_path, file_path+'_txt.npy')
    img_file_path = join(dir_path, file_path+'_img.npy')
    assert os.path.exists(text_file_path)
    assert os.path.exists(img_file_path)
    #assert 
    txt_feature = np.load(text_file_path)
    img_feature = np.load(img_file_path)

    #txt_feature = txt_feature.squeeze()
    img_feature = img_feature.squeeze()


    #if len(txt_feature.shape) == 2:
        #txt_feature = np.mean(txt_feature, axis=0)
    #if len(img_feature.shape) == 2:
        #img_feature = np.mean(img_feature, axis=0)
    if len(txt_feature.shape) == 1:
       print(len(txt_feature.shape), txt_feature.shape, len(img_feature.shape), img_feature.shape)
       #txt_feature = txt_feature.squeeze(1)
    
    return txt_feature, img_feature

def align_to_text( texts, videos):
    #print(len(texts), len(videos))
    for ii in range(len(texts)):
        dst_len = len(texts[ii])
        texts[ii]  = func_mapping_feature(texts[ii],  dst_len)
        videos[ii] = func_mapping_feature(videos[ii], dst_len)
    return  texts, videos
# (seqlen, featdim) -> (dst_len, featdim)
def func_mapping_feature(feature, dst_len):
    #print(feature.dtype)
    featlen, featdim = feature.shape
    if featlen == dst_len:
        return feature
    elif featlen < dst_len:
        pad_feature = np.zeros((dst_len-featlen, featdim))
        feature = np.concatenate((pad_feature, feature), axis=0)
    else:
        if featlen // dst_len == featlen / dst_len:
            pad_len = 0
            pool_size = featlen // dst_len
        else:
            pad_len = dst_len - featlen % dst_len
            pool_size = featlen // dst_len + 1
        pad_feature = np.zeros((pad_len, featdim))
        feature = np.concatenate([pad_feature, feature]).reshape(dst_len, pool_size, featdim) # 相邻时刻特征取平均
        feature = np.mean(feature, axis=1)
    return feature#.to(dtype=torch.float32)
# batch-level: generate batch 
def pad_to_maxlen_pre_modality( texts, videos):
    
    text_maxlen  = max([len(feature) for feature in texts ])
    video_maxlen = max([len(feature) for feature in videos])
    for ii in range(len(texts)):
        #print(texts[ii].shape, videos[ii].shape)
        texts[ii]  = func_mapping_feature(texts[ii],  text_maxlen)
        videos[ii] = func_mapping_feature(videos[ii], video_maxlen)
    return  texts, videos

def read_features(text_file, dir_path, is_test=False):
    # define the data
    data = {'id': [], 'text': [], 'image': [], 'label': []}
    l2id = {'not_propaganda': 0, 'propaganda': 1}
    # Open the JSON file and load it
    with open(join(dir_path,f'data/{text_file}'), 'r') as f:
        json_file = json.load(f)
    text_dim = 0
    image_dim = 0
    # Iterate over the JSON file with tqdm
    for row in tqdm(json_file, desc="Reading features and labels", unit="rows"): 
        file_path = row['id'] 
        text, img = read_one( file_path, dir_path)
        data['id'].append(file_path)
        data['image'].append(img)
        data['text'].append(text )
        data['label'].append(row['class_label'])
        text_dim = text_dim + text.shape[0] 
        image_dim = image_dim + img.shape[0] 
        #print()
   
    #data['text'], data['image'] = pad_to_maxlen_pre_modality( data['text'], data['image'])
    data['text'], data['image'] = align_to_text( data['text'], data['image'])
 
    dimt = np.array(data['text'], dtype=object)[0].shape[1]

    dimi = np.array(data['image'], dtype=object)[0].shape[1]  
   
    return  pd.DataFrame.from_dict(data), text_dim, image_dim 

class ArAiDataset(Dataset):

    def __init__(self, label_path, audio_root, text_root, video_root, data_type, debug=False):
        assert data_type in ['train', 'dev', 'test']
        self.name2audio, self.name2labels = read_features(label_path, audio_root, task='whole', data_type=data_type, debug=debug)
        self.name2text,  self.name2labels, self.tdim = read_features(label_path, text_root,  task='whole', data_type=data_type, debug=debug)
        self.name2video, self.name2labels, self.vdim = read_features(label_path, video_root, task='whole', data_type=data_type, debug=debug)
        self.names = [name for name in self.name2audio if 1==1]

    def __getitem__(self, index):
        name = self.names[index]
        return torch.FloatTensor(self.name2audio[name]),\
               torch.FloatTensor(self.name2text[name]),\
               self.name2labels[name]['emo'],\
               name

    def __len__(self):
        return len(self.names)

    def get_featDim(self):
        print (f'audio dimension: {self.adim}; text dimension: {self.tdim}; video dimension: {self.vdim}')
        return self.adim, self.tdim, self.vdim
class MultimodalDataset(Dataset):
    def __init__(self, ids, text_data, image_data, labels, is_test=False):
        self.text_data = text_data
        self.image_data = image_data
        self.ids = ids
        self.is_test = is_test
        self.labels = labels
 
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        id = self.ids[index]
        text = self.text_data[index]
        image = self.image_data[index]
        label = self.labels[index]

        fdata = {
            'id': id,
            'text': text,
            'image': image
        }
        if not self.is_test:
            fdata['label'] = torch.tensor(label, dtype=torch.long)
            return fdata
        else:
            return fdata


# Define the training and testing functions
def train(model, train_loader, criterion, optimizer, device):
    model.train()
    train_loss = 0.0
    correct = 0
    for data in tqdm(train_loader):
        optimizer.zero_grad()
        
        multi_feat = torch.cat([data["text"], data["image"]], dim=1).to(dtype=torch.float32)
        multi_feat = multi_feat.to(device)
        text = data["text"].to(device)
        
        image = data["image"].to(device)
        #mask = data["text_mask"].to(device)
        #print(image.shape)
        labels = data['label'].to(device)
        #labels = labels.to(torch.long)
        _,preedected = model(  text,image)
        #preedected = preedected.to(dtype=torch.float32)
         
        #predicted_classes = torch.argmax(preedected, dim=1)
        loss = criterion(preedected , labels)#.to(dtype=torch.float32))
 
        #print(loss)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * labels.size(0)
        _, predicted = torch.max(preedected, 1)
        correct += (predicted == labels).sum().item()
    train_loss /= len(train_loader.dataset)
    accuracy = correct / len(train_loader.dataset)
    return train_loss, accuracy

def test(model, test_loader, criterion, device):
    model.eval()
    test_loss = 0.0
    correct = 0
    with torch.no_grad():
        for data in tqdm(test_loader):
            text = data["text"].to(device)
            image = data["image"].to(device)
            #mask = data["text_mask"].to(device)
            labels  = data['label'].to(device)
            outputs = model(text, image)
            loss = criterion(outputs, labels)
            test_loss += loss.item() * labels.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
    test_loss /= len(test_loader.dataset)
    accuracy = correct / len(test_loader.dataset)
    return test_loss, accuracy



def evaluate(model, test_loader, device):
    model.eval()
    predictions = []
    y_test_pred = []
    ids = []
    with torch.no_grad():
        for data in tqdm(test_loader):
            multi_feat = torch.cat([data["text"], data["image"]], dim=1)
            multi_feat = multi_feat.to(device)
            text = data["text"].to(device)
            image = data["image"].to(device)
            #mask = data["text_mask"].to(device)
            fetures, output = model( text,image)
            _, predicted = torch.max(output, 1)
            predictions.append(predicted)
            ids.append(data["id"])

    with open(f'task2C_test_MODOS.tsv', 'w') as f:
      f.write("id\tclass_label\trun_id\n")
      indx = 0
      id2l = {0:'not_propaganda', 1:'propaganda'}
      for i, line in enumerate(predictions):
        for indx, l in enumerate(line.tolist()):
          f.write(f"{ids[i][indx]}\t{id2l[l]}\tArabianGPT+CLIP\n")

 
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--layers', type=str, default='512,256,128', help='hidden size in model training')
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR', help='learning rate')
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ######################## LOAD DATASET #####################################
    dir_path  = "/home/slasher/araieval_arabicnlp24/task2/"
    train_file = 'arabic_memes_propaganda_araieval_24_train.json'
    validation_file = 'arabic_memes_propaganda_araieval_24_dev.json'
    test_file = 'arabic_memes_propaganda_araieval_24_test.json'

    label2idx, idx2label = {}, {}
    classes_label = ['not_propaganda', 'propaganda']  
    for ii, label in enumerate(classes_label): label2idx[label] = ii
    for ii, label in enumerate(classes_label): idx2label[ii] = label

    l2id = {'not_propaganda': 0, 'propaganda': 1}

    train_df, dim_dt, dim_dv = read_features(train_file, dir_path)
    train_df['label'] = train_df['label'].map(l2id)
    
    train_df = MultimodalDataset(train_df['id'], train_df['text'], train_df['image'], train_df['label'])

    validation_df, _,_ = read_features(validation_file, dir_path)
    validation_df['label'] = validation_df['label'].map(l2id)
    validation_df = MultimodalDataset(validation_df['id'], validation_df['text'], validation_df['image'], validation_df['label'])
    
 
 
    train_df = torch.utils.data.DataLoader(train_df, batch_size=1, shuffle=True, drop_last=True)
    validation_df = torch.utils.data.DataLoader(validation_df, batch_size=1, shuffle=True, drop_last=True)
   
    
 
    mode1l = MLP(input_dim = 768 ,
                        output_dim=2,
                        layers= args.layers)
    model = Attention(  text_dim=dim_dt,
                              image_dim=dim_dv,
                              output_dim=2,
                              hidden_dim=128,
                              layers=args.layers)
    model1 = MULT( text_dim=dim_dt,
                         image_dim=dim_dv,
                         output_dim1=2,
                         layers=  4,#[2, 4, 6],
                         dropout= 0.5,# [0.0, 0.1, 0.2, 0.3]
                         num_heads= 8,
                         hidden_dim = 256,
                         conv1d_kernel_size = 1,
                         grad_clip = 0.6 )
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Train the model
    num_epochs = 0
    for epoch in range(num_epochs):
        train_loss, acc = train(model, train_df, criterion, optimizer, device)
        #dev_loss, accuracy = test(model, validation_df, criterion, device)
        print('Epoch {}/{}: Train Loss = {:.4f}, Accuracy = {:.4f}'.format(epoch+1, num_epochs, train_loss, acc))

    #evaluate(model, validation_df, device)


