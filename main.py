
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
from sklearn.model_selection import StratifiedKFold

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
from sklearn.metrics import f1_score

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
       txt_feature = txt_feature.squeeze(1)
    
    return txt_feature, img_feature

def align_to_text( texts, videos):
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
    return feature #.to(dtype=torch.float32)
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
 
    # Iterate over the JSON file with tqdm
    for row in tqdm(json_file, desc="Reading features and labels", unit="rows"): 
        file_path = row['id'] 
        text, img = read_one( file_path, dir_path)
        #print(text.shape , img.shape)
        #assert text.shape == img.shape, "Text and image shapes are not equal"
        data['id'].append(file_path)
        data['image'].append(img)
        data['text'].append(text )
         
 
        if is_test == False:
           data['label'].append(row['class_label'])
   
    #data['text'], data['image'] = pad_to_maxlen_pre_modality( data['text'], data['image'])
    data['text'], data['image'] = align_to_text( data['text'], data['image'])
    dimt = np.array(data['text'],dtype=object)[0].shape[1]  
 
    dimi = np.array(data['image'],dtype=object)[0].shape[1]  
   
    return  pd.DataFrame.from_dict(data), dimt, dimi 

def read_featuresl(text_file, dir_path, is_test=False):
    # define the data
    data = {'id': [], 'text': [], 'image': []}
    l2id = {'not_propaganda': 0, 'propaganda': 1}
    # Open the JSON file and load it
    with open(join(dir_path,f'data/{text_file}'), 'r') as f:
        json_file = json.load(f)
 
    # Iterate over the JSON file with tqdm
    for row in tqdm(json_file, desc="Reading features and labels", unit="rows"): 
        file_path = row['id'] 
        text, img = read_one( file_path, dir_path)
        #print(text.shape , img.shape)
        #assert text.shape == img.shape, "Text and image shapes are not equal"
        data['id'].append(file_path)
        data['image'].append(img)
        data['text'].append(text )
 
    #data['text'], data['image'] = pad_to_maxlen_pre_modality( data['text'], data['image'])
    data['text'], data['image'] = align_to_text( data['text'], data['image'])
    dimt = np.array(data['text'],dtype=object)[0].shape[1]  
 
    dimi = np.array(data['image'],dtype=object)[0].shape[1]  
   
    return  pd.DataFrame.from_dict(data), dimt, dimi 


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

class MultimodaldDataset(Dataset):
    def __init__(self, ids, text_data, image_data,  is_test=False):
        self.text_data = text_data
        self.image_data = image_data
        self.ids = ids
        self.is_test = is_test
  
 
    def __len__(self):
        return len(self.text_data)

    def __getitem__(self, index):
        id = self.ids[index]
        text = self.text_data[index]
        image = self.image_data[index]
 

        fdata = {
            'id': id,
            'text': text,
            'image': image
        }
 
        return fdata

# Define the training and testing functions
def train(model, train_loader, criterion, optimizer, device):
    model.train()
    train_loss = 0.0
    correct = 0
    all_predictions = []
    all_labels = []
    for data in tqdm(train_loader):
        optimizer.zero_grad()
        
        #multi_feat = torch.cat([data["text"], data["text"]], dim=1)
        #multi_feat = multi_feat.to(device)
        text = data["text"].to(device)
        #print(text.shape)
        image = data["image"].to(device)
        #mask = data["text_mask"].to(device)
        #print(mask.shape)
        labels = data['label'].to(device)
        features, output = model(text,image)
        #print(output.shape,labels )
        loss = criterion(output, labels)
        #print(loss)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * labels.size(0)
        _, predicted = torch.max(output, 1)
        correct += (predicted == labels).sum().item()
        all_predictions.extend(predicted.tolist())
        all_labels.extend(labels.tolist())
    train_loss /= len(train_loader.dataset)
    accuracy = correct / len(train_loader.dataset)
    f1 = f1_score(all_labels, all_predictions, average='weighted')
    return train_loss, accuracy,f1

def test(model, test_loader, criterion, device):
    model.eval()
    test_loss = 0.0
    correct = 0
    all_predictions = []
    all_labels = []
    with torch.no_grad():
        for data in tqdm(test_loader):
            text = data["text"].to(device)
            image = data["image"].to(device)
            #mask = data["text_mask"].to(device)
            labels = data['label'].to(device)
            f, output = model(text, image)
            loss = criterion(output, labels)
            test_loss += loss.item() * labels.size(0)
            _, predicted = torch.max(output, 1)
            correct += (predicted == labels).sum().item()
            all_predictions.extend(predicted.tolist())
            all_labels.extend(labels.tolist())
    test_loss /= len(test_loader.dataset)
    accuracy = correct / len(test_loader.dataset)
    f1 = f1_score(all_labels, all_predictions, average='weighted')
    return test_loss, accuracy,f1



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

    with open(f'task2C_MODOS.tsv', 'w') as f:
      f.write("id\tclass_label\trun_id\n")
      indx = 0
      id2l = {0:'not_propaganda', 1:'propaganda'}
      for i, line in enumerate(predictions):
        for indx, l in enumerate(line.tolist()):
          f.write(f"{ids[i][indx]}\t{id2l[l]}\tArabianGPT+CLIP\n")


def calculate_accuracy(predicted_file, true_labels_file):
    # Read predicted labels from the TSV file
    predicted_labels = {}
    with open(predicted_file, 'r') as f:
        next(f)  # Skip header
        for line in f:
            parts = line.strip().split('\t')
            predicted_labels[parts[0]] = parts[1]

    # Read true labels from the JSON file
    with open(true_labels_file, 'r') as f:
        true_labels = json.load(f)

    # Match predicted labels with true labels and calculate accuracy
    correct_count = 0
    total_count = len(true_labels)
    for idx, true_label in true_labels.items():
        predicted_label = predicted_labels.get(idx)
        if predicted_label == true_label:
            correct_count += 1

    accuracy = correct_count / total_count if total_count > 0 else 0.0
    return accuracy

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--layers', type=str, default='256,128', help='hidden size in model training')
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR', help='learning rate')
    args = parser.parse_args()
    #run_baselines(args.data_dir, args.test_split, args.train_file_name, args.test_file_name)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ######################## LOAD DATASET #####################################
    dir_path  = "/home/slasher/araieval_arabicnlp24/task2/"
    dir_test_path  = "/home/slasher/araieval_arabicnlp24/task2/dat"
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
    train_df_f =  train_df
    train_df = MultimodalDataset(train_df['id'], train_df['text'], train_df['image'], train_df['label'])

    validation_df, _, _= read_features(validation_file, dir_path)
    validation_df['label'] = validation_df['label'].map(l2id)
    validation_df = MultimodalDataset(validation_df['id'], validation_df['text'], validation_df['image'], validation_df['label'])
     

    test_df,_, _ = read_featuresl(test_file, dir_test_path,is_test=True)
    test_df = MultimodaldDataset(test_df['id'], test_df['text'], test_df['image'])#, eval_dataset['label'])
     

    train_df = torch.utils.data.DataLoader(train_df, batch_size=1, shuffle=True, drop_last=True)
    validation_df = torch.utils.data.DataLoader(validation_df, batch_size=1, shuffle=True, drop_last=True)
    test_dataset = torch.utils.data.DataLoader(test_df, batch_size=1, shuffle=True, drop_last=True)
    

    #model = MultimodalClassifier(num_classes=2)
    model2 = MLP(input_dim = dim_dt + dim_dv,
                        output_dim=2,
                        layers= args.layers)
    model = Attention(  text_dim=dim_dt,
                              image_dim=dim_dv,
                              output_dim=2,
                              hidden_dim=256,
                              layers=args.layers,)
    model22 = MULT( text_dim=dim_dt,
                         image_dim=dim_dv,
                         output_dim1=2,
                         layers=  4,#[2, 4, 6],
                         dropout= 0.5,# [0.0, 0.1, 0.2, 0.3]
                         num_heads= 8,
                         hidden_dim = 128,
                         conv1d_kernel_size = 1,
                         grad_clip = 0.6 )
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    num_folds = 3  # You can change this number as needed
    kfold = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)

 
    # Initialize lists to store evaluation metrics
    train_losses, train_accs, train_f1s = [], [], []
    val_losses, val_accs, val_f1s = [], [], []
    # Train the model
    num_epochs = 3
    X = train_df_f 
    X = X.drop(columns=['label'])
    y = train_df_f['label']
 
    for fold, (train_index, val_index) in enumerate(kfold.split(X, y)):
        print(f'Fold {fold+1}/{num_folds}:')
        # Split the data into train and validation sets for this fold
 
        train_fold = train_df_f.iloc[train_index].reset_index(drop=True)
        val_fold = train_df_f.iloc[val_index].reset_index(drop=True)

        #val_fold = torch.utils.data.Subset(train_df_f, val_index)
        #print(train_fold['id'])
        train_fold = MultimodalDataset(train_fold['id'], train_fold['text'], train_fold['image'], train_fold['label'])
        val_fold = MultimodalDataset(val_fold['id'], val_fold['text'], val_fold['image'], val_fold['label'])
        train_loader = torch.utils.data.DataLoader(train_fold, batch_size=1, shuffle=True, drop_last=True)
        val_loader = torch.utils.data.DataLoader(val_fold, batch_size=1, shuffle=True, drop_last=True)

        # Train the model for this fold
        for epoch in range(num_epochs):
            train_loss, acc, f1 = train(model, train_loader, criterion, optimizer, device)
            val_loss, val_acc, val_f1 = test(model, val_loader, criterion, device)
            print('Epoch {}/{}: Train Loss = {:.4f}, Accuracy = {:.4f}, F1 = {:.4f}'.format(epoch+1, num_epochs, train_loss, acc, f1))
            print('Epoch {}/{}: Validation Loss = {:.4f}, Accuracy = {:.4f}, F1 = {:.4f}'.format(epoch+1, num_epochs, val_loss, val_acc, val_f1))
    # Store evaluation metrics for this fold
    train_losses.append(train_loss)
    train_accs.append(acc)
    train_f1s.append( f1)
    val_losses.append(val_loss)
    val_accs.append(val_acc)
    val_f1s.append(val_f1)

    # Calculate average metrics across folds
    avg_train_loss = sum(train_losses) / num_folds
    avg_train_acc = sum(train_accs) / num_folds
    avg_train_f1 = sum(train_f1s) / num_folds
    avg_val_loss = sum(val_losses) / num_folds
    avg_val_acc = sum(val_accs) / num_folds
    avg_val_f1 = sum(val_f1s) / num_folds

    print('Average Training Loss = {:.4f}, Accuracy = {:.4f}, F1 = {:.4f}'.format(avg_train_loss, avg_train_acc, avg_train_f1))
    print('Average Validation Loss = {:.4f}, Accuracy = {:.4f}, F1 = {:.4f}'.format(avg_val_loss, avg_val_acc, avg_val_f1))

    num_epochs = 10
    for epoch in range(num_epochs):
        train_loss, acc,f1 = train(model, train_df, criterion, optimizer, device)
        dev_loss, accuracy,fcore = test(model, validation_df, criterion, device)
        print('Epoch {}/{}: Train Loss = {:.4f}, Accuracy = {:.4f},  F1 = {:.4f}'.format(epoch+1, num_epochs, train_loss, acc,f1))
        print('Epoch {}/{}: test Loss = {:.4f}, Accuracy = {:.4f},  F1 = {:.4f}'.format(epoch+1, num_epochs, dev_loss, accuracy,fcore))
         
    evaluate(model, test_dataset, device)



