
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

 

## 这两个模块都是用在 TFN 中的 (video|audio)
class MLPEncoder(nn.Module):
    '''
    The subnetwork that is used in TFN for video and audio in the pre-fusion stage
    '''

    def __init__(self, in_size, hidden_size, dropout):
        '''
        Args:
            in_size: input dimension
            hidden_size: hidden layer dimension
            dropout: dropout probability
        Output:
            (return value in forward) a tensor of shape (batch_size, hidden_size)
        '''
        super(MLPEncoder, self).__init__()
        # self.norm = nn.BatchNorm1d(in_size)
        self.drop = nn.Dropout(p=dropout)
        self.linear_1 = nn.Linear(in_size, hidden_size)
        self.linear_2 = nn.Linear(hidden_size, hidden_size)
        self.linear_3 = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        '''
        Args:
            x: tensor of shape (batch_size, in_size)
        '''
         
        # normed = self.norm(x)
        dropped = self.drop(x) # torch.mean(x, dim=1)
         
        y_1 = F.relu(self.linear_1(dropped))
        y_2 = F.relu(self.linear_2(y_1))
        y_3 = F.relu(self.linear_3(y_2))

        return y_3


# TFN 中的文本编码，额外需要lstm 操作 [感觉是audio|video]
class LSTMEncoder(nn.Module):
    '''
    The LSTM-based subnetwork that is used in TFN for text
    '''

    def __init__(self, in_size, hidden_size, dropout, num_layers=1, bidirectional=False):

        super(LSTMEncoder, self).__init__()

        if num_layers == 1:
            rnn_dropout = 0.0
        else:
            rnn_dropout = dropout

        self.rnn = nn.LSTM(in_size, hidden_size, num_layers=num_layers, dropout=rnn_dropout, bidirectional=bidirectional, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.linear_1 = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        '''
        Args:
            x: tensor of shape (batch_size, sequence_len, in_size)
            因为用的是 final_states ，所以特征的 padding 是放在前面的
        '''
        _, final_states = self.rnn(x)
        h = self.dropout(final_states[0].squeeze(0))
        y_1 = self.linear_1(h)
        return y_1


class MLP(nn.Module):
    def __init__(self, input_dim,  output_dim=2, layers='256,128', dropout=0.3):
        super(MLP, self).__init__()

        self.all_layers = []
        layers = list(map(lambda x: int(x), layers.split(',')))
        for i in range(0, len(layers)):
            self.all_layers.append(nn.Linear(input_dim, layers[i]))
            self.all_layers.append(nn.ReLU())
            self.all_layers.append(nn.Dropout(dropout))
            input_dim = layers[i]
        self.modul = nn.Sequential(*self.all_layers)
        self.fc_out = nn.Linear(layers[-1], output_dim)
        self.fc_out1 = nn.Linear(layers[-1], 1)

        
    def forward(self, inputs):
       
        features = self.modul(inputs) #torch.mean(inputs.squeeze(), dim=0)
         
        out = self.fc_out(features)
        out1 = self.fc_out1(features)
        #print(out)
        return features, out

class LSTMEncoder1(nn.Module):
    '''
    The LSTM-based subnetwork that is used in TFN for text
    '''

    def __init__(self, in_size, hidden_size, dropout, num_layers=1, bidirectional=False):

        super(LSTMEncoder, self).__init__()

        if num_layers == 1:
            rnn_dropout = 0.0
        else:
            rnn_dropout = dropout

        self.linear = nn.Linear(in_size, hidden_size)  # Added linear layer for non-sequential input
        self.rnn = nn.LSTM(hidden_size, hidden_size, num_layers=num_layers, dropout=rnn_dropout, bidirectional=bidirectional, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.linear_1 = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        '''
        Args:
            x: tensor of shape (batch_size, in_size)
        '''
        x = self.linear(x)  # Pass through linear layer for non-sequential input
        x = x.unsqueeze(1)  # Add sequence dimension
        _, final_states = self.rnn(x)
        h = self.dropout(final_states[0].squeeze(0))
        y_1 = self.linear_1(h)
        return y_1

class Attention(nn.Module):
    def __init__(self, text_dim,image_dim, output_dim=2, hidden_dim= 128, layers='512,256,128', dropout=0.3, feat_type='frm_unalign'):
        super(Attention, self).__init__()

        if feat_type in ['utt']:
            self.text_encoder  = MLPEncoder(text_dim,  hidden_dim, dropout)
            self.image_encoder = MLPEncoder(image_dim, hidden_dim, dropout)
        elif feat_type in ['frm_align', 'frm_unalign']:
            self.text_encoder  = LSTMEncoder(text_dim,  hidden_dim, dropout)
            self.image_encoder = LSTMEncoder(image_dim, hidden_dim, dropout)



        self.text_mlp  = self.MLP(text_dim,  layers, dropout)
        self.image_mlp = self.MLP(image_dim, layers, dropout)

        self.attention_mlp = MLPEncoder(hidden_dim *2, hidden_dim, dropout)

        layers_list = list(map(lambda x: int(x), layers.split(',')))
        #hiddendim = layers_list[-1] * 3
        #self.attention_mlp = self.MLP(hidden_dim * 3 , layers, dropout)

        self.fc_text   = nn.Linear(hidden_dim, hidden_dim)
        self.fc_img   = nn.Linear(hidden_dim, hidden_dim)
        self.fc_att   = nn.Linear(hidden_dim, 2)
        self.fc_out_1 = nn.Linear(hidden_dim, output_dim)
 
        #self.fc_out_3 = nn.Linear(layers_list[-1], output_dim3)
    
    def MLP(self, input_dim, layers, dropout):
        all_layers = []
        layers = list(map(lambda x: int(x), layers.split(',')))
        for i in range(0, len(layers)):
            all_layers.append(nn.Linear(input_dim, layers[i]))
            all_layers.append(nn.ReLU())
            all_layers.append(nn.Dropout(dropout))
            input_dim = layers[i]
        module = nn.Sequential(*all_layers)
        return module
    
    def forward(self, text_input, image_input):
        #text_hidden  = text_input.squeeze()    
        #image_hidden = image_input.squeeze()
        #print(text_input,image_input)
        #print(text_hidden.shape,image_hidden.shape)
        text_hidden  = self.text_encoder(text_input.to(dtype=torch.float32))   # [32, 128]
        image_hidden = self.image_encoder(image_input.to(dtype=torch.float32)) # [32, 128]
                # Compute attention weights for text and image
        text_weight = torch.sigmoid(self.fc_text(text_hidden))  # [batch_size, 1]
        image_weight = torch.sigmoid(self.fc_img(image_hidden))  # [batch_size, 1]

        # Apply weights
        weighted_text_hidden = text_hidden * text_weight
        weighted_image_hidden = image_hidden * image_weight

        # Concatenate weighted features
        multi_hidden1 = torch.cat([weighted_text_hidden, weighted_image_hidden], dim=1) # [batch_size, hidden_dim * 2]
 
        #multi_hidden1 = torch.cat([ text_hidden, image_hidden], dim=1) # [32, 384]
        #print(multi_hidden1.shape )
        attention = self.attention_mlp(multi_hidden1)
        attention = self.fc_att(attention)
        #print(attention.shape )
        attention = torch.unsqueeze(attention, 2) # [32, 3, 1]
        #print(attention.shape )
        multi_hidden2 = torch.stack([ text_hidden, image_hidden], dim=2) # [32, 128, 3]
        #print(multi_hidden2.shape, attention.shape)
        fused_feat = torch.matmul(multi_hidden2, attention)
        fused_feat = fused_feat.squeeze(axis=2) # [32, 128]
        out  = self.fc_out_1(fused_feat)

        #interloss = torch.tensor(0).cuda()
        return fused_feat, out
