"""
From: https://github.com/yaohungt/Multimodal-Transformer
Paper: Multimodal Transformer for Unaligned Multimodal Language Sequences
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.transformer import TransformerEncoder

class MULT(nn.Module):
    def __init__(self, text_dim, image_dim, output_dim1 ,
                  layers, dropout, num_heads, hidden_dim, conv1d_kernel_size, grad_clip):
        super(MULT, self).__init__()
        self.d_l, self.d_a, self.d_v = 256, 32, 128
        combined_dim = self.d_l + self.d_a 
 
        # params: analyze args
        self.image_dim   = image_dim
        self.text_dim    = text_dim

        self.output_dim1 = output_dim1
 

        # params: analyze args
        self.attn_mask = True
        self.layers = layers # 4 
        self.dropout = dropout
        self.num_heads = num_heads # 8
        self.hidden_dim = hidden_dim # 128
        self.conv1d_kernel_size = conv1d_kernel_size # 5
        self.grad_clip = grad_clip
        
        # params: intermedia
        combined_dim =(self.hidden_dim + self.hidden_dim)
        output_dim = self.hidden_dim // 2
        
        # 1. Temporal convolutional layers
        
        self.proj_l = nn.Conv1d(text_dim,  self.d_l, kernel_size=self.conv1d_kernel_size, padding=0, bias=False)
        self.proj_i = nn.Conv1d(image_dim, self.d_l, kernel_size=self.conv1d_kernel_size, padding=0, bias=False)

        # 2. Crossmodal Attentions
        self.trans_l_with_i = self.get_network(self_type='la')
        self.trans_i_with_l = self.get_network(self_type='al')
 

        # 3. Self Attentions (Could be replaced by LSTMs, GRUs, etc.)
        self.trans_l_mem = self.get_network(self_type='l_mem', layers=3)
        self.trans_a_mem = self.get_network(self_type='a_mem', layers=3)
        self.trans_v_mem = self.get_network(self_type='v_mem', layers=3)

        # Projection layers
        self.proj1 = nn.Linear(combined_dim, combined_dim)
        self.proj2 = nn.Linear(combined_dim, combined_dim)
        self.out_layer = nn.Linear(combined_dim, output_dim)

        # cls layers
        self.fc_out_1 = nn.Linear(output_dim, output_dim1)
        self.cls_out = nn.Linear(output_dim, 2)
 


    def get_network(self, self_type='l', layers=-1):
        if self_type in ['l', 'al', 'vl']:
            embed_dim, attn_dropout = self.hidden_dim, self.dropout
        elif self_type in ['a', 'la', 'va']:
            embed_dim, attn_dropout = self.hidden_dim, self.dropout
        elif self_type in ['v', 'lv', 'av']:
            embed_dim, attn_dropout = self.hidden_dim, self.dropout
        elif self_type == 'l_mem':
            embed_dim, attn_dropout = 2*self.hidden_dim, self.dropout
        elif self_type == 'a_mem':
            embed_dim, attn_dropout = 2*self.hidden_dim, self.dropout
        elif self_type == 'v_mem':
            embed_dim, attn_dropout = 2*self.hidden_dim, self.dropout
        else:
            raise ValueError("Unknown network type")
        
        # TODO: Replace with nn.TransformerEncoder
        return TransformerEncoder(embed_dim=embed_dim,
                                  num_heads=self.num_heads,
                                  layers=max(self.layers, layers),
                                  attn_dropout=attn_dropout,
                                  relu_dropout=self.dropout,
                                  res_dropout=self.dropout,
                                  embed_dropout=self.dropout,
                                  attn_mask=self.attn_mask)


    def forward(self, texts,images): # audio_feat, text_feat, visual_feat ):# batch
        '''
            audio_feat: tensor of shape (batch, seqlen1, audio_in)
            video_feat: tensor of shape (batch, seqlen2, video_in)
            text_feat:  tensor of shape (batch, seqlen3, text_in)
        '''
        #texts = texts.to(dtype=torch.float32)
        #images = images.to(dtype=torch.float32)
        #texts = self.linear_t(texts)  # Pass through linear layer for non-sequential input
        #texts = texts.unsqueeze(0)  # Add sequence dimension
        #images = self.linear_i(images)  # Pass through linear layer for non-sequential input
        #images = images.unsqueeze(0)  # Add sequence dimension
         
        x_l = texts.transpose(1, 2)
        x_i = images.transpose(1, 2)
 
        # Project the textual/visual/audio features
        proj_x_l = self.proj_l(x_l).permute(2, 0, 1)
        proj_x_i = self.proj_i(x_i).permute(2, 0, 1)
        # (V,A) --> L
        h_l_with_is = self.trans_l_with_i(proj_x_l, proj_x_i,proj_x_i) 
        h_l_with_si = self.trans_i_with_l(proj_x_i, proj_x_l,proj_x_l)
         
        h_ls = torch.cat([h_l_with_is, h_l_with_is], dim=2)
        h_ls = self.trans_l_mem(h_ls)
        if type(h_ls) == tuple:
            h_ls = h_ls[0]
        last_h_l = last_hs = h_ls[-1]
         
        last_hs = torch.cat([last_h_l, last_h_l], dim=1)
 
        # A residual block
        last_hs_proj = self.proj2(F.dropout(F.relu(self.proj1(last_h_l), inplace=True), p=self.dropout, training=self.training))
        last_hs_proj += last_h_l
         
        features = self.out_layer(last_hs_proj)
        
        # store results
        out  = self.fc_out_1(features)

        interloss = torch.tensor(0).cuda()
 
        return features, out
