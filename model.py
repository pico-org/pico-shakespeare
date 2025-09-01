import torch
import torch.nn as nn
import config 
import torch.nn.functional as F

from config import def_config
conf = def_config()
import dataset as d



class FeedForwardLayer(nn.Module):
    def __init__(self,num_output):
        super().__init__()
        self.fc = nn.Sequential(
            nn.LazyLinear(num_output*2,bias=True),
            nn.ReLU(),
            nn.LazyLinear(num_output,bias=True)
        )
    def forward(self,x):
        return self.fc(x)
    


class SingleHead(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.n_emb = config['n_emb']
        self.head_dim = self.n_emb//config['num_head']
        self.key = nn.Linear(self.n_emb,self.head_dim)
        self.query = nn.Linear(self.n_emb,self.head_dim)
        self.value = nn.Linear(self.n_emb,self.head_dim)
        self.register_buffer('tril',torch.tril(torch.ones(config['seq_len'],config['seq_len'])))

    def forward(self,x):
        B,T,S = x.shape
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)
        wei = q @ k.transpose(-2, -1) * (self.head_dim**-0.5)
        wei = wei.masked_fill(self.tril[:T,:T]==0,float('-inf'))
        wei = F.softmax(wei,-1)
        out = wei@v
        return out # (batch_size,seq_len,head_dim)
    

class MultiHeadAttention(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.heads = nn.ModuleList([SingleHead(config) for _ in range (config['num_head'])])
        
    def forward(self,x):
        return torch.cat([h(x) for h in self.heads],dim = -1) #(batch_size,seq_len,n_emb)



