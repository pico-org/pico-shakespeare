import torch
import torch.nn as nn
import config 

config = config.def_config()

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
        self.head_dim = config['head_dim']
        self.key = nn.Linear(self.n_emb,self.head_dim)
        self.query = nn.Linear(self.n_emb,self.head_dim)
        self.value = nn.Linear(self.n_emb,self.head_dim)

    def forward(self,x):
        pass

        
 