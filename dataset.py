import torch
import torch.nn as nn
from config import def_config

config = def_config()

# loading dataset
with open("/home/smruti/Desktop/git repos/pico-shakespeare/input.txt","r") as f:
    text = f.read()

class Vocabulary:
    def __init__(self):
        pass

    def __call__(self,text):
        self.vocab = sorted(list(set(text)))
        return self.vocab
    
    def get_vocab_size(self):
        return len(self.vocab)
    

class Tokenizer:
    def __init__(self):
        pass

    def __call__(self,char):
        self.char = char
        self.stoi = {j:i for i,j in enumerate(self.char)}
        self.itos = {i:j for i,j in enumerate(self.char)}
        return (self.stoi,self.itos)
    
    def encoder(self,s):
        return [self.stoi[c] for c in s]
    
    def decoder(self,i):
        return "".join(self.itos[_] for _ in i)
    
    def build_(self,text):
        return torch.tensor(self.encoder(text),dtype = torch.long)


class Train_Test_Split():
    def __init__(self,s_percentage):
        self.s_percentage = s_percentage
    
    def __call__(self,data):
        n = int(self.s_percentage*len(data))
        train_data = data[:n]
        val_data = data[n:]
        return train_data,val_data
    

class Batching:
  def __init__(self,train_data,val_data,config):
    self.train_data = train_data
    self.val_data = val_data
    self.batch_size = config['batch_size']
    self.block_size = config['seq_len']

  def __call__(self,split):
    self.split = split
    data = self.train_data if self.split == "train" else self.val_data
    ix = torch.randint(0,len(data)-self.block_size,(self.batch_size,))
    x = torch.stack([data[i:i+self.block_size] for i in ix])
    y = torch.stack([data[i+1:i+self.block_size+1] for i in ix])
    return x,y



class Embedding(nn.Module):
    def __init__(self,n_vocab,n_embed):
        super().__init__()
        self.n_vocab = n_vocab
        self.n_embed = n_embed
        self.embeddibg = nn.Embedding(self.n_vocab,self.n_embed)

    def forward(self,data):
        return self.embeddibg(data)
        


# voc = Vocabulary()
# vocab = voc(text)
# vocab_size = voc.get_vocab_size()
# print(vocab_size)

# tok = Tokenizer()

# stoi,itos = tok(vocab)
# data = tok.build_(text)
# tts = Train_Test_Split(0.9)
# train_data,val_data = tts(data)
# batching = Batching(train_data,val_data,config)
# x,y = batching("train")

# emb = Embedding(65,30)
# x = emb(x)

# print(x.shape)
# # print("/n")
# # print(y)