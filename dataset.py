import torch
import torch.nn as nn

# loading dataset
with open("input.txt","r") as f:
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
    self.block_size = config['block_size']

  def __call__(self,split):
    self.split = split
    data = self.train_data if self.split == "train" else self.val_data
    ix = torch.randint(0,len(data)-self.block_size,(self.batch_size,))
    x = torch.stack([data[i:i+self.block_size] for i in ix])
    y = torch.stack([data[i+1:i+self.block_size+1] for i in ix])
    return x,y


