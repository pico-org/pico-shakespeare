import torch
import torch.nn as nn

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


voc = Vocabulary()
vocab = voc(text)
vocab_size = voc.get_vocab_size()
print(vocab_size)

tok = Tokenizer()

stoi,itos = tok(vocab)
print(tok.encoder("smruti"))
print(tok.decoder(tok.encoder("smruti")))