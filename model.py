import torch
import torch.nn as nn
import torch.nn.functional as F
from config import def_config
import dataset as d

config = def_config()


with open("input.txt", "r") as f:
    text = f.read()

voc = d.Vocabulary()
vocab = voc(text)
vocab_size = voc.get_vocab_size()
print(f"Vocabulary size: {vocab_size}")

config['vocab_size'] = vocab_size

tok = d.Tokenizer()
stoi, itos = tok(vocab)
data = tok.build_(text)
tts = d.Train_Test_Split(0.9)
train_data, val_data = tts(data)
batching = d.Batching(train_data, val_data, config)


class FeedForwardLayer(nn.Module):
    def __init__(self, n_emb, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_emb, 4 * n_emb),
            nn.ReLU(),
            nn.Linear(4 * n_emb, n_emb),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        return self.net(x)


class SingleHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_emb = config['n_emb']
        self.head_dim = self.n_emb // config['num_head']
        self.key = nn.Linear(self.n_emb, self.head_dim, bias=False)
        self.query = nn.Linear(self.n_emb, self.head_dim, bias=False)
        self.value = nn.Linear(self.n_emb, self.head_dim, bias=False)
        self.dropout = nn.Dropout(config.get('dropout', 0.1))
        self.register_buffer('tril', torch.tril(torch.ones(config['block_size'], config['block_size'])))

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)
        
        # Scaled dot-product attention
        wei = q @ k.transpose(-2, -1) * (self.head_dim ** -0.5)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        out = wei @ v
        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.heads = nn.ModuleList([SingleHead(config) for _ in range(config['num_head'])])
        self.proj = nn.Linear(config['n_emb'], config['n_emb'])
        self.dropout = nn.Dropout(config.get('dropout', 0.1))
        
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config['n_emb'])
        self.attn = MultiHeadAttention(config)
        self.ln2 = nn.LayerNorm(config['n_emb'])
        self.ffwd = FeedForwardLayer(config['n_emb'], config.get('dropout', 0.1))

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.token_embedding_table = nn.Embedding(config['vocab_size'], config['n_emb'])
        self.position_embedding_table = nn.Embedding(config['block_size'], config['n_emb'])
        self.blocks = nn.Sequential(*[TransformerBlock(config) for _ in range(4)]) 
        self.ln_f = nn.LayerNorm(config['n_emb']) 
        self.lm_head = nn.Linear(config['n_emb'], config['vocab_size'])

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)  
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device)) 
        x = tok_emb + pos_emb  
        x = self.blocks(x)  
        x = self.ln_f(x)  
        logits = self.lm_head(x) 

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config['block_size']:]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :]  
            probs = F.softmax(logits, dim=-1)  
            idx_next = torch.multinomial(probs, num_samples=1)  
            idx = torch.cat((idx, idx_next), dim=1) 
        return idx


def train_model():
    model = Model(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'])

    model.train()
    for epoch in range(config.get('epochs', 5000)):
        # Sample a batch of data
        xb, yb = batching("train")
        
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        
        if epoch % 1000 == 0:
            print(f"Epoch [{epoch}/{config.get('epochs', 5000)}], Loss: {loss.item():.4f}")
    
    return model


def generate_text(model, tokenizer, max_new_tokens=500):
    """Generate text using the trained model"""
    model.eval()
    context = torch.zeros((1, 1), dtype=torch.long)
    generated_idx = model.generate(context, max_new_tokens)[0].tolist()
    generated_text = tokenizer.decoder(generated_idx)
    return generated_text


if __name__ == "__main__":

    print("Starting training...")
    trained_model = train_model()
    print("\nGenerating text...")
    generated = generate_text(trained_model, tok, max_new_tokens=500)
    print("Generated text:")
    print(generated)