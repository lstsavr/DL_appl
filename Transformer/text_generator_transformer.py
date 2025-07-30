import torch
import torch.nn as nn
from torch.nn import functional as F
import random
import textwrap

batch_size = 64 
block_size = 256 
max_iters = 1000 
eval_interval = 500 
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200 
n_embd = 384 
n_head = 6 
n_layer = 6 
dropout = 0.2 
wrap_width = 50
# ------------

torch.manual_seed(325) 
file_name = "西游记.txt"

with open(file_name, 'r', encoding='utf-8') as f:
    text = f.read()

# here are all the unique characters that occur in this text
chars = sorted(list(set(text))) 
vocab_size = len(chars) 

stoi = { ch:i for i,ch in enumerate(chars) } 
itos = { i:ch for i,ch in enumerate(chars) } 
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long) 
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

print(f"File {file_name} has been read and processed.")


#-----定义函数与模型------------

# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,)) 
    x = torch.stack([data[i:i+block_size] for i in ix]) 
    y = torch.stack([data[i+1:i+block_size+1] for i in ix]) 
    x, y = x.to(device), y.to(device)
    return x, y 

@torch.no_grad() 
def estimate_loss(model):
    out = {}
    model.eval() 
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters) 
        for k in range(eval_iters):
            X, Y = get_batch(split) 
            logits, loss = model(X, Y) 
            losses[k] = loss.item()
        out[split] = losses.mean() 
    model.train() 
    return out

class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False) 
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size))) 

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # input of size (batch, time-step, channels)
        # output of size (batch, time-step, head size)
        B,T,C = x.shape 
        k = self.key(x)   
        q = self.query(x) 
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5 
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei) 
        v = self.value(x) # (B,T,hs)
        out = wei @ v # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        return out # (B, T, hs)

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out # (B, T, n_embd)

class FeedFoward(nn.Module):

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x) # (B, T, n_embd)

class Block(nn.Module):

    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head 
        self.sa = MultiHeadAttention(n_head, head_size) # sa = self attention
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x)) 
        x = x + self.ffwd(self.ln2(x))
        return x # (B, T, n_embd)

class GPTLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd) 
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)]) 
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None): 
        B, T = idx.shape 

        tok_emb = self.token_embedding_table(idx) 
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) 
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

    def generate(self, token_sequ, max_new_tokens):
        # token_sequ is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            tokens_input = token_sequ[:, -block_size:]
            logits, loss = self.forward(tokens_input)
            logits = logits[:, -1, :] # becomes (B, vocab_size)
            probs = F.softmax(logits, dim=-1) # (B, vocab_size)
            token_next = torch.multinomial(probs, num_samples=1) 
            token_sequ = torch.cat((token_sequ, token_next), dim=1) 
        new_tokens = token_sequ[:, -max_new_tokens:] 
        return new_tokens

def main():
    print(f"训练内容：{file_name}")
    model = GPTLanguageModel() 
    model = model.to(device)
    # print the number of parameters in the model
    print(sum(p.numel() for p in model.parameters())/1e6, 'M parameters')

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    for iter in range(max_iters):

        if iter % eval_interval == 0 or iter == max_iters - 1:
            losses = estimate_loss(model)
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        xb, yb = get_batch('train') 

        # evaluate the loss
        logits, loss = model(xb, yb) 
        optimizer.zero_grad(set_to_none=True) 
        loss.backward() 
        optimizer.step() 

    print ("Training complete")
    # generate from the model
    max_new_tokens = 500
    start_idx = random.randint(0, len(val_data)-block_size) # val_data 

    context = torch.zeros((1, block_size), dtype=torch.long, device=device) #(B, T) T = block_size
    real_next_tokens = torch.zeros((1, max_new_tokens), dtype=torch.long, device=device)

    context[0, :] = val_data[start_idx: start_idx+block_size] 
    context_str = decode(context[0].tolist()) 
    wrapped_context_str = textwrap.fill(context_str, width=wrap_width)

    real_next_tokens[0, :] = val_data[start_idx+block_size: start_idx+block_size+max_new_tokens] 
    real_next_str = decode(real_next_tokens[0].tolist()) # [0] 
    wrapped_real_next_str = textwrap.fill(real_next_str, width=wrap_width)

    generated_tokens = model.generate(context, max_new_tokens)
    generated_str = decode(generated_tokens[0].tolist())
    wrapped_generated_str = textwrap.fill(generated_str, width=wrap_width)

    print("context：")
    print(wrapped_context_str)
    print("generate:")
    print(wrapped_generated_str)
    print("Real next content:")
    print(wrapped_real_next_str)
    #open('more.txt', 'w').write(decode(model.generate(context, max_new_tokens=10000)[0].tolist()))
