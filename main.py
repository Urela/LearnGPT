import torch
with open('input.txt', 'r', encoding='utf-8') as f:
  text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)
#print(f'length of characters: {len(text)}')
#print(text[:1000])
#print('|'.join(chars))
#print(f'length: {vocab_size}')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
block_size = 8 # maximum context length for predictions
batch_size = 32 # how many independent sequences will we process in paraell ?
max_iters = 5000
learning_rate = 1e-3
eval_interval = 500
eval_iters = 200
n_embed = 32
dropout = 0.2


torch.manual_seed(1337)

## Tokenisation
stoi = {c:i for i,c in enumerate(chars)}
itos = {i:c for i,c in enumerate(chars)}
encode = lambda x: [stoi[c] for c in x] 
decode = lambda x: ''.join([itos[i] for i in x]) 
#print(encode('the tall man'))
#print(decode(encode('the tall man')))

data = torch.tensor(encode(text), dtype=torch.long)
#print(data.shape, data.dtype)
#print(data[:1000])

## DataLoader
n = int(0.9*len(data))
train_data = data[:n]
test_data  = data[n:]


def get_batch(split='train'):
  data = train_data if split=='train' else test_data
  ix = torch.randint(len(data) - block_size, (batch_size,))
  x = torch.stack([data[i:i+block_size] for i  in ix])
  y = torch.stack([data[i+1:i+block_size+1] for i  in ix])
  x,y = x.to(device), y.to(device)
  return x, y

xb, yb = get_batch()
"""
print(f'inputs: {xb.shape} targets {yb.shape}')
for b in range(batch_size):
  for t in range(block_size):
    context = xb[b, :t+1]
    target  = yb[b, t]
    print(f"when input is {context.tolist()} the target: {target}")
print(xb)
"""
import torch.nn as nn
from torch.nn import functional as F

class Head(nn.Module):
  def __init__(self, head_size):
    super().__init__()
    self.key   = nn.Linear(n_embed, head_size, bias=False)
    self.query = nn.Linear(n_embed, head_size, bias=False)
    self.value = nn.Linear(n_embed, head_size, bias=False)
    self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
    self.dropout = nn.Dropout(dropout) 
  def forward(self, x):
    B,T,C = x.shape
    k = self.key(x)
    q = self.query(x)
    wei = q@k.transpose(-2,-1) * C**-0.5
    wei = wei.masked_fill( self.tril[:T, :T] **0, float('-inf'))
    wei = F.softmax(wei,dim=-1) # B,T,T
    wei = self.dropout(wei)
    v = self.value(x)
    out = wei @ v
    return out


class MultiHeadAttention(nn.Module):
  def __init__(self, num_heads, head_size):
    super().__init__()
    self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)]) 
    self.proj = nn.Linear(n_embed, n_embed)
    self.dropout = nn.Dropout(dropout) 
  def forward(self, x):
    out = torch.cat( [h(x) for h in self.heads], dim=-1 )
    out = self.dropout(self.proj(out)) 
    return out

class FeedForward(nn.Module):
  def __init__(self, n_embed):
    super().__init__()
    self.net = nn.Sequential(
      nn.Linear(n_embed, 4*n_embed), 
      nn.ReLU(),
      nn.Linear(4*n_embed, n_embed),  # projection layer
      nn.Dropout(dropout) 
    )
  def forward(self, x):
    return self.net(x)


class Block(nn.Module):
  def __init__(self, n_embed, n_head):
    super().__init__()
    head_size = n_embed// n_head
    self.sa = MultiHeadAttention(n_head, head_size)
    self.ffwd = FeedForward(n_embed)
    self.ln1 = nn.LayerNorm(n_embed)
    self.ln2 = nn.LayerNorm(n_embed)
  def forward(self, x):
    x = x + self.sa(self.ln1(x))
    x = x + self.ffwd(self.ln2(x))
    return x

class BigramLLM(nn.Module):
  def __init__(self):
    super().__init__()
    self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
    self.position_embedding_table = nn.Embedding(block_size, n_embed)
    #self.sa_head = Head(n_embed) # self attention head
    self.blocks = nn.Sequential(
      Block(n_embed, n_head=4),
      Block(n_embed, n_head=4),
      Block(n_embed, n_head=4),
      nn.LayerNorm(n_embed)
    )
    self.lm_head = nn.Linear(n_embed, vocab_size)

  def forward(self, idx, targets = None):
    # idx and targets are both (B,T)_ tensor of integers
    B,T = idx.shape
    tok_emb = self.token_embedding_table(idx) # (B,T,C1)
    pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (B,T,C1)
    x = tok_emb + pos_emb
    x = self.blocks(x) # (B,T,C)
    logits = self.lm_head(x) # (B,T,C2)

    if targets is not None:
      B,T,C = logits.shape
      logits = logits.view(B*T,C)
      targets = targets.view(B*T)
      loss = F.cross_entropy(logits, targets)
    else: loss = None
    return logits, loss

  def generate(self, idx, max_new_tokens):
    for _ in range(max_new_tokens):
      idx_cond = idx[:, -block_size:]
      logits, loss = self(idx_cond)
      #focus on the last time step
      logits = logits[:,-1,:] # (B,C)
      probs = F.softmax(logits, dim=-1)
      idx_next = torch.multinomial(probs, num_samples=1)
      idx = torch.cat((idx, idx_next), dim=1)
    return idx

model = BigramLLM().to(device)
optim = torch.optim.AdamW(model.parameters(), lr=learning_rate)
logits, loss = model(xb,yb)

@torch.no_grad()
def estimate_loss():
  out = {}
  model.eval()
  for split in ['train', 'test']:
      losses = torch.zeros(eval_iters)
      for k in range(eval_iters):
        x, y = get_batch(split)
        logits, loss = model(x, y)
        losses[k] = loss.item()
      out[split] = losses.mean()
  model.train()
  return out

for steps in range(max_iters):
  if steps % eval_interval == 0 or steps == max_iters - 1:
    losses = estimate_loss()
    print(f"step {steps}: train loss {losses['train']:.4f}, val loss {losses['test']:.4f}")
  # sampled a batch of data
  xb, yb = get_batch("train")
  logits, loss = model(xb, yb)
  optim.zero_grad(set_to_none=True)
  loss.backward()
  optim.step()
context = torch.zeros((1,1), dtype=torch.long, device=device)
print(decode(model.generate(context, max_new_tokens=500)[0].tolist()))
