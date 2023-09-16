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
max_iter = 3000
eval_interval = 300
eval_iters = 200


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
"""
print(xb)

import torch.nn as nn
from torch.nn import functional as F
class BigramLLM(nn.Module):
  def __init__(self, vocab_size):
    super().__init__()
    self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)
  def forward(self, idx, targets = None):
    # idx and targets are both (B,T)_ tensor of integers
    logits = self.token_embedding_table(idx) # (B,T,C)

    if targets is not None:
      B,T,C = logits.shape
      logits = logits.view(B*T,C)
      targets = targets.view(B*T)
      loss = F.cross_entropy(logits, targets)
    else: loss = None
    return logits, loss

  def generate(self, idx, max_new_tokens):
    for _ in range(max_new_tokens):
      logits, loss = self(idx)
      #focus on the last time step
      logits = logits[:,-1,:] # (B,C)
      probs = F.softmax(logits, dim=-1)
      idx_next = torch.multinomial(probs, num_samples=1)
      idx = torch.cat((idx, idx_next), dim=1)
    return idx

model = BigramLLM(vocab_size).to(device)
logits, loss = model(xb,yb)
print(loss)
#print(logits.shape)
print(decode(model.generate(idx = torch.zeros((1,1), dtype=torch.long), max_new_tokens=100)[0].tolist()))

@torch.no_grad()
def estimate_loss():
  out = {}
  #model.eval()
  for split in ['train', 'test']:
      losses = torch.zeros(eval_iters)
      for k in range(eval_iters):
        x, y = get_batch(split)
        logits, loss = model(x, y)
        losses[k] = loss.item()
      out[split] = losses.mean()
  #model.train()
  return out

optim = torch.optim.AdamW(model.parameters(), lr=1e-3)
batch_size = 32

for steps in range(10000):
  if steps % eval_interval == 0:
    losses = estimate_loss()
    print(f"step {steps}: train loss {losses['train']:.4f}, val loss {losses['test']:.4f}")
  # sampled a batch of data
  xb, yb = get_batch("train")
  logits, loss = model(xb, yb)
  optim.zero_grad()
  loss.backward()
  optim.step()
print(loss.item())
context = torch.zeros((1,1), dtype=torch.long, device=device)
print(decode(model.generate(context, max_new_tokens=500)[0].tolist()))
