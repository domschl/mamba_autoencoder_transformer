import os
import torch
import time
import math
import random
from data_loader import load_text_data, Tokenizer, DataLoader
from model import GPT

# Hyperparameters
batch_size = 64 # how many independent sequences will we process in parallel?
block_size = 128 # what is the maximum context length for predictions?
max_iters = 100000
eval_interval = 200
learning_rate = 4e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
if torch.backends.mps.is_available():
    device = 'mps'
eval_iters = 50
n_embd = 256
n_head = 8  
n_layer = 4
dropout = 0.1

torch.manual_seed(1337)
random.seed(1337)

# Load data
dataset_dir = os.path.join(os.path.dirname(__file__), 'dataset')
text = load_text_data(dataset_dir)
if not text:
    print("Error: No text found in dataset directory!")
    exit(1)

tokenizer = Tokenizer()
vocab_size = tokenizer.vocab_size

# Create data loader
train_loader = DataLoader(text, tokenizer, block_size, batch_size, device, cache_dir=dataset_dir, train_split=0.9)

# Model
model = GPT(vocab_size, n_embd, block_size, n_head, n_layer, dropout, device)
m = model.to(device)

# print(model)

# print the number of parameters in the model
print(str(sum(p.numel() for p in m.parameters())/1e6) + ' M parameters')

# Optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate) # , weight_decay=1e-2)

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = train_loader.get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# Training loop
print("Starting training...")
start_time = time.time()

for iter in range(max_iters):

    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print()
        print("=" * 50)
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        
        # Save checkpoint
        checkpoint_path = f"checkpoint_step_{iter}.pt"
        torch.save(model.state_dict(), checkpoint_path)

        # Generate sample
        context = torch.zeros((1, 1), dtype=torch.long, device=device)
        for temperature in [1.0]:
            print(f"Generating sample at step {iter}, temperature={temperature}...")
            print(tokenizer.decode(m.generate(context, max_new_tokens=128, temperature=temperature)[0].tolist()))
            print("-" * 50)

    # sample a batch of data
    xb, yb = train_loader.get_batch('train')

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    print(f"\rstep {iter}: train loss {loss.item():.4f}", end="")

print()
print(f"Training finished in {time.time() - start_time:.2f} seconds")
