import os
import sys
import torch
import time
import math
import random
import glob
import re
from data_loader import load_text_data, Tokenizer, DataLoader
from model import GPT

# Hyperparameters
batch_size = 32 # how many independent sequences will we process in parallel?
block_size = 128 # what is the maximum context length for predictions?
max_iters = 100000
eval_interval = 200
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
if torch.backends.mps.is_available():
    device = 'mps'
eval_iters = 50
n_embd = 256
n_head = 8  
n_layer = 24
dropout = 0.1

torch.manual_seed(1337)
random.seed(1337)

# Load data
dataset_dir = os.path.join(os.path.dirname(__file__), 'dataset')
text = load_text_data(dataset_dir)
if not text:
    print(f"Error: No text found in dataset directory: {dataset_dir}")
    exit(1)

tokenizer = Tokenizer()
vocab_size = tokenizer.vocab_size

# Create data loader
train_loader = DataLoader(text, tokenizer, block_size, batch_size, device, cache_dir=dataset_dir, train_split=0.9)
del text

# Model
model = GPT(vocab_size, n_embd, block_size, n_head, n_layer, dropout, device).to(device)

# print the number of parameters in the model
print(str(sum(p.numel() for p in model.parameters())/1e6) + ' M parameters')

# Optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# Check for existing checkpoints
def get_latest_checkpoint():
    checkpoint_dir = os.path.dirname(os.path.abspath(__file__))
    pattern = os.path.join(checkpoint_dir, "checkpoint_step_*.pt")
    checkpoints = glob.glob(pattern)
    if not checkpoints:
        return None, None
    
    # Extract step numbers and find the max
    def get_step(f):
        match = re.search(r"checkpoint_step_(\d+)\.pt", f)
        return int(match.group(1)) if match else -1
    
    latest_checkpoint = max(checkpoints, key=get_step)
    step = get_step(latest_checkpoint)
    return latest_checkpoint, step

start_iter = 0
checkpoint_path, latest_step = get_latest_checkpoint()
if checkpoint_path:
    print(f"Found checkpoint: {checkpoint_path}. Loading...")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Support both old (state_dict only) and new (dict with metadata) checkpoints
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_iter = checkpoint.get('iter', latest_step) + 1
        print(f"Loaded checkpoint from step {checkpoint.get('iter', latest_step)}")
    else:
        model.load_state_dict(checkpoint)
        start_iter = latest_step + 1
        print(f"Loaded model state from {checkpoint_path}, starting at step {start_iter}")
else:
    print("No checkpoint found. Starting from scratch.")

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
last_output = None
mean_loss = None

for iter in range(start_iter, max_iters):

    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        mean_loss = losses['train']
        print()
        print("=" * 50)
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        
        # Save checkpoint
        checkpoint_dir = os.path.dirname(os.path.abspath(__file__))
        checkpoint_file = f"checkpoint_step_{iter}.pt"
        checkpoint_path = os.path.join(checkpoint_dir, checkpoint_file)
        checkpoint = {
            'iter': iter,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'losses': losses,
        }
        torch.save(checkpoint, checkpoint_path)

        # Generate sample
        context = torch.zeros((1, 1), dtype=torch.long, device=device)
        for temperature in [1.0]:
            print(f"Generating sample at step {iter}, temperature={temperature}...")
            print(tokenizer.decode(model.generate(context, max_new_tokens=128, temperature=temperature)[0].tolist()))
            print("-" * 50)

    # sample a batch of data
    xb, yb = train_loader.get_batch('train')

    # evaluate the loss
    logits, loss = model(xb, yb)
    if mean_loss is not None:
        mean_loss = mean_loss * 0.9 + loss.item() * 0.1
        if last_output is None or time.time() - last_output > 1:
            print(f"\rstep {iter}: train loss {loss.item():.4f}, mean loss {mean_loss:.4f}", end="")
            sys.stdout.flush()
            last_output = time.time()
    else:
        mean_loss = loss.item()
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    print(f"\rstep {iter}: train loss {loss.item():.4f}", end="")

print()
print(f"Training finished in {time.time() - start_time:.2f} seconds")
