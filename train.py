import os
import sys
import torch
import time
import math
import random
from data_loader import load_text_data, Tokenizer, DataLoader
from model import GPT, print_model_summary

# Hyperparameters
batch_size = 32 # how many independent sequences will we process in parallel?
block_size = 64 # what is the maximum context length for predictions?
max_iters = 100000
eval_interval = 512
learning_rate = 1e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
if torch.backends.mps.is_available():
    device = 'mps'
eval_iters = 50
n_embd = 512
n_head = 8
n_layer = 24
dropout = 0.1
use_conv_compressor = True
kernel_size = 11
base_c = 32
compression_rate = 8
n_compress_layers = 4 # number of residual causal conv blocks
n_decompress_layers = 2 # number of residual causal conv blocks
compress_causal = True
decompress_causal = False
use_sos = False


torch.manual_seed(1337)
random.seed(1337)

# Load data
dataset_dir = os.path.join(os.path.dirname(__file__), 'dataset')``
text = load_text_data(dataset_dir)
if not text:
    print("Error: No text found in dataset directory!")
    exit(1)

if  use_conv_compressor:
    tokenizer = None
    vocab_size = 256
else:
    tokenizer = Tokenizer()
    vocab_size = tokenizer.vocab_size

# Create data loader
train_loader = DataLoader(text, tokenizer, block_size, batch_size, device, cache_dir=dataset_dir, train_split=0.9)
print("Data loaded")

# Model
model = GPT(vocab_size, n_embd, block_size, n_head, n_layer, dropout, device, 
            kernel_size=kernel_size, use_conv_compressor=use_conv_compressor, compression_rate=compression_rate, 
            n_compress_layers=n_compress_layers, n_decompress_layers=n_decompress_layers, 
            causal=compress_causal, decompress_causal=decompress_causal, use_sos=use_sos, base_c=base_c)
m = model.to(device)
print_model_summary(m)

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
last_output = None
mean_loss:None|float = None

for iter in range(max_iters):

    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        mean_loss = losses['train']
        print()
        print("=" * 50)
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        
        # Save checkpoint
        checkpoint_path = f"checkpoint_step_{iter}.pt"
        torch.save(model.state_dict(), checkpoint_path)

        # Generate sample
        if use_conv_compressor:
            context = torch.tensor([[32,32,32,32]], dtype=torch.long, device=device)
        else:
            context = torch.zeros((1, 1), dtype=torch.long, device=device)
        for temperature in [0.8, 1.0]:
            print(f"Generating sample at step {iter}, temperature={temperature}...")
            if use_conv_compressor:
                bytes_list = m.generate(context, max_new_tokens=128, temperature=temperature)[0].tolist()
                # Convert bytes to string
                text = bytes(bytes_list).decode('utf-8', errors='ignore')
                text = text.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
                old_text = ""
                while old_text != text:
                    old_text = text
                    text = text.replace('  ', ' ')
                print(text)
            else:
                print(tokenizer.decode(m.generate(context, max_new_tokens=128, temperature=temperature)[0].tolist()))
            print("-" * 50)

    # sample a batch of data
    xb, yb = train_loader.get_batch('train')

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    if mean_loss is None:
        mean_loss = loss.item()
    else:
        mean_loss = (mean_loss * 99 + loss.item()) / 100.0
    if last_output is None or time.time() - last_output > 1:
        print(f"\rstep {iter}: train loss {mean_loss:.4f}", end="")
        sys.stdout.flush()
        last_output = time.time()

print()
print(f"Training finished in {time.time() - start_time:.2f} seconds")
