import os
import sys
import torch
import time
import math
import random
from data_loader import load_text_data, Tokenizer, DataLoader
from model import GPT, print_model_summary

# Hyperparameters
batch_size = 64 # how many independent sequences will we process in parallel?
block_size = 64 # what is the maximum context length for predictions?
max_iters = 100000
eval_interval = 512
learning_rate = 1e-4
# pretrain_lr = 3e-4 # Removed for compound loss
device = 'cuda' if torch.cuda.is_available() else 'cpu'
if torch.backends.mps.is_available():
    device = 'mps'
eval_iters = 50
n_embd = 384
n_head = 12
n_layer = 8
dropout = 0.2
use_conv_compressor = True
kernel_size = 11
base_c = 48
compression_rate = 4
n_compress_layers = 2 # number of residual causal conv blocks
n_decompress_layers = 0 # number of residual causal conv blocks
compress_causal = True
decompress_causal = False
use_sos = True
ae_loss_weight = 0.5 # Compound Loss Weight


torch.manual_seed(1337)
random.seed(1337)

# Load data
dataset_dir = os.path.join(os.path.dirname(__file__), 'dataset')
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
# Optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate) # , weight_decay=1e-2)

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        ae_losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = train_loader.get_batch(split)
            logits, loss, ae_loss = model(X, Y, return_ae_loss=True)
            losses[k] = loss.item()
            if ae_loss is not None:
                ae_losses[k] = ae_loss.item()
        out[split] = losses.mean()
        out[f'{split}_ae'] = ae_losses.mean()
    model.train()
    return out

# Training loop
print("Starting training...")
start_time = time.time()
last_output = None
mean_loss:None|float = None
mean_ae_loss:None|float = None

for iter in range(max_iters):


    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        mean_loss = losses['train']
        mean_ae_loss = losses['train_ae']
        print()
        print("=" * 50)
        print(f"step {iter}: train loss {losses['train']:.4f} (ae: {losses['train_ae']:.4f}), val loss {losses['val']:.4f} (ae: {losses['val_ae']:.4f})")
        
        # Save checkpoint
        checkpoint_path = f"checkpoint_step_{iter}.pt"
        torch.save(model.state_dict(), checkpoint_path)

        # Generate sample (only relevant in Phase 2 or for checking reconstruction)
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
    # Use compound loss: return_ae_loss=True
    logits, loss_gpt, loss_ae = model(xb, yb, return_ae_loss=True)
    
    total_loss = loss_gpt
    if loss_ae is not None:
        total_loss = total_loss + ae_loss_weight * loss_ae
        
    optimizer.zero_grad(set_to_none=True)
    total_loss.backward()
    optimizer.step()
    
    if mean_loss is None:
        mean_loss = loss_gpt.item()
        mean_ae_loss = loss_ae.item() if loss_ae is not None else 0.0
    else:
        mean_loss = (mean_loss * 99 + loss_gpt.item()) / 100.0
        if loss_ae is not None:
             mean_ae_loss = (mean_ae_loss * 99 + loss_ae.item()) / 100.0
        
    if last_output is None or time.time() - last_output > 1:
        print(f"\rstep {iter}: train loss {mean_loss:.4f} (ae: {mean_ae_loss:.4f})", end="")
        sys.stdout.flush()
        last_output = time.time()

print()
print(f"Training finished in {time.time() - start_time:.2f} seconds")
