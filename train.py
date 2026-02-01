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
pretrain_lr = 3e-4
main_lr = 1e-4
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
# Phase 1: Pre-train, Phase 2: GPT, Phase 3: Repair, etc.
# [Pre1, GPT1, Repair1, GPT2, Repair2, ...]
phase_schedule = [2000, 500, 500, 2000, 500, 2000, 500] 


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
optimizer = torch.optim.AdamW(model.parameters(), lr=pretrain_lr) # Start with pretrain LR

@torch.no_grad()
def estimate_loss(pretrain_mode=False):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = train_loader.get_batch(split)
            # In pretrain_mode, we compare against X (input reconstruction)
            logits, loss = model(X, X if pretrain_mode else Y, pretrain_mode=pretrain_mode)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# Training loop
print("Starting training...")
start_time = time.time()
last_output = None
mean_loss:None|float = None

# Phase Management
current_phase_idx = 0
phase_start_step = 0
current_phase_duration = phase_schedule[0]
pretrain_mode = True # Start in Pretrain
total_training_steps = max_iters + sum(phase_schedule)

for multi_iter in range(total_training_steps):
    
    # Check for phase switch
    if current_phase_idx < len(phase_schedule) and (multi_iter - phase_start_step) >= current_phase_duration:
        phase_start_step = multi_iter
        current_phase_idx += 1
        
        # Determine new phase properties
        if current_phase_idx < len(phase_schedule):
            current_phase_duration = phase_schedule[current_phase_idx]
            # Even indices (0, 2, 4...) are PRETRAIN/REPAIR, Odd are GPT
            new_pretrain_mode = (current_phase_idx % 2 == 0)
        else:
            # Schedule exhausted: GPT forever
            new_pretrain_mode = False
            
        # Update model mode and optimizer
        if pretrain_mode != new_pretrain_mode:
            pretrain_mode = new_pretrain_mode
            phase_type = "REPAIR" if (current_phase_idx % 2 == 0 and current_phase_idx > 0) else ("PRETRAIN" if current_phase_idx == 0 else "GPT")
            if current_phase_idx >= len(phase_schedule): phase_type = "FINAL-GPT"
            
            # LR Rule: Only index 0 gets pretrain_lr
            current_lr = pretrain_lr if current_phase_idx == 0 else main_lr
            
            print(f"\n" + "="*50)
            print(f"SWITCHING TO PHASE {current_phase_idx}: {phase_type} (Duration: {current_phase_duration if current_phase_idx < len(phase_schedule) else 'INF'}, LR: {current_lr})")
            print("="*50)
            
            mean_loss = None # Reset loss smoothing
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr

    iter = multi_iter - phase_start_step

    # every once in a while evaluate the loss on train and val sets
    if multi_iter % eval_interval == 0 or multi_iter == total_training_steps - 1:
        losses = estimate_loss(pretrain_mode=pretrain_mode)
        mean_loss = losses['train']
        print()
        print("=" * 50)
        phase_name = "PRETRAIN" if pretrain_mode else "GPT"
        print(f"step {iter} ({phase_name}): train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        
        # Save checkpoint
        checkpoint_path = f"checkpoint_{phase_name.lower()}_step_{iter}.pt"
        torch.save(model.state_dict(), checkpoint_path)

        # Generate sample (only relevant in Phase 2 or for checking reconstruction)
        if use_conv_compressor:
            context = torch.tensor([[32,32,32,32]], dtype=torch.long, device=device)
        else:
            context = torch.zeros((1, 1), dtype=torch.long, device=device)
        
        sample_iters = [0.8, 1.0] if not pretrain_mode else [1.0]
        for temperature in sample_iters:
            if pretrain_mode:
                # In pretraining, just show reconstruction of a real batch
                with torch.no_grad():
                    xb_sample, _ = train_loader.get_batch('val')
                    logits_sample, _ = model(xb_sample[:1], pretrain_mode=True)
                    preds_sample = torch.argmax(logits_sample, dim=-1)
                    orig_bytes = xb_sample[0][:64].tolist()
                    recon_bytes = preds_sample[0][:64].tolist()
                    print(f"Original:      {bytes(orig_bytes).decode('utf-8', errors='ignore')}")
                    print(f"Reconstructed: {bytes(recon_bytes).decode('utf-8', errors='ignore')}")
            else:
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
    logits, loss = model(xb, xb if pretrain_mode else yb, pretrain_mode=pretrain_mode)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    
    if mean_loss is None:
        mean_loss = loss.item()
    else:
        mean_loss = (mean_loss * 99 + loss.item()) / 100.0
        
    if last_output is None or time.time() - last_output > 1:
        phase_label = "pre" if pretrain_mode else "gpt"
        print(f"\rstep {iter} ({phase_label}): train loss {mean_loss:.4f}", end="")
        sys.stdout.flush()
        last_output = time.time()

print()
print(f"Training finished in {time.time() - start_time:.2f} seconds")
