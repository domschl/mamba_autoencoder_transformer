import os
import sys
import gc
import torch
import time
import math
import random
import glob
import re
from data_loader import Tokenizer, DataLoader
from model import GPT

# Hyperparameters
batch_size = 64 # how many independent sequences will we process in parallel?
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
n_layer = 4
dropout = 0.1
attention_type = 'standard' # 'standard' or 'mamba'
compile = False # use torch.compile() for speed

torch.manual_seed(1337)
random.seed(1337)

# Load data
dataset_dir = os.path.join(os.path.dirname(__file__), 'dataset')

tokenizer = Tokenizer()
vocab_size = tokenizer.vocab_size

# Create data loader
train_loader = DataLoader(dataset_dir, tokenizer, block_size, batch_size, device, cache_dir=dataset_dir, train_split=0.9)

# Model
model = GPT(vocab_size, n_embd, block_size, n_head, n_layer, dropout, device, attention_type=attention_type).to(device)

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
    # Load to CPU first to be device-agnostic
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    # Support both old (state_dict only) and new (dict with metadata) checkpoints
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            # Ensure optimizer state is on the correct device
            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(device)
        start_iter = checkpoint.get('iter', latest_step) + 1
        print(f"Loaded checkpoint from step {checkpoint.get('iter', latest_step)}")
    else:
        model.load_state_dict(checkpoint)
        start_iter = latest_step + 1
        print(f"Loaded model state from {checkpoint_path}, starting at step {start_iter}")
    
    # Cleanup to save RAM
    del checkpoint
    gc.collect()
    if device == 'cuda':
        torch.cuda.empty_cache()
else:
    print("No checkpoint found. Starting from scratch.")

if compile:
    if hasattr(torch, 'compile'):
        if sys.platform == 'darwin':
            print("WARNING: torch.compile() is experimental on this platform")
        print("Compiling model...")
        if device == 'cuda':
            torch.set_float32_matmul_precision('high')
        model = torch.compile(model)
        print(f"Model compiled on {model.device}.")
    else:
        print("torch.compile() not implemented on this platform/version, skipping.")

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        eval_loader_state = None
        eval_model_states = None
        for k in range(eval_iters):
            if attention_type == 'mamba':
                X, Y, eval_loader_state = train_loader.get_book_sequential_batch(eval_loader_state, split=split)
                # Reset states if new book started
                for i, (_, _, _, new_book) in enumerate(eval_loader_state):
                    if new_book and eval_model_states is not None:
                        for layer_idx in range(len(eval_model_states)):
                            if eval_model_states[layer_idx] is not None:
                                eval_model_states[layer_idx][i] = 0
                logits, loss, eval_model_states = model(X, Y, states=eval_model_states)
            else:
                X, Y = train_loader.get_batch(split)
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# Training loop
print(f"Starting training with {attention_type} attention...")
start_time = time.time()
last_output = None
mean_loss = None
loader_state = None
model_states = None

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
        
        # Un-wrap compiled model for saving
        raw_model = model._orig_mod if hasattr(model, '_orig_mod') else model
        
        # Offload to CPU for cross-platform compatibility
        model_state_dict = {k: v.cpu() for k, v in raw_model.state_dict().items()}
        
        # Helper to offload dict of tensors (like optimizer state)
        def to_cpu(obj):
            if isinstance(obj, torch.Tensor):
                return obj.cpu()
            if isinstance(obj, dict):
                return {k: to_cpu(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [to_cpu(v) for v in obj]
            return obj
            
        optimizer_state_dict = to_cpu(optimizer.state_dict())
        
        checkpoint = {
            'iter': iter,
            'model_state_dict': model_state_dict,
            'optimizer_state_dict': optimizer_state_dict,
            'losses': losses,
            'attention_type': attention_type,
        }
        torch.save(checkpoint, checkpoint_path)

        # Generate sample
        context = torch.zeros((1, 1), dtype=torch.long, device=device)
        for temperature in [1.0]:
            print(f"Generating sample at step {iter}, temperature={temperature}...")
            # For generation, we don't carry the training state
            print(tokenizer.decode(model.generate(context, max_new_tokens=128, temperature=temperature)[0].tolist()))
            print("-" * 50)

    # sample a batch of data
    if attention_type == 'mamba':
        xb, yb, loader_state = train_loader.get_book_sequential_batch(loader_state)
        # Check for new book starts to reset states
        if model_states is not None:
            for i, (_, _, _, new_book) in enumerate(loader_state):
                if new_book:
                    for layer_idx in range(len(model_states)):
                        if model_states[layer_idx] is not None:
                            model_states[layer_idx][i] = 0
    else:
        xb, yb = train_loader.get_batch('train')

    # evaluate the loss
    if attention_type == 'mamba':
        logits, loss, model_states = model(xb, yb, states=model_states)
        # Detach states to prevent backpropping through time across batches
        if model_states is not None:
            model_states = [s.detach() if s is not None else None for s in model_states]
    else:
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

print()
print(f"Training finished in {time.time() - start_time:.2f} seconds")
