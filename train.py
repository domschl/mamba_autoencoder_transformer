import os
import sys
import gc
import torch
import time
import math
import random
import glob
import re
import json
from data_loader import Tokenizer, DataLoader
from model import GPT

def get_current_configuration():
    configuration_version = 1
    valid_configuration = True
    if os.path.exists("current_configuration.json"):
        with open("current_configuration.json", "r") as f:
            try:
                current_configuration = json.load(f)
                if current_configuration['configuration_version'] != configuration_version:
                    print("Configuration version mismatch. Please update the configuration.")
                    valid_configuration = False
            except Exception as e:
                print(f"Error loading configuration: {e}, incompatible configuration, resetting to default configuration.")
                valid_configuration = False

    if valid_configuration:
        try:
            config = current_configuration['config']
            device = current_configuration['device']
            compile = current_configuration['compile']
            dataset_dir = current_configuration['dataset_dir']
        except Exception as e:
            print(f"Error loading configuration: {e}, incompatible configuration, resetting to default configuration.")
            valid_configuration = False
    if valid_configuration is False:
        # Calculated or environment-dependent variables
        print("Invalid or new configuration in current_configuration.json. Creating new configuration.")
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if torch.backends.mps.is_available():
            device = 'mps'
        if device == 'cuda':
            compile = True # use torch.compile() for speed
        else:
            compile = False
        dataset_dir = os.path.join(os.path.dirname(__file__), 'dataset')
        # Hyperparameters
        config = {
            'batch_size': 64, # how many independent sequences will we process in parallel?
            'block_size': 128, # what is the maximum context length for predictions?
            'max_iters': 100000,
            'eval_interval': 200,
            'learning_rate': 3e-4,
            'eval_iters': 50,
            'n_embd': [256, 256, 192, 128, 128, 192, 256, 256],  # Optional bottleneck architecture
            'n_head': 8,
            'n_layer': 8,
            'dropout': 0.1,
            'attention_type': ['mamba', 'standard', 'standard', 'mamba', 'standard', 'standard', 'standard', 'standard'], # Optional: List of length n_layer with elements 'standard' or 'mamba'
        }
        with open("current_configuration.json", "w") as f:
            json.dump({
                'configuration_version': configuration_version,
                'config': config,
                'device': device,
                'compile': compile,
                'dataset_dir': dataset_dir
            }, f, indent=4)
        print("Please review configuration file: current_configuration.json")
        sys.exit(0)
    return device, compile, config, dataset_dir

device, compile, config, dataset_dir = get_current_configuration()
announce_new_book = True

torch.manual_seed(1337)
random.seed(1337)

# Load data
tokenizer = Tokenizer()
config['vocab_size'] = tokenizer.vocab_size

# Create data loader
train_loader = DataLoader(dataset_dir, tokenizer, config['block_size'], config['batch_size'], device, cache_dir=dataset_dir, train_split=0.9)
batches_per_epoch = train_loader.get_dataset_batch_count()

# Model
model = GPT(
    vocab_size=config['vocab_size'], 
    n_embd=config['n_embd'], 
    block_size=config['block_size'], 
    n_head=config['n_head'], 
    n_layer=config['n_layer'], 
    dropout=config['dropout'], 
    device=device, 
    attention_type=config['attention_type']
).to(device)

# print the number of parameters in the model
print(str(sum(p.numel() for p in model.parameters())/1e6) + ' M parameters')

# Optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'])

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
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    except Exception as e:
        print(f"Error loading checkpoint: {e}, incompatible model, restarting training.")
        checkpoint = None
    
    if checkpoint is None:
        start_iter = 0
        checkpoint_path = None
        latest_step = None
    else:
        # Validate hyperparameters
        checkpoint_config = checkpoint.get('config')
        if checkpoint_config:
            mismatch = False
            # Parameters that MUST match for the model to be compatible
            critical_params = ['n_embd', 'n_head', 'n_layer', 'block_size', 'vocab_size', 'attention_type']
            for param in critical_params:
                if checkpoint_config.get(param) != config.get(param):
                    print(f"CRITICAL Hyperparameter mismatch for {param}: checkpoint={checkpoint_config.get(param)}, current={config.get(param)}")
                    mismatch = True
            
            if mismatch:
                print("Incompatible hyperparameters found in checkpoint. Restarting training from scratch.")
                start_iter = 0
                checkpoint_path = None
                latest_step = None
                # Reset model to ensure it's fresh (though it already is)
                model = GPT(
                    vocab_size=config['vocab_size'], 
                    n_embd=config['n_embd'], 
                    block_size=config['block_size'], 
                    n_head=config['n_head'], 
                    n_layer=config['n_layer'], 
                    dropout=config['dropout'], 
                    device=device, 
                    attention_type=config['attention_type']
                ).to(device)
                optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'])
            else:
                model.load_state_dict(checkpoint['model_state_dict'])
                if 'optimizer_state_dict' in checkpoint:
                    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    # Ensure optimizer state is on the correct device
                    for state in optimizer.state.values():
                        for k, v in state.items():
                            if isinstance(v, torch.Tensor):
                                state[k] = v.to(device)
                start_iter = checkpoint.get('iter', latest_step) + 1
                loaded_epoch = (checkpoint.get('iter', latest_step) * config['batch_size']) / batches_per_epoch
                print(f"Loaded checkpoint from step {checkpoint.get('iter', latest_step)} (epoch {loaded_epoch:.4f})")
        else:
            print("Checkpoint has no config information. Loading state_dict anyway (legacy support).")
            model.load_state_dict(checkpoint['model_state_dict'])
            if 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_iter = checkpoint.get('iter', latest_step) + 1
    
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
        losses = torch.zeros(config['eval_iters'])
        eval_loader_state = None
        eval_model_states = None
        for k in range(config['eval_iters']):
            if isinstance(config['attention_type'], list) and 'mamba' in config['attention_type'] or config['attention_type'] == 'mamba':
                X, Y, eval_loader_state = train_loader.get_book_sequential_batch(eval_loader_state, split=split, announce_new_book=False)
                # Reset states if new book started
                for i, (_, _, _, new_book) in enumerate(eval_loader_state):
                    if new_book and eval_model_states is not None:
                        for layer_idx in range(len(eval_model_states)):
                            if eval_model_states[layer_idx] is not None:
                                eval_model_states[layer_idx][i] = 0
                logits, loss, eval_model_states = model(X, Y, states=eval_model_states)
            else:
                X, Y = train_loader.get_batch(split)
                logits, loss, _ = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# Training loop
print(f"Starting training with {config['attention_type']} attention...")
start_time = time.time()
last_output = None
mean_loss = None
loader_state = None
model_states = None

for iter in range(start_iter, config['max_iters']):

    epoch = (iter * config['batch_size']) / batches_per_epoch
    # every once in a while evaluate the loss on train and val sets
    if iter % config['eval_interval'] == 0 or iter == config['max_iters'] - 1:
        losses = estimate_loss()
        mean_loss = losses['train']
        print()
        print("=" * 50)
        print(f"step {iter} (epoch {epoch:.4f}): train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        
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
            'config': config,
        }
        torch.save(checkpoint, checkpoint_path)

        # Generate sample
        context = torch.zeros((1, 1), dtype=torch.long, device=device)
        for temperature in [0.8, 1.0]:
            print(f"Generating sample at step {iter}, temperature={temperature}...")
            # For generation, we don't carry the training state
            print(tokenizer.decode(model.generate(context, max_new_tokens=128, temperature=temperature)[0].tolist()))
            print("-" * 50)
        gc.collect()
        if device == 'cuda':
            torch.cuda.empty_cache()

    # sample a batch of data
    if isinstance(config['attention_type'], list) and 'mamba' in config['attention_type'] or config['attention_type'] == 'mamba':
        xb, yb, loader_state = train_loader.get_book_sequential_batch(loader_state, announce_new_book=announce_new_book)
        # Check for new book starts to reset states
        if model_states is not None:
            for i, (_, _, _, new_book) in enumerate(loader_state):
                if new_book:
                    for layer_idx in range(len(model_states)):
                        if model_states[layer_idx] is not None:
                            model_states[layer_idx][i] = 0
    else:
        xb, yb = train_loader.get_batch('train')

    logits, loss, model_states = model(xb, yb, states=model_states)
    # Detach states to prevent backpropping through time across batches
    if model_states is not None:
        model_states = [s.detach() if s is not None else None for s in model_states]

    if mean_loss is not None:
        mean_loss = mean_loss * 0.9 + loss.item() * 0.1
        if last_output is None or time.time() - last_output > 1:
            print(f"\rstep {iter} (epoch {epoch:.2f}): train loss {loss.item():.4f}, mean loss {mean_loss:.4f}", end="")
            sys.stdout.flush()
            last_output = time.time()
    else:
        mean_loss = loss.item()
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

print()
print(f"Training finished in {time.time() - start_time:.2f} seconds")
