import os
import torch
import torch.nn as nn
from torch.nn import functional as F
import time
from data_loader import load_text_data, Tokenizer, DataLoader
from model import ConvByteCompressor, ConvByteDecompressor, print_model_summary

# Hyperparameters (matching the user's latest train.py setup)
batch_size = 32
block_size = 64
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
if torch.backends.mps.is_available():
    device = 'mps'

n_embd = 512
compression_rate = 8
kernel_size = 11
base_c = 32
n_conv_compress_layers = 4
n_conv_decompress_layers = 2
vocab_size = 256

class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.ic = base_c * compression_rate
        self.oc = base_c * compression_rate
        # self.dropout = nn.Dropout(0.25)
        self.compressor = ConvByteCompressor(d_model=n_embd, ic=self.ic, oc=self.oc, 
                                            kernel_size=kernel_size,
                                            compression_rate=compression_rate, 
                                            n_layers=n_conv_compress_layers,
                                            causal=True)
        self.decompressor = ConvByteDecompressor(d_model=n_embd, vocab_size=vocab_size, 
                                                ic=self.ic, oc=self.oc, 
                                                kernel_size=kernel_size,
                                                compression_rate=compression_rate, 
                                                n_layers=n_conv_decompress_layers,
                                                causal=False)

    def forward(self, x, targets=None):
        # x: (B, T)
        # 1. Encode
        h = self.compressor(x) # (B, T_compressed, n_embd)
        # h = self.dropout(h)
        # 2. Decode (reconstruct the SAME x)
        logits = self.decompressor(h) # (B, T, vocab_size)
        
        if targets is None:
            targets = x # Autoencoder task
            
        B, T, C = logits.shape
        loss = F.cross_entropy(logits.view(B*T, C), targets.view(B*T))
        
        return logits, loss

# Load data
dataset_dir = os.path.join(os.path.dirname(__file__), 'dataset')
text = load_text_data(dataset_dir)
if not text:
    print("Error: No text found in dataset directory!")
    exit(1)

# Create data loader (we don't need a tokenizer for this test)
train_loader = DataLoader(text, None, block_size, batch_size, device, cache_dir=dataset_dir, train_split=0.9)

# Model
model = Autoencoder().to(device)
print_model_summary(model)

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

print("Starting Autoencoder Reconstruction Test...")
start_time = time.time()
mean_loss = None

for iter in range(1001): # Fast test
    xb, yb = train_loader.get_batch('train')
    
    # In autoencoder task, we want to reconstruct xb, not predict yb
    logits, loss = model(xb, xb)
    
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    
    if mean_loss is None:
        mean_loss = loss.item()
    else:
        mean_loss = (mean_loss * 0.99 + loss.item() * 0.01)
        
    if iter % 100 == 0:
        # Check actual reconstruction
        with torch.no_grad():
            preds = torch.argmax(logits, dim=-1)
            correct = (preds == xb).float().mean().item()
            print(f"step {iter}: loss {mean_loss:.4f}, pixel-accuracy {correct:.4f}")
            
            # Show a sample
            if iter % 500 == 0:
                sample_idx = 0
                original = xb[sample_idx][:50].tolist()
                reconstructed = preds[sample_idx][:50].tolist()
                print(f"Original:      {bytes(original).decode('utf-8', errors='ignore')}")
                print(f"Reconstructed: {bytes(reconstructed).decode('utf-8', errors='ignore')}")
                print("-" * 50)

print(f"Reconstruction test finished in {time.time() - start_time:.2f} seconds")
