
import torch
import torch.nn as nn
from torch.nn import functional as F
import math
import random


class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, **kwargs):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, 
                              stride=stride, padding=0, dilation=dilation, **kwargs)

    def forward(self, x):
        # x: [B, C, T]
        # Pad on the left
        x = F.pad(x, (self.padding, 0))
        return self.conv(x)


class ResidualCausalConvBlock(nn.Module):
    def __init__(self, channels, kernel_size=3):
        super().__init__()
        self.conv1 = CausalConv1d(channels, channels, kernel_size)
        self.relu1 = nn.ReLU()
        self.conv2 = CausalConv1d(channels, channels, kernel_size)
        self.relu2 = nn.ReLU()
        self.ln = nn.LayerNorm(channels)

    def forward(self, x):
        # x shape: [B, C, T]
        residual = x
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.conv2(out)
        # Residual connection
        out = out + residual
        out = self.relu2(out)
        
        # LayerNorm expects [B, T, C]
        out = out.transpose(1, 2)
        out = self.ln(out)
        out = out.transpose(1, 2)
        return out


class ConvByteCompressor(nn.Module):
    def __init__(self, d_model=512, ic=64, oc=128, compression_rate=4, n_layers=2):
        super().__init__()
        self.ic = ic
        self.oc = oc
        # 1. Raw Byte Embedding
        self.byte_embedding = nn.Embedding(256, ic)
        
        # 2. Residual Pre-processing
        self.residual_blocks = nn.ModuleList([
            ResidualCausalConvBlock(ic) for _ in range(n_layers)
        ])
        
        # 3. Convolutional Stem: This replaces the tokenizer's compression
        self.conv_stem = nn.Sequential(
            CausalConv1d(in_channels=ic, out_channels=oc, kernel_size=3),
            nn.ReLU(),
            CausalConv1d(in_channels=oc, out_channels=d_model, 
                         kernel_size=compression_rate, 
                         stride=compression_rate), # Key 'compression' step
            nn.ReLU(),
        )
        self.ln = nn.LayerNorm(d_model)

    def forward(self, byte_ids):
        # byte_ids shape: [batch, seq_len]
        x = self.byte_embedding(byte_ids) # [batch, seq_len, self.ic]
        
        # Conv1d expects [batch, channels, length]
        x = x.transpose(1, 2) 
        
        # Apply residual pre-processing
        for block in self.residual_blocks:
            x = block(x)
            
        # Compress the sequence length by 'compression_rate'
        x = self.conv_stem(x)
        
        # Return to Transformer-ready shape: [batch, compressed_len, d_model]
        x = self.ln(x.transpose(1, 2))
        return x


class ConvByteDecompressor(nn.Module):
    def __init__(self, d_model=512, ic=64, oc=128, vocab_size=256, compression_rate=4, n_layers=2):
        super().__init__()
        # 1. Upsample the sequence length back to original
        self.deconv1 = nn.ConvTranspose1d(in_channels=d_model, out_channels=oc, 
                                        kernel_size=compression_rate, 
                                        stride=compression_rate)
        self.relu1 = nn.ReLU()
        
        # 2. Residual Post-processing
        self.residual_blocks = nn.ModuleList([
            ResidualCausalConvBlock(oc) for _ in range(n_layers)
        ])
        
        # 3. Refine the sequence
        self.conv_refine = CausalConv1d(in_channels=oc, out_channels=ic, kernel_size=3)
        self.relu2 = nn.ReLU()
        # 4. Project to byte vocabulary
        self.lm_head = nn.Linear(ic, vocab_size)

    def forward(self, x):
        # x shape: [batch, compressed_len, d_model]
        x = x.transpose(1, 2) # [batch, d_model, compressed_len]
        
        # Decompress length
        x = self.deconv1(x) # [batch, oc, original_len]
        x = self.relu1(x)
        
        # Apply residential post-processing
        for block in self.residual_blocks:
            x = block(x)
            
        # Refine channels
        x = self.conv_refine(x) # [batch, ic, original_len]
        x = self.relu2(x)
        
        # Final head expects [batch, original_len, ic]
        x = x.transpose(1, 2)
        logits = self.lm_head(x) # [batch, original_len, vocab_size]
        
        return logits



class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size, n_embd, block_size, dropout):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)   # (B,T,16)
        q = self.query(x) # (B,T,16)
        v = self.value(x) # (B,T,16)

        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2, -1) * C**-0.5 # (B, T, 16) @ (B, 16, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)
        out = wei @ v # (B, T, T) @ (B, T, 16) -> (B, T, 16)
        return out

    
class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size, n_embd, block_size, dropout):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size, n_embd, block_size, dropout) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)
        self.attention_type = 'standard'

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        out = self.dropout(out)
        return out

class FeedForward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd, dropout, activation_type='relu'):
        super().__init__()
        self.activation_type = activation_type
        layers = [
            nn.Linear(n_embd, 4 * n_embd),
        ]
        
        if activation_type == 'arnold':
            self.activation = ArnoldActivation()
            layers.append(self.activation)
        else:
            layers.append(nn.ReLU())
            
        layers.extend([
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        ])
        
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head, block_size, dropout, activation_type='relu', 
                 attention_type='standard', Omega=0.618033988749895, Omega_rnd_std=None, init_K=1.0, 
                 residual_attention_mix=False, K_phase=False):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        if attention_type == 'arnold':
            self.saa = ArnoldAttentionLayer(n_embd, n_head, Omega, Omega_rnd_std, init_K, K_phase)
        if attention_type == 'standard' or residual_attention_mix is True:
            self.sa = MultiHeadAttention(n_head, head_size, n_embd, block_size, dropout)
        self.ffwd = FeedForward(n_embd, dropout, activation_type=activation_type)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.attention_type = attention_type
        self.residual_attention_mix = residual_attention_mix
        self.K_phase = K_phase

    def forward(self, x):
        if self.attention_type == 'arnold':
            x = self.ln1(x)
            if self.residual_attention_mix:
                x = x + self.saa(x) + self.sa(x)
            else:
                x = x + self.saa(x)
            x = x + self.ffwd(self.ln2(x))
        else:
            x = x + self.sa(self.ln1(x))
            x = x + self.ffwd(self.ln2(x))
        return x

class GPT(nn.Module):

    def __init__(self, vocab_size, n_embd, block_size, n_head, n_layer, dropout, device, use_conv_compressor=False, compression_rate=4, n_conv_layers=2):
        super().__init__()
        self.device = device
        self.block_size = block_size
            
        # each token directly reads off the logits for the next token from a lookup table
        self.use_conv_compressor = use_conv_compressor
        if use_conv_compressor:
            self.ic = 256 # 64
            self.oc = 512 # 128
            self.compressor = ConvByteCompressor(d_model=n_embd, ic=self.ic, oc=self.oc, compression_rate=compression_rate, n_layers=n_conv_layers)
            self.decompressor = ConvByteDecompressor(d_model=n_embd, vocab_size=vocab_size, ic=self.ic, oc=self.oc, compression_rate=compression_rate, n_layers=n_conv_layers)
        else:
            self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
            self.lm_head = nn.Linear(n_embd, vocab_size)
        self.positional_encoding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.ModuleList([Block(n_embd, n_head, block_size, dropout) for i in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm
        
        if use_conv_compressor:
            # We need an SOS token to represent the context BEFORE the first block
            self.sos_token = nn.Parameter(torch.randn(1, 1, n_embd))

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        if self.use_conv_compressor:
            x = self.compressor(idx) # x is (B, T_compressed, C)
        else:
            tok_emb = self.token_embedding_table(idx) # (B,T,C)
            pos_emb = self.positional_encoding_table(torch.arange(T, device=self.device)) # (T,C)
            x = tok_emb + pos_emb # (B,T,C)
            
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x) # (B, T_compressed or T, C)

        if self.use_conv_compressor:
            # HIERARCHICAL CAUSAL ALIGNMENT:
            # We must shift the compressed hidden states so that H[i] predicts block i+1.
            # Block 0 is predicted by the learned SOS token.
            sos = self.sos_token.expand(B, 1, -1)
            # Prepend SOS and drop the last hidden state to maintain sequence length (compressed)
            x_shifted = torch.cat((sos, x), dim=1)[:, :-1, :]
            
            # Decompress back to byte-level sequence
            logits = self.decompressor(x_shifted) # (B, T_expanded, vocab_size)
            # Crop to original sequence length T
            logits = logits[:, :T, :]
        else:
            logits = self.lm_head(x) # (B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens, temperature=1.0):
        # Temp > 1.0 = more random
        # Temp < 1.0 = more deterministic
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -self.block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # Correct prediction for the next token is at the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            
            # scale by temperature
            logits = logits / temperature
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx


def print_model_summary(model):
    """
    Prints a detailed summary of the model layers and their parameters.
    """
    print("\n" + "="*80)
    print(f"{'Layer (type)':<30} {'Output Shape':<25} {'Param #'}")
    print("-" * 80)
    
    total_params = 0
    trainable_params = 0
    
    # We use named_modules to get a flat list of layers
    # But we only want to print layers that have parameters or are interesting (like activation/norm)
    for name, module in model.named_modules():
        # Only print leaf modules (modules with no children) to avoid double counting
        if len(list(module.children())) == 0:
            layer_params = sum(p.numel() for p in module.parameters())
            total_params += layer_params
            if any(p.requires_grad for p in module.parameters()):
                trainable_params += layer_params
            
            # Try to get some info about the layer
            info = ""
            if isinstance(module, nn.Linear):
                info = f"in={module.in_features}, out={module.out_features}"
            elif isinstance(module, nn.Conv1d):
                info = f"in={module.in_channels}, out={module.out_channels}, k={module.kernel_size}, s={module.stride}"
            elif isinstance(module, nn.ConvTranspose1d):
                info = f"in={module.in_channels}, out={module.out_channels}, k={module.kernel_size}, s={module.stride}"
            elif isinstance(module, nn.Embedding):
                info = f"num={module.num_embeddings}, d={module.embedding_dim}"
            elif isinstance(module, nn.LayerNorm):
                info = f"shape={module.normalized_shape}"
            
            class_name = module.__class__.__name__
            print(f"{name:<30} {class_name:<25} {layer_params:,} {info}")

    print("=" * 80)
    print(f"Total params: {total_params:,}")
    print(f"Trainable params: {trainable_params:,}")
    print(f"Non-trainable params: {total_params - trainable_params:,}")
    print("=" * 80 + "\n")
