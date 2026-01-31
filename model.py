
import torch
import torch.nn as nn
from torch.nn import functional as F
import math
import random

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

    def __init__(self, vocab_size, n_embd, block_size, n_head, n_layer, dropout, device):
        super().__init__()
        self.device = device
        self.block_size = block_size
            
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.positional_encoding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.ModuleList([Block(n_embd, n_head, block_size, dropout) for i in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.positional_encoding_table(torch.arange(T, device=self.device)) # (T,C)
        x = tok_emb + pos_emb # (B,T,C)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size)

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
            # focus only on the last time step
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
