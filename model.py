
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

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        out = self.dropout(out)
        return out

class MambaBlock(nn.Module):
    """ Simplified Mamba-style Selective SSM block """

    def __init__(self, n_embd, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.n_embd = n_embd
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.n_embd)
        self.dt_rank = math.ceil(self.n_embd / 16)

        self.in_proj = nn.Linear(self.n_embd, self.d_inner * 2, bias=False)

        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=True,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
        )

        self.activation = nn.SiLU()

        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        # S4D real initialization for A
        A = torch.arange(1, self.d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))

        self.out_proj = nn.Linear(self.d_inner, self.n_embd, bias=False)

    def selective_scan(self, x, dt, A, B, C, D, h=None):
        b, l, d_in = x.shape
        n = A.shape[1]
        
        # Discretize A and B (ZOH approximation)
        dA = torch.exp(torch.einsum('bld,dn->bldn', dt, A)) # (b, l, d_in, n)
        dB = torch.einsum('bld,bln->bldn', dt, B) # (b, l, d_in, n)
        
        # Recurrent scan
        if h is None:
            h = torch.zeros(b, d_in, n, device=x.device)
        
        ys = []
        for i in range(l):
            h = dA[:, i, :, :] * h + dB[:, i, :, :] * x[:, i, :].unsqueeze(-1)
            y_i = torch.einsum('bdn,bn->bd', h, C[:, i, :])
            ys.append(y_i)
        
        y = torch.stack(ys, dim=1)
        return y + x * D, h

    def forward(self, x, h=None):
        (b, l, d) = x.shape

        x_and_res = self.in_proj(x)  # (b, l, 2 * d_inner)
        (x, res) = x_and_res.split(split_size=[self.d_inner, self.d_inner], dim=-1)

        x = x.transpose(1, 2)
        x = self.conv1d(x)[:, :, :l]
        x = x.transpose(1, 2)

        x = self.activation(x)

        # Compute SSM parameters
        (d_in, n) = self.A_log.shape
        A = -torch.exp(self.A_log.float())  # (d_in, n)
        D = self.D.float()

        x_dbl = self.x_proj(x)  # (b, l, dt_rank + 2*n)

        (dt, B, C) = x_dbl.split(split_size=[self.dt_rank, n, n], dim=-1)
        dt = F.softplus(self.dt_proj(dt))  # (b, l, d_in)

        y, h = self.selective_scan(x, dt, A, B, C, D, h=h)

        y = y * self.activation(res)

        output = self.out_proj(y)

        return output, h

class FeedForward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd, dropout, out_n_embd=None):
        super().__init__()
        if out_n_embd is None:
            out_n_embd = n_embd
        layers = [
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, out_n_embd),
            nn.Dropout(dropout),
        ]
        
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head, block_size, dropout, attention_type='standard', out_n_embd:int|None=None):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        if out_n_embd is None:
            out_n_embd = n_embd
        if attention_type == 'standard':
            self.sa = MultiHeadAttention(n_head, head_size, n_embd, block_size, dropout)
        elif attention_type == 'mamba':
            self.mamba = MambaBlock(n_embd)
        else:
            raise ValueError(f"Unknown attention type: {attention_type}")
        self.ffwd = FeedForward(n_embd, dropout, out_n_embd = out_n_embd)
        if n_embd != out_n_embd:
            self.no_residual = True
        else:
            self.no_residual = False
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.attention_type = attention_type

    def forward(self, x, states=None):
        if self.attention_type == 'mamba':
            mamba_out, new_state = self.mamba(self.ln1(x), h=states)
            x = x + mamba_out
            x = x + self.ffwd(self.ln2(x))
            return x, new_state
        else:
            x = x + self.sa(self.ln1(x))
            if self.no_residual:
                x = self.ffwd(self.ln2(x))  # We can't use residual connection if the dimensions don't match
            else:
                x = x + self.ffwd(self.ln2(x))
            return x, states

class GPT(nn.Module):

    def __init__(self, vocab_size, n_embd:int|list[int], block_size, n_head:int|list[int], n_layer, dropout, device, attention_type:str|list[str]='standard'):
        super().__init__()
        self.device = device
        self.block_size = block_size
        self.attention_type:list[str] = []
        if isinstance(attention_type, list):
            if len(attention_type) != n_layer:
                raise ValueError("attention_type must be a list of length n_layer")
            self.attention_type = attention_type
        else:
            self.attention_type = [attention_type] * n_layer

        if isinstance(n_embd, list):
            if len(n_embd) != n_layer:
                raise ValueError("n_embd must be a list of length n_layer")
            n_embd = n_embd
        else:
            n_embd = [n_embd] * n_layer

        if isinstance(n_head, list):
            if len(n_head) != n_layer:
                raise ValueError("n_head must be a list of length n_layer")
            n_head = n_head
        else:
            n_head = [n_head] * n_layer

        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd[0])
        self.positional_encoding_table = nn.Embedding(block_size, n_embd[0])
        self.blocks = nn.ModuleList([Block(n_embd[i], n_head[i], block_size, dropout, out_n_embd=n_embd[(i+1)%n_layer], attention_type=attention_type[i]) for i in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd[0]) # final layer norm (n_embd[(i+1)%n_layer] is the last element of n_embd, hence again index 0)
        self.lm_head = nn.Linear(n_embd[0], vocab_size)

    def forward(self, idx, targets=None, states=None):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.positional_encoding_table(torch.arange(T, device=self.device)) # (T,C)
        x = tok_emb + pos_emb # (B,T,C)
        
        if states is None:
            states = [None] * len(self.blocks)
        if self.attention_type == 'mamba':
            for i, block in enumerate(self.blocks):
                x, s_new = block(x, states=states[i])
                states[i] = s_new
        else:
            for i, block in enumerate(self.blocks):
                x, _ = block(x, states=states[i])  # States not modified
        
        x = self.ln_f(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        if self.attention_type == 'mamba':
            return logits, loss, states
        else:
            return logits, loss, states

    def generate(self, idx, max_new_tokens, temperature=1.0):
        # Temp > 1.0 = more random
        # Temp < 1.0 = more deterministic
        # idx is (B, T) array of indices in the current context
        states = None
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -self.block_size:]
            # get the predictions
            logits, loss, states = self(idx_cond, states=states)
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
