import os
import json
import logging
import torch
import tiktoken
import numpy as np

def progress_bar_string(progress: float, bar_length: int=20, start_bracket: str | None="⦃", end_bracket: str | None="⦄") -> str:
    """ progress: float (0%) 0.0 .. 1.0 (100%) """
    if progress < 0.0 or progress > 1.0:
        raise ValueError
    num_blocks = int(bar_length * progress)
    rem = bar_length * progress - num_blocks
    blocks = " ▏▎▍▌▋▊▉█"
    remainder_index = int(rem * len(blocks))
    bar = blocks[-1] * num_blocks
    if remainder_index > 0:
        bar += blocks[remainder_index]
    bar += " " * (bar_length - len(bar))
    if start_bracket is not None:
        bar = start_bracket + bar
    if end_bracket is not None:
        bar += end_bracket
    return bar

def shorten(text:str, length:int, left_align:bool|None=None, ellipsis:str='⋯') -> str:
    if len(text) == length:
        return text
    if len(text) == 0:
        return ' ' * length
    elif len(text) < length:
        if left_align is None:
            return text + ' '*(length - len(text))
        elif left_align is True:
            return text + ' '*(length - len(text))
        else:
            return ' '*(length - len(text)) + text
    else:
        if length == 0:
            return ""
        if length == 1:
            return ellipsis
        if left_align is None:
            l = length // 3
            r = length - l - 1
            return text[:l] + ellipsis + text[-r:]
        elif left_align is True:
            w = length - 1
            return text[:w] + ellipsis
        else:
            w = length - 1
            return text[-w:] + ellipsis

class Tokenizer:
    def __init__(self, text=None):
        self.log = logging.getLogger("Tokenizer")
        # We use GPT-2 encoding
        self.enc = tiktoken.get_encoding("gpt2")
        # Pad vocab size to 50304 (multiple of 64) for efficiency
        self.vocab_size = 50304 
        self.eot_token = self.enc.eot_token
        self.log.info(f"Tiktoken tokenizer initialized. Native vocab: {self.enc.n_vocab}, Padded to: {self.vocab_size}")

    def encode(self, s):
        # Allow special tokens if needed, but for now just standard encode
        return self.enc.encode(s, allowed_special={"<|endoftext|>"})

    def decode(self, tokens, errors="replace"):
        # Filter out tokens that are out of bounds for the BPE vocab
        valid_tokens = [t for t in tokens if t < self.enc.n_vocab]
        try:
            # Try standard decode first
            return self.enc.decode(valid_tokens)
        except Exception:
            # Fallback to byte-level decode with error replacement for invalid sequences
            return self.enc.decode_bytes(valid_tokens).decode("utf-8", errors=errors)

class DataLoader:
    def __init__(self, dataset_dir, tokenizer, block_size, batch_size, device='cpu', cache_dir=None, train_split=0.9):
        self.loader_version = 2
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.batch_size = batch_size
        self.device = device
        self.log = logging.getLogger("DataLoader")
        self.tensor_offsets:list[int] = []
        self.filenames:list[str] = []
        self.tensor_lengths:list[int] = []
        self.tensor_data:torch.Tensor | None = None

        text_length = self._read_length(dataset_dir)
        if cache_dir is not None and os.path.isdir(cache_dir) is False:
            self.log.warning(f"cache_dir {cache_dir} is not a directory")
            cache_dir = None
        cache_file = None
        meta = {}
        if cache_dir is not None:
            cache_meta = os.path.join(cache_dir, "dataset_cache_metadata.json")
            if os.path.exists(cache_meta) is True:
                with open(cache_meta, 'r') as f:
                    try:
                        meta = json.load(f)
                    except:
                        meta = {'dataset_size': -1 }
                        self.log.warning(f"Invalid metadata in {cache_meta}")
                    if 'dataset_size' in meta and meta['dataset_size'] == text_length and 'loader_version' in meta and meta['loader_version'] == self.loader_version:    
                        cache_file = os.path.join(cache_dir, "dataset_cache.pt")
                    else:
                        self.log.warning("Dataset has changed or is invalid, invalidating cache")
                        if 'dataset_size' in meta and meta['dataset_size'] != text_length:
                            self.log.warning(f"Dataset size has changed: {meta['dataset_size']} -> {text_length}")
                        if 'loader_version' in meta and meta['loader_version'] != self.loader_version:
                            self.log.warning(f"Loader version has changed: {meta['loader_version']} -> {self.loader_version}")

        cached_data = False
        if cache_file is not None and os.path.exists(cache_file):
            try:
                self.tensor_data = torch.load(cache_file)
                self.tensor_offsets = meta['tensor_offsets']
                self.filenames = meta['filenames']
                self.tensor_lengths = meta['tensor_lengths']
                cached_data = True
            except:
                pass
        if cached_data is False:
            # Encode the entire text
            self._encode_data(dataset_dir, text_length)
            if cache_dir is not None and os.path.isdir(cache_dir):
                cache_meta = os.path.join(cache_dir, "dataset_cache_metadata.json")
                cache_file = os.path.join(cache_dir, "dataset_cache.pt")
                torch.save(self.tensor_data, cache_file)
                meta = {'dataset_size': text_length,
                        'loader_version': self.loader_version,
                        'filenames': self.filenames,
                        'tensor_lengths': self.tensor_lengths,
                        'tensor_offsets': self.tensor_offsets}
                with open(cache_meta, 'w') as f:
                    json.dump(meta, f)
                self.log.info(f"Encoded data saved to {cache_file}, metadata to {cache_meta}")
        else:
            self.log.info(f"Encoded data loaded from cache file {cache_file}")
        
        # Split into train and validation
        self.train_val_split = int(train_split * len(self.tensor_data))
        
    def _read_length(self, directory):
        files = [f for f in os.listdir(directory) if f.endswith('.txt')]
        text_length = 0
        for filename in sorted(files): # Sort for deterministic order
            path = os.path.join(directory, filename)
            with open(path, 'r', encoding='utf-8') as f:
                try:
                    content = f.read()
                    text_length += len(content)
                except Exception as e:
                    self.log.warning(f"Error reading {path}: {e}")
        self.log.info(f"Read {text_length} characters from {len(files)} files")
        return text_length

    def _encode_data(self, directory, max_length, progress=True):
        files = [f for f in os.listdir(directory) if f.endswith('.txt')]
        self.log.info(f"Found {len(files)} text files in {directory}")
        current_offset = 0
        text_length = 0
        self.log.info("Encoding data")
        all_tokens = []
        for filename in sorted(files): # Sort for deterministic order
            path = os.path.join(directory, filename)
            with open(path, 'r', encoding='utf-8') as f:
                if progress is True:
                    print(f"\rProcessing {shorten(filename, 40)} ({progress_bar_string(text_length / max_length)})", end="")
                try:
                    content = f.read()
                except Exception as e:
                    self.log.warning(f"Error reading {path}: {e}")
                    continue
                tokens = self.tokenizer.encode(content)
                decoded = self.tokenizer.decode(tokens)
                if decoded != content:
                    print()
                    print(f"Tokenizer failed to round-trip: {path}, ignoring document.")
                    print()
                    continue
                self.tensor_offsets.append(current_offset)
                self.filenames.append(filename)
                self.tensor_lengths.append(len(tokens))
                text_length += len(content)
                current_offset += len(tokens)
                all_tokens.append(torch.tensor(tokens, dtype=torch.long))

        if all_tokens:
            self.tensor_data = torch.cat(all_tokens, dim=0)
        else:
            self.tensor_data = torch.empty(0, dtype=torch.long)
        print()
        self.log.info(f"Read {text_length} tokens from {len(files)} files")
        # Keep data on CPU to save VRAM; we'll move batches to device on-demand
        self.tensor_data = self.tensor_data.to('cpu')
        return text_length

    def get_batch(self, split):
        offset = self.train_val_split if split == 'val' else 0
        length = self.train_val_split if split == 'train' else len(self.tensor_data) - self.train_val_split
        ix = torch.randint(length - self.block_size - 1, (self.batch_size,))
        x = torch.stack([self.tensor_data[i+offset:i+offset+self.block_size] for i in ix])
        y = torch.stack([self.tensor_data[i+1+offset:i+1+offset+self.block_size] for i in ix])
        # Move to device on-demand
        x, y = x.to(self.device), y.to(self.device)
        return x, y

    def get_random_book_start_and_length(self, split=None):
        if split is None:
            # Select any book, probability is proportional to length
            book_idx = torch.multinomial(torch.tensor(self.tensor_lengths, dtype=torch.float), 1).item()
            return self.tensor_offsets[book_idx], self.tensor_lengths[book_idx]
        
        # Filter books based on their starting offset relative to train_val_split
        indices = []
        lengths = []
        for i, offset in enumerate(self.tensor_offsets):
            if split == 'train':
                if offset < self.train_val_split:
                    indices.append(i)
                    lengths.append(self.tensor_lengths[i])
            else: # val
                if offset >= self.train_val_split:
                    indices.append(i)
                    lengths.append(self.tensor_lengths[i])
        
        if not indices:
            # Fallback if no books in split (should not happen if split is reasonable)
            return self.get_random_book_start_and_length(None)
            
        rel_idx = torch.multinomial(torch.tensor(lengths, dtype=torch.float), 1).item()
        book_idx = indices[rel_idx]
        return self.tensor_offsets[book_idx], self.tensor_lengths[book_idx]

    def get_book_sequential_batch(self, state, split=None):
        """
        returns a tuple of (batch_x, batch_y, state)
        state is a list of tuples of (offset, length, pos, new_book)
        new_book is a boolean indicating whether the book was just started
        """
        if state is None:
            state = []
            for i in range(self.batch_size):
                offset, length = self.get_random_book_start_and_length(split=split)
                state.append((offset, length, 0, True))
        batch_x = torch.zeros((self.batch_size, self.block_size), dtype=torch.long, device=self.device)
        batch_y = torch.zeros((self.batch_size, self.block_size), dtype=torch.long, device=self.device)
        for i in range(self.batch_size):
            offset, length, pos, _ = state[i]
            this_is_new_book = False
            if pos + self.block_size + 1 >= length:
                offset, length = self.get_random_book_start_and_length(split=split)
                pos = 0
                this_is_new_book = True
            batch_x[i] = self.tensor_data[offset + pos:offset + pos + self.block_size]
            batch_y[i] = self.tensor_data[offset + pos + 1:offset + pos + 1 + self.block_size]
            state[i] = (offset, length, pos + self.block_size, this_is_new_book)
        
        # Move the batch to the correct device
        batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
        return batch_x, batch_y, state

 
if __name__ == "__main__":
    # fast test
    dataset_dir = os.path.join(os.path.dirname(__file__), 'dataset')
    tokenizer = Tokenizer()
    batch_size = 5
    loader = DataLoader(dataset_dir, tokenizer, block_size=8, batch_size=batch_size, cache_dir=dataset_dir)
    state = None
    res = ["" for _ in range(batch_size)]
    for i in range(10):
        xb, yb, state = loader.get_book_sequential_batch(state)
        if i==0:
            print("Batch shape:", xb.shape)
            print("Input:", xb[0].tolist())
            print("Target:", yb[0].tolist())
            print("Decoded input sample: ", end="")
        for b in range(batch_size):
            res[b] += tokenizer.decode(xb[b].tolist())
    for b in range(batch_size):
        print(f"Batch {b}:------------------------------")
        print(res[b])
