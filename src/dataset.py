import os
import pickle

import torch
import tiktoken
import tiktoken_ext.openai_public


class CharacterDataset(torch.utils.data.Dataset):
    def __init__(self, filepath, seq_len=8):
        self.seq_len = seq_len
        with open(filepath, 'r', encoding='utf-8') as f:
            self.data = f.read()
        self.chars = sorted(list(set(self.data)))
        self.vocab_dim = len(self.chars)
        self.char_to_idx = {char:idx for idx, char in enumerate(self.chars)}
        self.idx_to_char = {idx:char for idx, char in enumerate(self.chars)}
        self.encoded_data = torch.tensor(self.encode(self.data), dtype=torch.int64)

    def __getitem__(self, i):
        x = self.encoded_data[i:i+self.seq_len]
        y = self.encoded_data[i+1:i+self.seq_len+1]
        return x, y

    def __len__(self):
        return len(self.data) - self.seq_len

    def encode(self, string):
        return [self.char_to_idx[char] for char in string]

    def decode(self, array):
        return ''.join([self.idx_to_char[idx] for idx in array])


class WordDataset(torch.utils.data.Dataset):
    def __init__(self, filepath, seq_len=8):
        self.seq_len = seq_len
        with open(filepath, 'r', encoding='utf-8') as f:
            self.data = f.read()
        self.enc = tiktoken.get_encoding("gpt2")
        self.vocab_dim = self.enc.n_vocab
        self.encoded_data = torch.tensor(self.encode(self.data), dtype=torch.int64)

    def __getitem__(self, i):
        x = self.encoded_data[i:i+self.seq_len]
        y = self.encoded_data[i+1:i+self.seq_len+1]
        return x, y

    def __len__(self):
        return len(self.encoded_data) - self.seq_len

    def encode(self, string):
        return self.enc.encode(string)

    def decode(self, array):
        return self.enc.decode(array)
    
    def save_encodings(self, path_dir):
        tiktoken_gpt2 = tiktoken_ext.openai_public.gpt2()
        with open(os.path.join(path_dir, 'tiktoken_gpt2.pkl'), 'wb') as f:
            pickle.dump(tiktoken_gpt2, f)