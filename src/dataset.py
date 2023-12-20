import os
import pickle

import torch
import tiktoken


class CharacterDataset(torch.utils.data.Dataset):
    def __init__(self, text: str, seq_len: int=8):
        self.text = text
        self.seq_len = seq_len
        self.chars = sorted(list(set(self.text)))
        self.vocab_dim = len(self.chars)
        self.char_to_idx = {char:idx for idx, char in enumerate(self.chars)}
        self.idx_to_char = {idx:char for idx, char in enumerate(self.chars)}
        self.encoded_text = torch.tensor(self.encode(self.text), dtype=torch.int64)

    def __getitem__(self, i):
        x = self.encoded_text[i:i+self.seq_len]
        y = self.encoded_text[i+1:i+self.seq_len+1]
        return x, y

    def __len__(self):
        return len(self.text) - self.seq_len

    def encode(self, string):
        return [self.char_to_idx[char] for char in string]

    def decode(self, array):
        return ''.join([self.idx_to_char[idx] for idx in array])


class WordDataset(torch.utils.data.Dataset):
    def __init__(self, text: str, seq_len: int=8, tiktoken_config=None):
        self.text = text
        self.seq_len = seq_len
        if tiktoken_config is not None:
            self.enc = tiktoken.core.Encoding(
                tiktoken_config['name'],
                explicit_n_vocab=tiktoken_config['explicit_n_vocab'],
                pat_str=tiktoken_config['pat_str'],
                mergeable_ranks=tiktoken_config['mergeable_ranks'],
                special_tokens=tiktoken_config['special_tokens'],
            )
        else:
            self.enc = tiktoken.get_encoding("gpt2")
        self.vocab_dim = self.enc.n_vocab
        self.encoded_text = torch.tensor(self.encode(self.text), dtype=torch.int64)

    def __getitem__(self, i):
        x = self.encoded_text[i:i+self.seq_len]
        y = self.encoded_text[i+1:i+self.seq_len+1]
        return x, y

    def __len__(self):
        return len(self.encoded_text) - self.seq_len

    def encode(self, string):
        return self.enc.encode(string)

    def decode(self, array):
        return self.enc.decode(array)
