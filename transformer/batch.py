from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from nltk.tokenize import wordpunct_tokenize
from torch import optim
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader, Subset


def tokenize(text):
    """Turn text into discrete tokens.

    Remove tokens that are not words.
    """
    text = text.lower()
    tokens = wordpunct_tokenize(text)

    # Only keep words
    tokens = [token for token in tokens
              if all(char.isalpha() for char in token)]

    return tokens


class EnglishFrenchTranslations(Dataset):
    def __init__(self, path, max_vocab, max_seq_len):
        self.max_vocab = max_vocab
        
        # Extra tokens to add
        self.padding_token = '<PAD>'
        self.start_of_sequence_token = '<SOS>'
        self.end_of_sequence_token = '<EOS>'
        self.unknown_word_token = '<UNK>'
        self.max_seq_len = max_seq_len
        
        # Helper function
        self.flatten = lambda x: [sublst for lst in x for sublst in lst]
        
        # Load the data into a DataFrame
        df = pd.read_csv(path, names=['english', 'french'], sep='\t')
        
        # filter out too long sequences
        df = self.filter_seq_len(df, max_len=self.max_seq_len)
        
        # Tokenize inputs (English) and targets (French)
        self.tokenize_df(df)

        # To reduce computational complexity, replace rare words with <UNK>
        self.replace_rare_tokens(df)
        
        # Prepare variables with mappings of tokens to indices
        self.create_token2idx(df)
        
        # Remove sequences with mostly <UNK>
        df = self.remove_mostly_unk(df)
        
        # Every sequence (input and target) should start with <SOS>
        # and end with <EOS>
        self.add_start_and_end_to_tokens(df)
        
        # Convert tokens to indices
        self.tokens_to_indices(df)
        
    def __getitem__(self, idx):
        """Return example at index idx."""
        return self.indices_pairs[idx][0], self.indices_pairs[idx][1]
    
    def tokenize_df(self, df):
        """Turn inputs and targets into tokens."""
        df['tokens_inputs'] = df.english.apply(tokenize)
        df['tokens_targets'] = df.french.apply(tokenize)
        
    def replace_rare_tokens(self, df):
        """Replace rare tokens with <UNK>."""
        common_tokens_inputs = self.get_most_common_tokens(
            df.tokens_inputs.tolist(),
        )
        common_tokens_targets = self.get_most_common_tokens(
            df.tokens_targets.tolist(),
        )
        
        df.loc[:, 'tokens_inputs'] = df.tokens_inputs.apply(
            lambda tokens: [token if token in common_tokens_inputs 
                            else self.unknown_word_token for token in tokens]
        )
        df.loc[:, 'tokens_targets'] = df.tokens_targets.apply(
            lambda tokens: [token if token in common_tokens_targets
                            else self.unknown_word_token for token in tokens]
        )

    def get_most_common_tokens(self, tokens_series):
        """Return the max_vocab most common tokens."""
        all_tokens = self.flatten(tokens_series)
        # Substract 4 for <PAD>, <SOS>, <EOS>, and <UNK>
        common_tokens = set(list(zip(*Counter(all_tokens).most_common(
            self.max_vocab - 4)))[0])
        return common_tokens

    def remove_mostly_unk(self, df, threshold=0.99):
        """Remove sequences with mostly <UNK>."""
        calculate_ratio = (
            lambda tokens: sum(1 for token in tokens if token != '<UNK>')
            / len(tokens) > threshold
        )
        df = df[df.tokens_inputs.apply(calculate_ratio)]
        df = df[df.tokens_targets.apply(calculate_ratio)]
        return df
    
    def filter_seq_len(self, df, max_len=100):
        mask = (df['english'].str.count(' ') < max_len) & (df['french'].str.count(' ') < max_len)
        return df.loc[mask]
        
    def create_token2idx(self, df):
        """Create variables with mappings from tokens to indices."""
        unique_tokens_inputs = set(self.flatten(df.tokens_inputs))
        unique_tokens_targets = set(self.flatten(df.tokens_targets))
        
        for token in reversed([
            self.padding_token,
            self.start_of_sequence_token,
            self.end_of_sequence_token,
            self.unknown_word_token,
        ]):
            if token in unique_tokens_inputs:
                unique_tokens_inputs.remove(token)
            if token in unique_tokens_targets:
                unique_tokens_targets.remove(token)
                
        unique_tokens_inputs = sorted(list(unique_tokens_inputs))
        unique_tokens_targets = sorted(list(unique_tokens_targets))

        # Add <PAD>, <SOS>, <EOS>, and <UNK> tokens
        for token in reversed([
            self.padding_token,
            self.start_of_sequence_token,
            self.end_of_sequence_token,
            self.unknown_word_token,
        ]):
            
            unique_tokens_inputs = [token] + unique_tokens_inputs
            unique_tokens_targets = [token] + unique_tokens_targets
            
        self.token2idx_inputs = {token: idx for idx, token
                                 in enumerate(unique_tokens_inputs)}
        self.idx2token_inputs = {idx: token for token, idx
                                 in self.token2idx_inputs.items()}
        
        self.token2idx_targets = {token: idx for idx, token
                                  in enumerate(unique_tokens_targets)}
        self.idx2token_targets = {idx: token for token, idx
                                  in self.token2idx_targets.items()}
        
    def add_start_and_end_to_tokens(self, df):
        """Add <SOS> and <EOS> tokens to the end of every input and output."""
        df.loc[:, 'tokens_inputs'] = (
            [self.start_of_sequence_token]
            + df.tokens_inputs
            + [self.end_of_sequence_token]
        )
        df.loc[:, 'tokens_targets'] = (
            [self.start_of_sequence_token]
            + df.tokens_targets
            + [self.end_of_sequence_token]
        )
        
    def tokens_to_indices(self, df):
        """Convert tokens to indices."""
        df['indices_inputs'] = df.tokens_inputs.apply(
            lambda tokens: [self.token2idx_inputs[token] for token in tokens])
        df['indices_targets'] = df.tokens_targets.apply(
            lambda tokens: [self.token2idx_targets[token] for token in tokens])
             
        self.indices_pairs = list(zip(df.indices_inputs, df.indices_targets))
        
    def __len__(self):
        return len(self.indices_pairs)


def collate(batch, src_pad, trg_pad, device):
    inputs = [torch.LongTensor(item[0]) for item in batch]
    targets = [torch.LongTensor(item[1]) for item in batch]
    
    # Pad sequencse so that they are all the same length (within one minibatch)
    padded_inputs = pad_sequence(inputs, padding_value=src_pad, batch_first=True)
    padded_targets = pad_sequence(targets, padding_value=trg_pad, batch_first=True)
    
    # Sort by length for CUDA optimizations
    lengths = torch.LongTensor([len(x) for x in inputs])
    lengths, permutation = lengths.sort(dim=0, descending=True)

    return padded_inputs[permutation].to(device), padded_targets[permutation].to(device), lengths.to(device)


def no_peak_mask(size):
    mask = np.triu(np.ones((1, size, size)), k=1).astype('uint8')
    mask =  Variable(torch.from_numpy(mask) == 0)
    return mask


def create_masks(src, trg, src_pad_idx, trg_pad_idx):
    src_mask = (src != src_pad_idx).unsqueeze(-2)
    if trg is not None:
        trg_mask = (trg != trg_pad_idx).unsqueeze(-2)
        size = trg.size(1) # get seq_len for matrix
        np_mask = no_peak_mask(size).to(trg_mask.device)
        trg_mask = trg_mask & np_mask
    else:
        trg_mask = None
    return src_mask, trg_mask
