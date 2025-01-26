import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import json
from task1 import WordPieceTokenizer

class Word2VecDataset(Dataset):
    def __init__(self, data, tokenizer, corpus, window_size = 2):
        self.data = data
        self.window_size = window_size
        self.tokenizer = tokenizer
        self.corpus = corpus

        self.data, self.vocab = self.preprocess_data()

        def preprocess_data(self):
            tokenized_sentences = [self.tokenizer.tokenize(sentence) for sentence in self.corpus]
            vocab = set([token for sentence in tokenized_sentences for token in sentence])
            data = []
            for sentence in tokenized_sentences:
                count = 0
                for word in sentence:
                    pass
                    
        

# class Word2VecModel(nn.Module):


def train(corpus, context_window, embedding_dim, epochs, batch_size, learning_rate):
    dataset = Word2VecDataset(corpus, context_window)
