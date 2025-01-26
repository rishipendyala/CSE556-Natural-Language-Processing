import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import json
from task1.py import WordPieceTokenizer

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
                for idx, word in enumerate(sentence):
                    context = []
                    for neighbor in range(-self.window_size, self.window_size + 1):
                        if neighbor != 0 and 0 <= idx + neighbor < len(sentence):
                            context.append(sentence[idx + neighbor])
                    if context:
                        data.append((context, word))
            return data, list(vocab)
        

class Word2VecModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(Word2VecModel, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear = nn.Linear(embedding_dim, vocab_size)

    def forward(self, context):
        context_embeds = self.embeddings(context)
        avg_context_embeds = context_embeds.mean(dim=1)
        out = self.linear(avg_context_embeds)
        return out


def train(corpus, context_window, embedding_dim, epochs, batch_size, learning_rate):
    dataset = Word2VecDataset(corpus, context_window)
