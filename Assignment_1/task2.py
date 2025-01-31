import torch
from torch.utils.data import Dataset, DataLoader
from task1 import WordPieceTokenizer
import torch.nn as nn
from torch.utils.data import TensorDataset

class Word2VecDataset(Dataset):
    def __init__(self, tokenizer, corpus, window_size = 2):
        self.tokenizer = tokenizer
        self.corpus = corpus
        self.window_size = window_size
        self.data, self.vocab = self.preprocess_data()
        self.word_to_index = {word: idx for idx, word in enumerate(self.vocab)}
        print(f"Vocabulary size: {len(self.vocab)} words")
        print("Dataset initialization complete")

    def preprocess_data(self):
        tokenized_sentences = []
        for sentence in self.corpus:
            tokens = self.tokenizer.tokenize(sentence)
            tokenized_sentences.append(tokens)

        vocab = set()
        for sentence in tokenized_sentences:
            for token in sentence:
                vocab.add(token)
        data = []
        for sentence in tokenized_sentences:
            count = 0
            for word in sentence:
                context_word = []
                for nearby_words in range(-self.window_size, self.window_size + 1):
                    if nearby_words != 0 and 0 <= count + nearby_words < len(sentence):
                        context_word.append(sentence[count + nearby_words])
                if len(context_word) > 0:
                    data.append((context_word, word))
                count += 1
        return data, list(vocab)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        context_words, target_word = self.data[idx]
        context_indices = [self.word_to_index[word] for word in context_words]
        target_index = self.word_to_index[target_word]
        return (context_indices, target_index)

        # return (context_words, target_word)

class Word2VecModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, context_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.linear = nn.Linear(embedding_dim, vocab_size)
    
    def forward(self, context_words, target_word):
        embeddings = self.embedding(context_words).mean(1)
        return self.linear(embeddings)

def train(corpus, embedding_dim, context_size, learning_rate, epochs, batch_size):
    
    tokenizer = WordPieceTokenizer()
    tokenizer.construct_vocabulary(corpus, vocab_size=1000)
    
    dataset = Word2VecDataset(tokenizer, corpus)
    print(f"Dataset created with {len(dataset)} samples")

    # for i in range(len(dataset)):
    #     print(dataset[i])

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    print(f"DataLoader created with {len(dataloader)} batches")

    model = Word2VecModel(len(dataset.vocab), embedding_dim, context_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    context_data = []
    target_data = []
    for i in range(len(dataset)):
        if len(dataset[i][0]) == 2*context_size:
            context_data.append(dataset[i][0])
            target_data.append(dataset[i][1])
        else:
            # Pad context with zeros until it reaches size 4
            padded_context = dataset[i][0] + [0] * (2*context_size - len(dataset[i][0]))
            context_data.append(padded_context)
            target_data.append(dataset[i][1])
            print(f"Padded sample {i} with zeros")
            # continue

    context_data = torch.tensor(context_data)
    target_data = torch.tensor(target_data)

    dataset = TensorDataset(context_data, target_data)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        for context, target in dataloader:
            output = model(context, target)
            loss = criterion(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch + 1}/{epochs}, Current loss: {loss.item():.4f}')

    torch.save(model.state_dict(), 'word2vec_model.pth')
    return model


if __name__ == "__main__":


    tokenizer = WordPieceTokenizer()
    with open("corpus.txt", "r") as file:
        corpus = file.readlines()

    model = train(corpus, embedding_dim=100, context_size=2, learning_rate=0.001, epochs=100, batch_size=32)
    print(model.embedding.weight)
