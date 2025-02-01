import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from task1 import WordPieceTokenizer
from task2 import Word2VecModel
class NeuralLMDataset(Dataset):
    def __init__(self, corpus, tokenizer, word2vec_model, max_context=5):
        self.tokenizer = tokenizer
        self.word2vec = word2vec_model
        self.max_context = max_context
        self.data = self.preprocess_data(corpus)

    def preprocess_data(self, corpus):
        data = []
        for line in corpus:
            tokens = self.tokenizer.tokenize(line)
            for i in range(len(tokens) - 1):
                context = tokens[max(0, i - self.max_context):i]
                target = tokens[i + 1]
                data.append((context, target))
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        context, target = self.data[idx]
        context_embeddings = torch.tensor([self.word2vec.embedding.weight[self.tokenizer.word_to_index.get(token, 0)] for token in context], dtype=torch.float32)
        target_index = self.tokenizer.word_to_index.get(target, 0)
        return context_embeddings.mean(dim=0), target_index

class NeuralLM1(nn.Module):
    def __init__(self, embedding_dim, vocab_size):
        super(NeuralLM1, self).__init__()
        self.fc = nn.Linear(embedding_dim, vocab_size)

    def forward(self, x):
        return self.fc(x)

class NeuralLM2(nn.Module):
    def __init__(self, embedding_dim, vocab_size):
        super(NeuralLM2, self).__init__()
        self.fc1 = nn.Linear(embedding_dim, embedding_dim * 2)
        self.fc2 = nn.Linear(embedding_dim * 2, vocab_size)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.activation(self.fc1(x))
        return self.fc2(x)

class NeuralLM3(nn.Module):
    def __init__(self, embedding_dim, vocab_size):
        super(NeuralLM3, self).__init__()
        self.fc1 = nn.Linear(embedding_dim, embedding_dim * 3)
        self.fc2 = nn.Linear(embedding_dim * 3, embedding_dim)
        self.fc3 = nn.Linear(embedding_dim, vocab_size)
        self.activation = nn.Tanh()

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        return self.fc3(x)

def train_model(model, dataloader, epochs, criterion, optimizer):
    model.train()
    train_losses = []

    for epoch in range(epochs):
        epoch_loss = 0.0
        for context, target in dataloader:
            optimizer.zero_grad()
            output = model(context)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        train_losses.append(epoch_loss / len(dataloader))
        print(f"Epoch {epoch + 1}, Loss: {train_losses[-1]:.4f}")

    return train_losses

def evaluate_model(model, dataloader):
    model.eval()
    total_loss = 0
    correct_predictions = 0
    total_predictions = 0
    perplexity = 0

    with torch.no_grad():
        for context, target in dataloader:
            output = model(context)
            loss = nn.CrossEntropyLoss()(output, target)
            total_loss += loss.item()

            predictions = torch.argmax(output, dim=1)
            correct_predictions += (predictions == target).sum().item()
            total_predictions += target.size(0)

    accuracy = correct_predictions / total_predictions
    perplexity = torch.exp(torch.tensor(total_loss / len(dataloader)))

    return accuracy, perplexity

def predict_next_tokens(model, tokenizer, word2vec, sentences, top_k=3):
    model.eval()
    predictions = {}

    for sentence in sentences:
        tokens = tokenizer.tokenize(sentence)
        context_embeddings = torch.tensor([word2vec.embedding.weight[tokenizer.word_to_index.get(token, 0)] for token in tokens], dtype=torch.float32).mean(dim=0)
        output = model(context_embeddings.unsqueeze(0))
        top_predictions = torch.topk(output, k=top_k, dim=1).indices.squeeze().tolist()
        predictions[sentence] = [tokenizer.index_to_word[idx] for idx in top_predictions]

    return predictions


with open("corpus.txt", "r") as file:
    corpus = file.readlines()

tokenizer = WordPieceTokenizer()
# Load Word2Vec model

print(model)
word2vec = model

dataset = NeuralLMDataset(corpus, tokenizer, word2vec)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

vocab_size = len(tokenizer.vocab)
embedding_dim = 100

# Initialize models
models = [NeuralLM1(embedding_dim, vocab_size), NeuralLM2(embedding_dim, vocab_size), NeuralLM3(embedding_dim, vocab_size)]
results = {}

for i, model in enumerate(models, 1):
    print(f"Training NeuralLM{i}")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    train_losses = train_model(model, dataloader, epochs=10, criterion=criterion, optimizer=optimizer)
    accuracy, perplexity = evaluate_model(model, dataloader)

    results[f"NeuralLM{i}"] = {
        "train_losses": train_losses,
        "accuracy": accuracy,
        "perplexity": perplexity
    }

    plt.plot(train_losses, label=f"NeuralLM{i}")

plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training Loss Comparison")
plt.legend()
plt.show()

# Predictions
test_sentences = ["The quick brown fox", "The capital of France"]
for i, model in enumerate(models, 1):
    predictions = predict_next_tokens(model, tokenizer, word2vec, test_sentences)
    print(f"Predictions for NeuralLM{i}: {predictions}")