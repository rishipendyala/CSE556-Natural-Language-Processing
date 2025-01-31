import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from collections import Counter
import numpy as np
from task1 import WordPieceTokenizer
from task2 import Word2VecDataset, Word2VecModel

class NeuralLMDataset(Dataset):
    def __init__(self, tokenizer, corpus, word2vec_model, context_size=2):
        """
        Initialize the dataset with tokenizer and pretrained word2vec model
        """
        self.tokenizer = tokenizer
        self.corpus = corpus
        self.context_size = context_size
        self.word2vec_model = word2vec_model
        self.train_data, self.val_data = self.preprocess_data()
        
    def preprocess_data(self):
        """
        Preprocess the corpus for next-word prediction task
        """
        tokenized_sentences = []
        for sentence in self.corpus:
            tokens = self.tokenizer.tokenize(sentence)
            if len(tokens) > self.context_size + 1:  # Need at least context + 1 target
                tokenized_sentences.append(tokens)

        # Split into train/val (80/20)
        split_idx = int(0.8 * len(tokenized_sentences))
        train_sentences = tokenized_sentences[:split_idx]
        val_sentences = tokenized_sentences[split_idx:]

        # Process train and val data
        train_data = self._create_sequences(train_sentences)
        val_data = self._create_sequences(val_sentences)

        return train_data, val_data

    def _create_sequences(self, sentences):
        """
        Create context-target pairs from sentences
        """
        data = []
        for sentence in sentences:
            for i in range(len(sentence) - self.context_size):
                context = sentence[i:i + self.context_size]
                target = sentence[i + self.context_size]
                if all(token in self.tokenizer.word_to_index for token in context + [target]):
                    context_indices = [self.tokenizer.word_to_index[token] for token in context]
                    target_index = self.tokenizer.word_to_index[target]
                    data.append((context_indices, target_index))
        return data

    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, idx):
        context_indices, target_index = self.train_data[idx]
        return torch.tensor(context_indices), torch.tensor(target_index)

    def get_validation_data(self):
        """
        Return validation data as a separate dataset
        """
        val_contexts = [item[0] for item in self.val_data]
        val_targets = [item[1] for item in self.val_data]
        return torch.tensor(val_contexts), torch.tensor(val_targets)

class NeuralLM1(nn.Module):
    """
    Basic Neural LM with single hidden layer and ReLU activation
    """
    def __init__(self, vocab_size, embedding_dim, context_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.fc1 = nn.Linear(context_size * embedding_dim, 128)
        self.fc2 = nn.Linear(128, vocab_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.embedding(x).view(x.size(0), -1)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

class NeuralLM2(nn.Module):
    """
    Enhanced Neural LM with deeper architecture and Tanh activation
    """
    def __init__(self, vocab_size, embedding_dim, context_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.fc1 = nn.Linear(context_size * embedding_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, vocab_size)
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(0.3)
        self.layer_norm = nn.LayerNorm(256)

    def forward(self, x):
        x = self.embedding(x).view(x.size(0), -1)
        x = self.dropout(self.layer_norm(self.tanh(self.fc1(x))))
        x = self.dropout(self.tanh(self.fc2(x)))
        x = self.fc3(x)
        return x

class NeuralLM3(nn.Module):
    """
    Advanced Neural LM with residual connections and LeakyReLU
    """
    def __init__(self, vocab_size, embedding_dim, context_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.fc1 = nn.Linear(context_size * embedding_dim, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, vocab_size)
        self.leaky_relu = nn.LeakyReLU(0.1)
        self.dropout = nn.Dropout(0.4)
        self.layer_norm1 = nn.LayerNorm(512)
        self.layer_norm2 = nn.LayerNorm(512)

    def forward(self, x):
        x = self.embedding(x).view(x.size(0), -1)
        
        # First block with residual connection
        identity = self.fc1(x)
        x = self.dropout(self.layer_norm1(self.leaky_relu(self.fc1(x))))
        x = self.dropout(self.layer_norm2(self.leaky_relu(self.fc2(x))))
        x = x + identity  # Residual connection
        
        x = self.dropout(self.leaky_relu(self.fc3(x)))
        x = self.fc4(x)
        return x

def compute_accuracy(model, data_loader):
    """
    Compute prediction accuracy
    """
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for context, target in data_loader:
            output = model(context)
            _, predicted = torch.max(output, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    return correct / total

def compute_perplexity(model, data_loader, criterion):
    """
    Compute model perplexity
    """
    model.eval()
    total_loss = 0
    total_tokens = 0
    with torch.no_grad():
        for context, target in data_loader:
            output = model(context)
            loss = criterion(output, target)
            total_loss += loss.item() * target.size(0)
            total_tokens += target.size(0)
    avg_loss = total_loss / total_tokens
    return torch.exp(torch.tensor(avg_loss))

def train(model, train_loader, val_loader, criterion, optimizer, epochs):
    """
    Train the model and return training history
    """
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        total_train_loss = 0
        for context, target in train_loader:
            optimizer.zero_grad()
            output = model(context)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
        
        # Validation phase
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for context, target in val_loader:
                output = model(context)
                loss = criterion(output, target)
                total_val_loss += loss.item()
        
        # Record losses
        avg_train_loss = total_train_loss / len(train_loader)
        avg_val_loss = total_val_loss / len(val_loader)
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        
        print(f'Epoch {epoch + 1}/{epochs}')
        print(f'Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')
        
    return train_losses, val_losses

def predict_next_tokens(model, tokenizer, sentence, num_tokens=3):
    """
    Predict next tokens for a given sentence
    """
    model.eval()
    tokens = tokenizer.tokenize(sentence)
    predictions = []
    
    with torch.no_grad():
        context = tokens[-2:]  # Use last 2 tokens as context
        for _ in range(num_tokens):
            if len(context) < 2:
                break
                
            context_indices = [tokenizer.word_to_index.get(token, 0) for token in context]
            context_tensor = torch.tensor(context_indices).unsqueeze(0)
            
            output = model(context_tensor)
            _, predicted_idx = torch.max(output, 1)
            predicted_token = tokenizer.index_to_word.get(predicted_idx.item(), '<UNK>')
            
            predictions.append(predicted_token)
            context = context[1:] + [predicted_token]
    
    return predictions

def plot_losses(train_losses, val_losses, model_name):
    """
    Plot training and validation losses
    """
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title(f'{model_name} - Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f'{model_name}_loss.png')
    plt.close()

if __name__ == "__main__":
    # Load the data and models
    with open("corpus.txt", "r") as file:
        corpus = file.readlines()
        
    # Initialize tokenizer
    tokenizer = WordPieceTokenizer()
    tokenizer.construct_vocabulary(corpus, vocab_size=6000)
    
    # Load pretrained word2vec model
    word2vec_model = torch.load('word2vec_model.pth')
    
    # Create dataset and dataloaders
    dataset = NeuralLMDataset(tokenizer, corpus, word2vec_model)
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    val_data = dataset.get_validation_data()
    val_dataset = torch.utils.data.TensorDataset(*val_data)
    val_loader = DataLoader(val_dataset, batch_size=32)
    
    # Training parameters
    vocab_size = len(tokenizer.vocab)
    embedding_dim = 100
    context_size = 2
    epochs = 20
    learning_rate = 0.001
    
    # Initialize models
    models = [
        ('NeuralLM1', NeuralLM1(vocab_size, embedding_dim, context_size)),
        ('NeuralLM2', NeuralLM2(vocab_size, embedding_dim, context_size)),
        ('NeuralLM3', NeuralLM3(vocab_size, embedding_dim, context_size))
    ]
    
    # Train and evaluate each model
    for model_name, model in models:
        print(f"\nTraining {model_name}")
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        # Train the model
        train_losses, val_losses = train(model, train_loader, val_loader, criterion, optimizer, epochs)
        
        # Plot losses
        plot_losses(train_losses, val_losses, model_name)
        
        # Compute metrics
        train_accuracy = compute_accuracy(model, train_loader)
        val_accuracy = compute_accuracy(model, val_loader)
        train_perplexity = compute_perplexity(model, train_loader, criterion)
        val_perplexity = compute_perplexity(model, val_loader, criterion)
        
        print(f"\n{model_name} Results:")
        print(f"Train Accuracy: {train_accuracy:.4f}")
        print(f"Validation Accuracy: {val_accuracy:.4f}")
        print(f"Train Perplexity: {train_perplexity:.4f}")
        print(f"Validation Perplexity: {val_perplexity:.4f}")
        
        # Save model
        torch.save(model.state_dict(), f'{model_name.lower()}_model.pth')
    
    # Test prediction on test.txt
    with open("sample_test.txt", "r") as file:
        test_sentences = file.readlines()
    
    print("\nPredicting next tokens for test sentences:")
    for sentence in test_sentences:
        sentence = sentence.strip()
        for model_name, model in models:
            predictions = predict_next_tokens(model, tokenizer, sentence)
            print(f"\n{model_name} predictions for: {sentence}")
            print(f"Next three tokens: {predictions}")