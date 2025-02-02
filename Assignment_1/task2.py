import torch
from torch.utils.data import Dataset, DataLoader
from task1 import WordPieceTokenizer
import torch.nn as nn
from torch.utils.data import TensorDataset
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import torch.nn.functional as F

class Word2VecDataset(Dataset):
    def __init__(self, tokenizer, corpus, vocab_file, window_size = 2):
        self.tokenizer = tokenizer
        self.corpus = corpus
        self.window_size = window_size
        self.vocab_file = vocab_file
        self.PAD_TOKEN = '[PAD]'  # Define padding token
        self.data, self.vocab = self.preprocess_data()
        # Ensure PAD_TOKEN is in vocabulary and has index 0
        if self.PAD_TOKEN not in self.vocab:
            self.vocab = [self.PAD_TOKEN] + self.vocab
        self.word_to_index = {word: idx for idx, word in enumerate(self.vocab)}
        print(f"Vocabulary size: {len(self.vocab)} words")
        print("Dataset initialization complete")

    def preprocess_data(self):
        if self.vocab_file:
            with open(self.vocab_file, "r") as vocab_file:
                vocab = [line.strip() for line in vocab_file.readlines()]
        else:
            vocab = self.tokenizer.construct_vocabulary(self.corpus, vocab_size=10000)
        
        tokenized_sentences = []
        for sentence in self.corpus:
            tokens = self.tokenizer.tokenize(sentence)
            for token in tokens:
                if token not in vocab:
                    print(f"Token {token} not in vocab")
            tokenized_sentences.append(tokens)

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
        return data, vocab
    
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
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.linear = nn.Linear(embedding_dim, vocab_size)
    
    def forward(self, context_words, target_word):
        embeddings = self.embedding(context_words).mean(1)
        return self.linear(embeddings)

def find_triplets(model, word_to_index, top_k=2):
    # Extract embeddings
    embeddings = model.embedding.weight.data
    
    # Compute cosine similarity matrix
    cosine_sim = F.cosine_similarity(embeddings.unsqueeze(1), embeddings.unsqueeze(0), dim=2)
    
    # Exclude self-similarity (diagonal elements)
    cosine_sim.fill_diagonal_(-1)
    
    # Find top-k similar pairs
    top_pairs = []
    for i in range(len(cosine_sim)):
        top_indices = torch.topk(cosine_sim[i], top_k + 1).indices
        for j in top_indices:
            if i != j:
                top_pairs.append((i, j, cosine_sim[i][j].item()))
    
    # Sort pairs by similarity
    top_pairs.sort(key=lambda x: x[2], reverse=True)
    
    # Select two triplets
    triplets = []
    used_indices = set()
    
    for pair in top_pairs:
        if pair[0] not in used_indices and pair[1] not in used_indices:
            # Find a dissimilar word
            dissimilar_word_idx = torch.argmin(cosine_sim[pair[0]]).item()
            if dissimilar_word_idx not in used_indices:
                triplets.append((pair[0], pair[1], dissimilar_word_idx))
                used_indices.update({pair[0], pair[1], dissimilar_word_idx})
        
        if len(triplets) == 2:
            break
    
    # Map indices to words
    index_to_word = {idx: word for word, idx in word_to_index.items()}
    triplet_words = []
    for triplet in triplets:
        triplet_words.append((index_to_word[triplet[0]], index_to_word[triplet[1]], index_to_word[triplet[2]]))
    
    return triplet_words

def train(corpus, embedding_dim, context_size, learning_rate, epochs, batch_size,vocab_file=None):
    
    tokenizer = WordPieceTokenizer()
    
    dataset = Word2VecDataset(tokenizer, corpus,vocab_file, 2)
    print(f"Dataset created with {len(dataset)} samples")

    model = Word2VecModel(len(dataset.vocab), embedding_dim, context_size)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    pad_idx = dataset.word_to_index[dataset.PAD_TOKEN]

    # Prepare data
    context_data = []
    target_data = []
    count = 0
    for i in range(len(dataset)):
        if len(dataset[i][0]) == 2*context_size:
            context_data.append(dataset[i][0])
            target_data.append(dataset[i][1])
        else:
            # Pad context with zeros until it reaches size 4
            count += 1
            padded_context = dataset[i][0] + [pad_idx] * (2 * context_size - len(dataset[i][0]))
            context_data.append(padded_context)
            target_data.append(dataset[i][1])
    print(f"Padded {count} samples")          

    context_data = torch.tensor(context_data)
    target_data = torch.tensor(target_data)

    # Split data into train and validation sets
    train_context, val_context, train_target, val_target = train_test_split(
        context_data, target_data, test_size=0.2, random_state=42
    )

    train_dataset = TensorDataset(train_context, train_target)
    val_dataset = TensorDataset(val_context, val_target)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    train_losses = []
    val_losses = []

    # Training loop
    for epoch in range(epochs):
        model.train()
        epoch_train_losses = []
        
        # Training phase
        for context, target in train_loader:
            optimizer.zero_grad()
            output = model(context, target)
            loss = criterion(output, target)
            # optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_train_losses.append(loss.item())
        
        # Validation phase
        model.eval()
        epoch_val_losses = []
        with torch.no_grad():
            for context, target in val_loader:
                output = model(context, target)
                val_loss = criterion(output, target)
                epoch_val_losses.append(val_loss.item())
        
        # Calculate average losses
        avg_train_loss = np.mean(epoch_train_losses)
        avg_val_loss = np.mean(epoch_val_losses)
        
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        
        print(f'Epoch {epoch + 1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')

    # Plot training and validation losses
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, epochs + 1), train_losses, label='Training Loss')
    plt.plot(range(1, epochs + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Over Time')
    plt.legend()
    plt.grid(True)
    plt.savefig('training_loss1.png')
    plt.close()

    torch.save(model, 'word2vec_model1.pth')

    return model, train_losses, val_losses, dataset

if __name__ == "__main__":

    tokenizer = WordPieceTokenizer()
    with open("corpus.txt", "r") as file:
        corpus = file.readlines()
    vocab_file = "vocabulary_66.txt"

    model, train_losses, val_losses, dataset = train(corpus, embedding_dim=100, context_size=2, learning_rate=0.001, epochs=30, batch_size=32, vocab_file=vocab_file)
    print(model.embedding.weight)

    triplets = find_triplets(model, dataset.word_to_index)
    print("Triplets (two similar words and one dissimilar word):")
    for triplet in triplets:
        print(triplet)