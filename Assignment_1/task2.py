import torch
from torch.utils.data import Dataset, DataLoader
from task1 import WordPieceTokenizer
import torch.nn as nn
from torch.utils.data import TensorDataset
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import torch.nn.functional as F
import random

class Word2VecDataset(Dataset):
    def __init__(self, tokenizer, corpus, vocab_file, window_size=2):
        self.tokenizer = tokenizer
        self.corpus = corpus
        self.window_size = window_size
        self.vocab_file = vocab_file
        self.PAD_TOKEN = '[PAD]'
        self.UNK_TOKEN = '[UNK]'
        self.data, self.vocab = self.preprocess_data()
        self.word_to_index = {word: idx for idx, word in enumerate(self.vocab)} # Added mapping
        print(f"Vocabulary size: {len(self.vocab)} words")
        print("Dataset initialization complete")

    def preprocess_data(self):
        '''
        Preprocess data:
        If file is provided, load vocabulary from file
        Else, construct vocabulary from corpus
        
        Returns:
        A list of tuples (context, target) and the vocabulary
        '''
        if self.vocab_file:
            with open(self.vocab_file, "r") as vocab_file:
                vocab = [line.strip() for line in vocab_file.readlines()]
        else:
            vocab = self.tokenizer.construct_vocabulary(self.corpus, vocab_size=10000)

        tokenized_sentences = []
        for sentence in self.corpus: # Tokenize each sentence
            tokens = self.tokenizer.tokenize(sentence)
            tokenized_sentences.append(tokens)

        data = []
        for sentence in tokenized_sentences: # For each sentence, get the context and target words
            for count, word in enumerate(sentence):
                context_word = []
                for nearby_words in range(-self.window_size, self.window_size + 1):
                    if nearby_words != 0 and 0 <= count + nearby_words < len(sentence):
                        context_word.append(sentence[count + nearby_words])
                if len(context_word) > 0:
                    data.append((context_word, word))
        return data, vocab

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        '''
        Get item from dataset:
        Convert context words to indices and target word to index

        Returns:
        A tuple (context_indices, target_index)
        '''
        context_words, target_word = self.data[idx]
        unk_idx = self.word_to_index[self.UNK_TOKEN] # Get index of UNK token
        context_indices = [self.word_to_index.get(word, unk_idx) for word in context_words] #convert context words to indices and replce with UNK if not found.
        target_index = self.word_to_index.get(target_word, unk_idx)
        return (context_indices, target_index)


class Word2VecModel(nn.Module):
    '''
    Reference:
    https://www.youtube.com/watch?v=Rqh4SRcZuDA : Video taken as reference to understand the architecture
    https://medium.com/towards-data-science/word2vec-with-pytorch-implementing-original-paper-2cd7040120b0 : Read through to understand the workimng of the architecture and to understand what each function does
    '''
    def __init__(self, vocab_size, embedding_dim, context_size):
        '''
        Initialize the model:
        - Embedding layer: Converts word indices to word vectors
        - Linear layer: Linear transformation to vocab_size dimensions
        '''
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.linear1 = nn.Linear(embedding_dim, vocab_size)
        # self.linear2 = nn.Linear(embedding_dim, vocab_size)

    def forward(self, context_words, target_word):
        '''
        Forward pass:
        Get embeddings of context words, average them and pass through linear layer

        Returns:
        A tensor of shape (batch_size, vocab_size)
        '''
        embeddings = self.embedding(context_words).mean(1).squeeze(1)
        return self.linear1(embeddings)
    
    def get_all_embeddings(self):
        '''
        Get all embeddings:
        Detach embeddings from graph and convert to numpy array

        Returns:
        A numpy array of shape (vocab_size, embedding_dim)
        '''
        return self.embedding.weight.detach().cpu().numpy() # Detach embeddings from graph and convert to numpy array
    
    def get_word_embedding(self, token, dataset):
        '''
        Get embedding for a specific token:
        If token is in dataset, get its index
        Else, get index of UNK token

        Returns:
        A tensor of shape (1, embedding_dim)
        '''
        if token in dataset.word_to_index:
            idx = dataset.word_to_index[token]
        else:
            idx = dataset.word_to_index[dataset.UNK_TOKEN]
        idx_tensor = torch.tensor([idx])
        return self.embedding(idx_tensor) # Get embedding for the token

def train(corpus, embedding_dim, context_size, learning_rate, epochs, batch_size,vocab_file=None):
    '''
    Train the Word2Vec model:
    Initialize tokenizer, dataset and model
    Prepare data and split into train and validation sets
    Train the model and plot losses
    Save the model and dataset

    Returns:
    The trained model, training and validation losses, and the dataset
    '''
    
    tokenizer = WordPieceTokenizer()
    
    dataset = Word2VecDataset(tokenizer, corpus,vocab_file, context_size)
    print(f"Dataset created with {len(dataset)} samples")

    model = Word2VecModel(len(dataset.vocab), embedding_dim, context_size)

    criterion = nn.CrossEntropyLoss(ignore_index=0) # Cross entropy loss with ignore index 0 because index 0 is [PAD]
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) 

    pad_idx = dataset.word_to_index[dataset.PAD_TOKEN] # Get index of PAD token
    
    
    context_data = []
    target_data = []

    #Padd context with PAD token index (0) if context size is not 2*context_size
    count = 0
    for i in range(len(dataset)):
        if len(dataset[i][0]) == 2*context_size:
            context_data.append(dataset[i][0])
            target_data.append(dataset[i][1])
        else:
            # Pad context with PAD token index (0)
            count += 1
            padded_context = dataset[i][0] + [pad_idx] * (2 * context_size - len(dataset[i][0]))
            context_data.append(padded_context)
            target_data.append(dataset[i][1])
    print(f"Padded {count} samples")          

    # Convert context and target data to tensors
    context_data = torch.tensor(context_data) 
    target_data = torch.tensor(target_data)

    # Split data into train and validation sets
    train_context, val_context, train_target, val_target = train_test_split(
        context_data, target_data, test_size=0.2, random_state=42
    )

    # Create TensorDatasets for train and validation sets
    train_dataset = TensorDataset(train_context, train_target)
    val_dataset = TensorDataset(val_context, val_target)

    # Create DataLoaders for train and validation sets
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

def calculate_cosine_similarity(model, token1, token2, dataset):
    '''
    Calculate cosine similarity between two tokens:
    Get embeddings of both tokens, calculate cosine similarity

    Returns:
    A tensor of shape (1)
    '''
    A = model.get_word_embedding(token1, dataset).squeeze(0) # Get embedding of token1 and squeeze to remove extra dimension
    B = model.get_word_embedding(token2, dataset).squeeze(0) # Get embedding of token2 and squeeze to remove extra dimension
    similarity = torch.dot(A, B) / (torch.norm(A) * torch.norm(B)) # Calculate cosine similarity
    return similarity

def find_triplets(model, dataset, num_triplets=2):
    '''
    Find triplets of tokens:
    Randomly select a token, find most and least similar tokens

    Returns:
    A list of triplets
    '''
    triplets = []
    vocab = list(dataset.word_to_index.keys())
    
    # Ignore [PAD] and [UNK] tokens
    vocab = [token for token in vocab if token not in ['[PAD]', '[UNK]']]
    
    for _ in range(num_triplets):
        # Randomly select a token
        anchor_token = random.choice(vocab)
        
        anchor_embedding = model.get_word_embedding(anchor_token, dataset).squeeze(0)
        
        # Store similarities for all other tokens
        similarities = {}
        for token in vocab:
            if token != anchor_token:
                token_embedding = model.get_word_embedding(token, dataset).squeeze(0)
                similarity = torch.dot(anchor_embedding, token_embedding) / (torch.norm(anchor_embedding) * torch.norm(token_embedding))
                similarities[token] = similarity.item()

        # Find the most similar and least similar tokens
        most_similar_token = max(similarities, key=similarities.get)
        least_similar_token = min(similarities, key=similarities.get)

        print(f"Anchor: {anchor_token}, Most Similar: {most_similar_token} (Cosine Similarity: {similarities[most_similar_token]:.4f}), Least Similar: {least_similar_token} (Cosine Similarity: {similarities[least_similar_token]:.4f})")

        # Add the triplet to the list
        triplets.append((anchor_token, most_similar_token, least_similar_token))

    return triplets

if __name__ == "__main__":
    tokenizer = WordPieceTokenizer()
    with open("corpus.txt", "r") as file:
        corpus = file.readlines()
    vocab_file = "vocabulary_66.txt"
    
    # model, train_losses, val_losses, dataset = train(corpus, embedding_dim=100, context_size=2, learning_rate=0.001, epochs=20, batch_size=32, vocab_file=vocab_file)
    # torch.save(dataset, 'dataset.pth')
    # print(model.embedding.weight)

    model = torch.load('word2vec_model1.pth')
    dataset = torch.load('dataset.pth')

    triplets = find_triplets(model, dataset, num_triplets=2)
    print(triplets)
