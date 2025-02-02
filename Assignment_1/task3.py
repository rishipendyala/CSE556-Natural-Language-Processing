import os
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from task1 import WordPieceTokenizer
from task2 import Word2VecDataset, Word2VecModel

class NeuralLMDataset(Dataset):
    def __init__(self, tokenizer, corpus, word2vec_model, context_size):
        self.tokenizer = tokenizer
        self.corpus = corpus
        self.context_size = context_size
        self.word2vec_model = word2vec_model
        self.train_data, self.val_data = self.preprocess_data()
        self.embedding_dim = self.word2vec_model.embedding.weight.shape[1]
        
    def preprocess_data(self):
        '''
        Preprocess the data by tokenizing the sentences, splitting the data into training 
        and validation sets, and creating sequences of context words and target words.
        '''
        tokenized_sentences = []
        for sentence in self.corpus:
            
            #using the tokenizer to tokenize the sentence
            tokens = self.tokenizer.tokenize(sentence)
            if len(tokens) > self.context_size + 1:
                tokenized_sentences.append(tokens)
                
        #splitting the data into training and validation sets
        split_idx = int(0.8 * len(tokenized_sentences))
        train_sentences = tokenized_sentences[:split_idx]
        val_sentences = tokenized_sentences[split_idx:]
        #Create sequences of context words and target words and using the tokenizer to convert words to indices
        train_data = self.sequence_creation(train_sentences)
        val_data = self.sequence_creation(val_sentences)
        print(f"Train data size: {len(train_data)}")
        print(f"Validation data size: {len(val_data)}")
        print(f"Validation data example: {val_data[:10]}")
        return train_data, val_data

    def sequence_creation(self, sentences):
        #Create sequences of context words and target words
        data = []
        for sentence in sentences:
            for i in range(len(sentence) - self.context_size):
                context = sentence[i:i + self.context_size]
                target = sentence[i + self.context_size]
                if all(token in self.tokenizer.word_to_index for token in context + [target]):
                    #Convert words to indices
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
        '''
        Get the validation data as tensors
        '''
        val_contexts = [item[0] for item in self.val_data]
        val_targets = [item[1] for item in self.val_data]
        return torch.tensor(val_contexts), torch.tensor(val_targets) #Convert data to tensors

class NeuralLM_1(nn.Module):
    # ARCHITECTURE IN DETAIL
    # 1. Embedding layer: Converts word indices to word vectors
    # 2. Fully connected layer (fc1): Linear transformation to 128 dimensions
    # 3. ReLU activation function: Non-linear activation function
    # 4. Dropout layer: Regularization technique to prevent overfitting
    # 5. Fully connected layer (fc2): Linear transformation to vocab_size dimensions
    
    def __init__(self, word2vec_model, vocab_size, context_size, embedding_dim):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim) #Embedding layer - converts word indices to word vectors
        self.embedding.weight.data.copy_(word2vec_model.embedding.weight.data) #Initialize with Word2Vec embeddings from task 2
        self.embedding.weight.requires_grad = True #Fine-tune embeddings during training
        
        #neural network architecture
        self.fc1 = nn.Linear(context_size * embedding_dim, 128)
        self.fc2 = nn.Linear(128, vocab_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2) #0.2 is the dropout rate (probability of an element to be zeroed)

    def forward(self, x):
        #Forward pass 
        x = self.embedding(x).view(x.size(0), -1)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

class NeuralLM__2(nn.Module):
    # ARCHITECTURE IN DETAIL
    # 1. Embedding layer: Converts word indices to word vectors
    # 2. Fully connected layer (fc1): Linear transformation to 256 dimensions
    # 3. Layer normalization: Normalize the output of fc1
    # 5. Dropout layer: Regularization technique to prevent overfitting
    # 6. Fully connected layer (fc2): Linear transformation to 128 dimensions
    # 7. Tanh activation function: Non-linear activation function
    # 9. Fully connected layer (fc3): Linear transformation to vocab_size dimensions
    
    def __init__(self, word2vec_model, vocab_size, context_size, embedding_dim):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim) #Embedding layer - converts word indices to word vectors
        self.embedding.weight.data.copy_(word2vec_model.embedding.weight.data) #Initialize with Word2Vec embeddings from task 2
        self.embedding.weight.requires_grad = True #Fine-tune embeddings during training
        
        #neural network architecture
        self.fc1 = nn.Linear(context_size * embedding_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, vocab_size)
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(0.3)
        self.layer_norm = nn.LayerNorm(256)

    def forward(self, x):
        #Forward pass
        x = self.embedding(x).view(x.size(0), -1)
        x = self.dropout(self.layer_norm(self.tanh(self.fc1(x))))
        x = self.dropout(self.tanh(self.fc2(x)))
        x = self.fc3(x)
        return x

class NeuralLM3(nn.Module):
    # ARCHITECTURE IN DETAIL
    # 1. Embedding layer: Converts word indices to word vectors
    # 2. Fully connected layer (fc1): Linear transformation to 512 dimensions
    # 3. Layer normalization: Normalize the output of fc1
    # 5. Dropout layer: Regularization technique to prevent overfitting
    # 6. Fully connected layer (fc2): Linear transformation to 512 dimensions
    # 7. Layer normalization: Normalize the output of fc2
    # 9. Residual connection: Add the output of fc1 to the output of fc2
    # 10. Fully connected layer (fc3): Linear transformation to 256 dimensions
    # 12. Fully connected layer (fc4): Linear transformation to vocab_size dimensions
    
    def __init__(self, word2vec_model, vocab_size, context_size, embedding_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding.weight.data.copy_(word2vec_model.embedding.weight.data)
        self.embedding.weight.requires_grad = True
        self.fc1 = nn.Linear(context_size * embedding_dim, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, vocab_size)
        self.leaky_relu = nn.LeakyReLU(0.1)
        self.dropout = nn.Dropout(0.4)
        self.layer_norm1 = nn.LayerNorm(512)
        self.layer_norm2 = nn.LayerNorm(512)

    def forward(self, x):
        '''
        Forward pass of the model with residual connections
        '''
        x = self.embedding(x).view(x.size(0), -1)
        identity = self.fc1(x)
        x = self.dropout(self.layer_norm1(self.leaky_relu(self.fc1(x))))
        x = self.dropout(self.layer_norm2(self.leaky_relu(self.fc2(x))))
        x = x + identity  #Residual connection
        
        x = self.dropout(self.leaky_relu(self.fc3(x)))
        x = self.fc4(x)
        return x

def compute_accuracy(model, data_loader):
    '''
    Accuracy is calculated as the number of correct predictions divided by the total number of samples
    '''
    model.eval() #Set model to evaluation mode
    correct = 0
    total = 0
    with torch.no_grad():
        for context, target in data_loader:
            output = model(context)
            _, predicted = torch.max(output, 1) #Get the predicted class
            total += target.size(0) #Get the number of samples in the batch
            correct += (predicted == target).sum().item() #Get the number of correct predictions
            acc = correct / total #Compute the accuracy
    return acc

def compute_perplexity(model, data_loader, criterion):
    '''
    Perplexity is calculated as the exponential of the loss
    '''
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

def train(model, train_loader, val_loader, criterion, optimizer, epochs, early_stopping=False):
    '''
    Train the model using the training data and validate it using the validation data
    Early stopping is used to prevent overfitting
    '''
    train_losses = []
    val_losses = []
    
    best_val_loss = float('inf')
    patience = 3
    patience_counter = 0

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

        # Record losses and perplexities
        avg_train_loss = total_train_loss / len(train_loader)
        avg_val_loss = total_val_loss / len(val_loader)
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)

        print(f'Epoch {epoch + 1}/{epochs}')
        print(f'Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')

        if early_stopping:
            # Check for early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print("Early stopping triggered")
                    break

    return train_losses, val_losses

def predict_next_tokens(model, tokenizer, sentence, num_tokens=3):
    '''
    predict the next tokens in a sentence using the model
    '''
    model.eval()
    tokens = tokenizer.tokenize(sentence)
    predictions = []
    
    with torch.no_grad():
        context = tokens[-3:]  #2 is the context window size
        for _ in range(num_tokens):
            if len(context) < 3:
                break
            
            #Convert context words to indices
            context_indices = [tokenizer.word_to_index.get(token, 0) for token in context]
            context_tensor = torch.tensor(context_indices).unsqueeze(0)
            
            #Get the predicted token
            output = model(context_tensor)
            _, predicted_idx = torch.max(output, 1)
            predicted_token = tokenizer.index_to_word.get(predicted_idx.item(), '<UNK>')
            
            #Append the predicted token to the list of predictions
            predictions.append(predicted_token)
            context = context[1:] + [predicted_token]
    return predictions

#Main execution
#Load data
with open("corpus.txt", "r") as file:
    corpus = file.readlines()

#Initialize tokenizer and load vocabulary
tokenizer = WordPieceTokenizer()
vocab_file = "vocabulary_66.txt"

#Load vocabulary into tokenizer
with open(vocab_file, "r") as f:
    vocab = [line.strip() for line in f.readlines()]
tokenizer.vocab = vocab
tokenizer.word_to_index = {word: idx for idx, word in enumerate(vocab)}
tokenizer.index_to_word = {idx: word for idx, word in enumerate(vocab)}

#Load Word2Vec model
word2vec = torch.load("word2vec_model1.pth")

vocab_size = word2vec.embedding.weight.shape[0]
context_size = 3
embedding_dim = word2vec.embedding.weight.shape[1]

dataset = NeuralLMDataset(tokenizer, corpus, word2vec, context_size)
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
val_data = dataset.get_validation_data()
val_dataset = torch.utils.data.TensorDataset(*val_data)
val_loader = DataLoader(val_dataset, batch_size=32)

#Initialize models
models = [
    ('NeuralLM_1', NeuralLM_1 (word2vec, vocab_size, context_size, embedding_dim)),
    ('NeuralLM_2', NeuralLM__2(word2vec, vocab_size, context_size, embedding_dim)),
    ('NeuralLM_3', NeuralLM3(word2vec, vocab_size, context_size, embedding_dim))
]

#implementting pipeline for test.txt file
print("\nImplementing pipeline for test.txt file:")
file_path = "test.txt"
if not os.path.exists(file_path):
    print(f"Error: The file '{file_path}' does not exist.")
else:
    print("Predicting next tokens for test sentences:")
    with open(file_path, "r") as file:
        test_sentences = file.readlines()
    for sentence in test_sentences:
        sentence = sentence.strip()
        for model_name, model in models:
            #load the model
            model.load_state_dict(torch.load(f'{model_name}.pth'))
            predictions = predict_next_tokens(model, tokenizer, sentence)
            print(f"\n{model_name} predictions for: {sentence}")
            print(f"Next three tokens: {predictions}")

n = input("Test or Train: ")

if n.lower() == "train":
    epochs = 10
    #Test samples
    with open("sample_test.txt", "r") as file:
        test_sentences = file.readlines()
        
    #Train and evaluate each model
    for model_name, model in models:
        print(f"\nTraining {model_name}")
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        #Train the model
        train_losses, val_losses = train(model, train_loader, val_loader, criterion, optimizer, epochs)
        
        #save the model
        torch.save(model.state_dict(), f'{model_name}.pth')
        
        #Plot losses
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label='Training Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.title(f'{model_name} - Training and Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(f'{model_name}_loss.png')
        plt.close()
        
        #Compute metrics
        train_accuracy = compute_accuracy(model, train_loader)
        val_accuracy = compute_accuracy(model, val_loader)
        train_perplexity = compute_perplexity(model, train_loader, criterion)
        val_perplexity = compute_perplexity(model, val_loader, criterion)
        
        print(f"\n{model_name} Results:")
        print(f"Train Accuracy: {train_accuracy}")
        print(f"Validation Accuracy: {val_accuracy}")
        print(f"Train Perplexity: {train_perplexity}")
        print(f"Validation Perplexity: {val_perplexity}")
        
    #Test prediction on test samples
    with open("sample_test.txt", "r") as file:
        test_sentences = file.readlines()

    print("\nPredicting next tokens for test sentences:")
    for sentence in test_sentences:
        sentence = sentence.strip()
        for model_name, model in models:
            predictions = predict_next_tokens(model, tokenizer, sentence)
            print(f"\n{model_name} predictions for: {sentence}")
            print(f"Next three tokens: {predictions}")

#implementting pipeline for test.txt file
print("\nImplementing pipeline for test.txt file:")
file_path = "test.txt"
if not os.path.exists(file_path):
    print(f"Error: The file '{file_path}' does not exist.")
else:
    print("Predicting next tokens for test sentences:")
    with open(file_path, "r") as file:
        test_sentences = file.readlines()
    for sentence in test_sentences:
        sentence = sentence.strip()
        for model_name, model in models:
            #load the model
            model.load_state_dict(torch.load(f'{model_name}.pth'))
            predictions = predict_next_tokens(model, tokenizer, sentence)
            print(f"\n{model_name} predictions for: {sentence}")
            print(f"Next three tokens: {predictions}")