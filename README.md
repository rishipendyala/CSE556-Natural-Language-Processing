# CSE556: Natural Language Processing

📚 **Course:** [Natural Language Processing (CSE556)](https://techtree.iiitd.edu.in/viewDescription/filename?=CSE556)   
👨‍🏫 **Instructor:** [Dr. Shad Akhtar](https://scholar.google.co.in/citations?user=KUcO6LAAAAAJ&hl=en), IIIT-Delhi  
🧠 **Semester:** Winter 2025
🧠 **Institute:** IIIT Delhi 
🛠️ **Repo Overview:** This repository contains all the assignments submitted as part of the graduate-level NLP course, covering core and advanced topics in language modeling, sentiment analysis, question answering, and multimodal understanding.

---

## 📌 Course Objectives

- Understand the fundamentals of statistical and neural NLP methods.
- Apply linguistic and syntactic analysis for deeper language understanding.
- Design, implement, and evaluate NLP models for real-world tasks like language modeling, sentiment analysis, and claim normalization.
- Explore state-of-the-art transformer-based and multimodal architectures.

---

## 🧑‍🏫 Course Topics (Brief Overview)

- Text Preprocessing
- Language Modelling
- Word Embeddings (Word2Vec, GloVe, FastText)
- PoS Tagging & Hidden Markov Models
- Sequence Learning
- Neural Language Models (MLP, GRU, LSTM)
- Transformers and Attention Mechanisms
- Fine-tuning of Pretrained Models (BERT, BART, RoBERTa, SpanBERT)
- Sequence Labeling and Aspect-Based Sentiment Analysis
- Text Classification (Fake News, Hate Speech, Deception Detection)
- Conversational Dialogue
- Summarization
- Question Answering (SQuAD v2)
- Multimodal NLP (Sarcasm Explanation via MuSE architecture)
- Syntax Parsing

---

## 📂 Assignment Breakdown

### 📝 Assignment 1

#### Task 1: WordPiece Tokenizer
- Implemented a custom WordPiece tokenizer **from scratch** using only standard Python libraries.
- Created a vocabulary from the corpus and tokenized sentences from a test dataset.
- Output includes a vocabulary file and tokenized JSON output.

#### Task 2: Word2Vec (CBOW) Model
- Built a CBOW-based Word2Vec model from scratch using **PyTorch**.
- Used the tokenizer from Task 1 to prepare the dataset.
- Trained embeddings and computed cosine similarities to validate the model.
- Included training/validation loss plots and similarity analysis.

#### Task 3: MLP-based Neural Language Model
- Developed and trained **three variants** of a multi-layer perceptron (MLP) for next-word prediction.
- Integrated custom Word2Vec embeddings.
- Compared model architectures based on accuracy and perplexity.
- Included a prediction pipeline for next-token generation.

---

### 📝 Assignment 2

#### Task 1: Aspect Term Extraction (ATE)
- Sequence labeling using RNNs/GRUs with GloVe/FastText  
- BIO tagging and F1-score evaluation

#### Task 2: Aspect-Based Sentiment Analysis (ABSA)
- Sentiment classification for aspect terms  
- Models: RNN/GRU/LSTM and fine-tuning BERT, BART, RoBERTa  

#### Task 3: Question Answering with SpanBERT
- Fine-tuning SpanBERT and SpanBERT-CRF for SQuAD v2  
- Evaluation using Exact Match (EM) score

---

### 📝 Assignment 3 

#### Task 1: Transformer from Scratch
- Implementation of transformer components (positional encoding, self-attention, etc.)  
- Language modeling using the Shakespeare dataset  

#### Task 2: Claim Normalization
- Fine-tuning BART and T5 for social media claim rewriting  
- Evaluation using ROUGE-L, BLEU-4, and BERTScore  

#### Task 3: Multimodal Sarcasm Explanation (MuSE)
- Vision + Text fusion using ViT and BART  
- Implementation of Shared Fusion Mechanism  
- Evaluation using ROUGE, BLEU, METEOR, BERTScore  

---

## 📊 Deliverables

Each assignment includes:
- 📜 Well-documented source code (with training, inference, and evaluation pipelines)
- 📝 Reports with architecture choices, hyperparameters, loss curves, evaluation metrics
- 📦 Saved model checkpoints for reproducibility
- 📈 Plots for loss, accuracy, perplexity, and metric trends

---

## 🤝 Acknowledgements

This work was completed under the mentorship of [Dr. Shad Akhtar](https://scholar.google.co.in/citations?user=KUcO6LAAAAAJ&hl=en), whose lectures and assignments deeply strengthened my understanding of modern NLP techniques.
This work was done in a group of three collaborators: Akshat Chaw Parmar, Rishi Pendyala and Vimal Jayant Subburaj

---

## 📬 Contact

Feel free to reach out for collaborations or questions related to the code or topics:  
📧rishi22403@iiitd.ac.in

