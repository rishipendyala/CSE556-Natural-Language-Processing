import re
import json
from collections import defaultdict, Counter

class WordPieceTokenizer:
    def __init__(self):
        self.vocab = []
        self.token_freq = Counter()

    def preprocess_data(self, text):
        text = text.lower()
        # replace punctuation with empty string
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text) # removes char that is not a-z, A-Z, 0-9, space
        # remove extra spaces
        text = re.sub('r\s+', '', text) # removes multiple spaces, space followed by tab or \n
        return text

    def construct_vocabulary(self, corpus, vocab_size):
        tokens = []
        for line in corpus:
            words = line.split()
            for word in words:
                if word:
                    tokens.append(word[0])
                    for char in word[1:]:
                        tokens.append(f'##{char}')

        self.vocab = sorted(set(tokens))
        self.token_freq = Counter(tokens)
        
        iter=1

        while len(self.vocab) < vocab_size:
            print(f"Iteration: {iter}")
            pairs = defaultdict(int)

            if iter == 1:
                for line in corpus:
                    words = line.split()
                    for word in words:
                        word_tokens = [word[0]] + [f'##{char}' for char in word[1:]]
                        for i in range(len(word_tokens) - 1):
                            pair = (word_tokens[i], word_tokens[i + 1])
                            pairs[pair] += 1
                            
            else:
                for i in range(len(tokens) - 1):
                    if tokens[i].startswith('##') and (not tokens[i + 1].startswith('##')):
                        continue
                    elif (not tokens[i].startswith('##')) and (not tokens[i + 1].startswith('##')):
                        continue
                    else:
                        pair = (tokens[i], tokens[i + 1])
                        pairs[pair] += 1

            if not pairs:
                break

            pair_scores = {}
            for pair in pairs:
                try:
                    score = pairs[pair] / (self.token_freq[pair[0]] * self.token_freq[pair[1]])
                    pair_scores[pair] = score
                except ZeroDivisionError:
                    print(f"Tokens: {self.token_freq}")
                    print(f"Division by zero error for pair: {pair}")
                    print(f"Frequency of {pair[0]}: {self.token_freq[pair[0]]}")
                    print(f"Frequency of {pair[1]}: {self.token_freq[pair[1]]}")

            best_pair = max(pairs, key=lambda pair: pairs[pair] / (self.token_freq[pair[0]] * self.token_freq[pair[1]]))

            new_token = best_pair[0] + (best_pair[1][2:] if best_pair[1].startswith("##") else best_pair[1])
            self.vocab.append(new_token)

            print("Best Pair: ", best_pair)

            new_tokens = []
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == best_pair:
                    new_tokens.append(new_token)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens
            self.token_freq = Counter(tokens)

            iter+=1
            print("Iteration Complete")
            print("------------------------------------------------------")

        print(self.vocab)
        with open('vocabulary_66.txt', 'w') as f:
            for token in self.vocab:
                f.write(f'{token}\n')

    def tokenize(self, sentence):
        sentence = self.preprocess_data(sentence)
        words = sentence.split()
        tokens = []
        for word in words:
            if word in self.vocab:
                tokens.append(word)
            else:
                for i in range(len(word)):
                    if word[i] in self.vocab and i==0:
                        tokens.append(word[i])
                    elif f'##{word[i]}' in self.vocab:
                        tokens.append(f'##{word[i]}')
                    else:
                        tokens.append(word[i])
        return tokens

if __name__ == "__main__":
    corpus_file = 'corpus.txt'
    with open(corpus_file, 'r') as f:
        corpus = f.readlines()
    
    tokenizer = WordPieceTokenizer()
    tokenizer.construct_vocabulary(corpus, vocab_size=1000)
    print(tokenizer.tokenize("hugging is crappy and quick and just fuzzy impostor two custom dumping blush"))
