'''
Task 1 - Assignment 1
WordPiece Tokenizer from scratch

References:
    Logic:
        1. https://www.youtube.com/watch?v=qpv6ms_t_1A&t=1s
        2. https://huggingface.co/learn/nlp-course/en/chapter6/6
    Code Organization and Comments:
        3. https://daily.dev/blog/10-code-commenting-best-practices-for-developers

Note: The assignment uses the word 'token', but the code uses 'element' as the subword token (this is according to the YouTube tutorial)
'''

import re
import json

class WordPieceTokenizer:

    '''
    WordPiece Tokenizer Class
        The main functionality is within the functions:
        1. preprocess_data
        2. construct_vocabulary
        3. tokenize
    '''
    def __init__(self):
        self.vocab = []
        self.element_freq = {}  
        self.word_to_index = {}  
        self.index_to_word = {}
        
    def preprocess_data(self, text):
        '''
        Preprocess text - lowercase, separate punctuation from words, remove redundant spaces
        '''
        text = text.lower()
        text = re.sub(r'([^\w\s])', r' \1 ', text)  # separate punctuation with spaces
        text = re.sub(r'\s+', ' ', text).strip() # remove extra spaces
        return text


    def construct_vocabulary(self, corpus, vocab_size):
        '''
        Constructs a vocabulary of subword tokens/elements
        1. Construct initial vocabulary
        2. Repeat until vocab size is reached
            (i) Parse pairs of elements
            (ii) Find best pair
            (iii) Merge best pair
            (iv) Update elements
            (v) Update vocab and frequencies
        3. Save final vocab
        
        Reference: https://www.youtube.com/watch?v=qpv6ms_t_1A&t=1s
        '''
        elements = []
        for line in corpus:
            line = self.preprocess_data(line)
            words = line.split()
            for word in words:
                if word:
                    elements.append(word[0])  # first char directly
                    for char in word[1:]:
                        elements.append(f"##{char}") # remaining chars with hashes

        # unique elements added to vocab
        self.vocab = sorted(set(elements))
        # element frequencies
        self.element_freq = self.count_frequencies(elements)
        
        self.word_to_index = {word: idx for idx, word in enumerate(self.vocab)}
        self.index_to_word = {idx: word for idx, word in enumerate(self.vocab)}
        
        iteration = 1

        # repeat until we reach req vocab size
        while len(self.vocab) < vocab_size:
            print(f"Iteration: {iteration}")
            pairs = self.parse_pairs(corpus, elements, iteration)

            if not pairs:
                break  

            best_pair = self.find_best_pair(pairs)
            print(f"Best Pair: {best_pair}")

            # merge best pair
            new_element = self.merge_elements(best_pair)
            # update elements
            elements = self.update_elements(elements, best_pair, new_element)

            # update vocab and frequencies
            self.vocab.append(new_element)

            # update element frequencies using the separate function
            self.element_freq = self.count_frequencies(elements)

            iteration += 1
            print("Iteration Complete")
            print("------------------------------------------------------")

        # add special tokens - reference: https://huggingface.co/learn/nlp-course/en/chapter6/6
        # [PAD] - padding token - for padding sequences to same/desired length
        # [UNK] - unknown token - OOV words
        self.vocab = ['[PAD]', '[UNK]'] + self.vocab
        # Save vocab
        self.save_vocabulary()

    def count_frequencies(self, elements):
        '''
        Makes a dict of element frequencies
        '''
        element_freq = {}
        for element in elements:
            element_freq[element] = element_freq.get(element, 0) + 1
        return element_freq

    def parse_pairs(self, corpus, elements, iteration):
        '''
        Calculates freq of pairs of elements
        First iter - pairs from corpus directly
        Subsequent iters - account for merged pairs
        '''
        pairs = {}

        # first iter - pairs from corpus directly
        if (iteration == 1):
            for line in corpus:  
                words = line.split()  
                for word in words:
                    if word:
                        # first char from word - append
                        # remaining char - add hashes then append
                        word_elements = []
                        word_elements.append(word[0])
                        remaining_chars = word[1:]
                        for char in remaining_chars:
                            word_elements.append(f"##{char}")
                        
                        # calc freq of       
                        for i in range(len(word_elements) - 1):
                            pair = (word_elements[i], word_elements[i + 1]) 
                            if (pair not in pairs):
                                pairs[pair] = 0
                            else:
                                pairs[pair] += 1

        else:
            # later - account for merged pairs
            for i in range(len(elements) - 1): 
                if (self.is_valid_pair(elements[i], elements[i + 1])):
                    pair = (elements[i], elements[i + 1])
                    # update freq of pair
                    if (pair not in pairs):
                        pairs[pair] = 0
                    pairs[pair] += 1  

        return pairs



    def is_valid_pair(self, element1, element2):
        '''
        Checks if pair is valid.
        '''
        # we cannot form pairs across words - invlaid pair
        if ((element1.startswith('##')) and (not element2.startswith('##'))):
            return False

        # two starting elements - invalid pair
        # 'hello' and 'world' - invalid pair
        if ((not element1.startswith('##')) and (not element2.startswith('##'))):
            return False

        # valid pair
        return True


    def find_best_pair(self, pairs):
        '''
        Finds the best pair of elements 
        Formula: score = freq of pair / freq of first element * freq of second element
        Reference: https://www.youtube.com/watch?v=qpv6ms_t_1A&t=1s : Timestamp 1.48
        '''
        best_pair = None
        highest_score = float('-inf') 

        # find best pair by score formula
        for pair in pairs:
            
            # numerator
            numerator = pairs[pair]

            # denominator
            first_el_freq = self.element_freq[pair[0]]
            second_el_freq = self.element_freq[pair[1]]
            denominator = first_el_freq * second_el_freq

            # calc score
            if (denominator > 0):
                score = numerator / denominator
            else:
                score = 0

            # higher score - udpate best pair
            if (score > highest_score):
                highest_score = score
                best_pair = pair

        return best_pair


    def merge_elements(self, best_pair):
        '''
        Merges best pair into a new single element
        Reference: https://www.youtube.com/watch?v=qpv6ms_t_1A&t=1s : Timestamp 2.35
        '''
        first_el = best_pair[0] # first element
        # if second el starts with '##' remove it 
        if (best_pair[1].startswith('##')):
            second_el = best_pair[1][2:]
        else:
            second_el = best_pair[1]
        return first_el + second_el


    def update_elements(self, elements, best_pair, new_element):
        '''
        Updates the element list by accounting for merged pairs
        A merged pair becomes a new element
        Reference: https://www.youtube.com/watch?v=qpv6ms_t_1A&t=1s : Timestamp 2.40
        '''
        updated_elements = []
        i = 0
        # loop through elements
        while (i < len(elements)):
            # check for best pair
            # merge new element - skip two ahead
            if (i < len(elements) - 1) and ((elements[i], elements[i + 1]) == best_pair):
                updated_elements.append(new_element)
                i += 2 
            else:
                # regular element - append
                updated_elements.append(elements[i])
                i += 1 # next element
        return updated_elements

    def save_vocabulary(self):
        '''
        Save the resulting vocabulary in a text file named vocabulary_{Group no.}.txt, where each line contains a unique token.
        Group No: 66
        '''
        file_name = 'vocabulary_66.txt'
        with open(file_name, 'w') as f:
            for element in self.vocab:
                f.write(f'{element}\n')
        print(f"Vocabulary saved to {file_name}")
        
    def tokenize(self, sentence):
        '''
        Tokenizes a given sentence into a list of tokens
        - Word in vocab - directly added
        - Word not in vocab - split into chars and append
        '''
        sentence = sentence.lower()
        words = sentence.split()
        tokens = []
        for word in words:
            if (word in self.vocab):
                tokens.append(word)
            else:
                for i in range(len(word)):
                    # first char in vocab - directly added
                    if ((word[i] in self.vocab) and (i==0)):
                        tokens.append(word[i])
                    # if hashed char in vocab - append
                    elif (f'##{word[i]}' in self.vocab):
                        tokens.append(f'##{word[i]}')
                    # else - append char as is
                    else:
                        tokens.append('[UNK]') # unknown token
        return tokens

def main():
    
    # files   
    corpus_file = "corpus.txt"
    test_file = "sample_test.json" 
    output_file = "tokenized_66.json"

    # read corpus
    with open(corpus_file, "r") as f:
        corpus = f.readlines()

    # train tokenizer
    tokenizer = WordPieceTokenizer()
    tokenizer.construct_vocabulary(corpus, vocab_size=10000) 

    # load test data
    with open(test_file, "r") as f:
        test_data = json.load(f)

    # tokenise
    tokenized_data = {}
    for entry in test_data:
        sentence = entry['sentence']
        tokenized_data[entry['id']] = tokenizer.tokenize(sentence)

    # save tokenised output
    with open(output_file, "w") as f:
        json.dump(tokenized_data, f, indent=4)

    print(f"Tokenized output saved to {output_file}")
    
if __name__ == "__main__":
    main()
