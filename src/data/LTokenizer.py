import pickle
from collections import namedtuple

class LTokenizer:
    def __init__(self, words):
        self.words = words

    def save(self, filename):
        with open(filename, 'wb') as file:
            pickle.dump(self.words, file)

    @staticmethod
    def from_file(filename):
        with open(filename, 'rb') as file:
            words = pickle.load(file)
        return LTokenizer(words)

    def get_vocab(self):
        return self.words

    def get_vocab_size(self):
        return len(self.words)

    def encode(self, sentence, add_special_tokens=True):
        tokens = [self.words.index(word) if word in self.words else self.words.index('[UNK]') for word in sentence]
        return type('', (object,), {'ids': tokens})()

    def decode(self, tokens):
        words = [self.words[token] if len(self.words) < token else '[UNK]' for token in tokens]
        return words