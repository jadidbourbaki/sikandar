"""
Simple word-level tokenizer
"""

import re
import json
from collections import Counter


class Tokenizer:
    """Word-level tokenizer with simple preprocessing"""

    def __init__(self, vocab_size=10000):
        self.vocab_size = vocab_size
        self.word_to_token = {}
        self.token_to_word = {}

        # Special tokens
        self.special = {
            '<PAD>': 0,
            '<UNK>': 1,
            '<BOS>': 2,
            '<EOS>': 3
        }
        self.word_to_token.update(self.special)
        self.token_to_word = {v: k for k, v in self.special.items()}

    def get_pad_id(self):
        """Get padding token ID"""
        return self.special['<PAD>']

    def preprocess(self, text):
        """Basic preprocessing: lowercase and normalize"""
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)  # Remove punctuation
        text = re.sub(r'\s+', ' ', text)      # Normalize whitespace
        return text.strip()

    def build_vocab(self, texts):
        """Build vocabulary from training texts"""
        # Count word frequencies
        word_counts = Counter()
        for text in texts:
            words = self.preprocess(text).split()
            word_counts.update(words)

        # Add most common words to vocabulary
        for word, _ in word_counts.most_common(self.vocab_size - len(self.special)):
            if word not in self.word_to_token:
                token_id = len(self.word_to_token)
                self.word_to_token[word] = token_id
                self.token_to_word[token_id] = word

    def encode(self, text):
        """Convert text to list of token IDs"""
        words = self.preprocess(text).split()
        return [self.word_to_token.get(word, self.special['<UNK>']) for word in words]

    def decode(self, token_ids):
        """Convert token IDs back to text"""
        words = []
        for tid in token_ids:
            word = self.token_to_word.get(tid, '<UNK>')
            if word not in ['<PAD>', '<BOS>', '<EOS>']:
                words.append(word)
        return ' '.join(words)

    def encode_with_special_tokens(self, text):
        """Encode with BOS and EOS tokens"""
        tokens = self.encode(text)
        return [self.special['<BOS>']] + tokens + [self.special['<EOS>']]

    def get_vocab_size(self):
        """Get vocabulary size"""
        return len(self.word_to_token)

    def get_special_token_ids(self):
        """Get special token IDs"""
        return self.special.copy()

    def save_vocab(self, filepath):
        """Save vocabulary to JSON file"""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.word_to_token, f, indent=2)

    def load_vocab(self, filepath):
        """Load vocabulary from JSON file"""
        with open(filepath, 'r', encoding='utf-8') as f:
            self.word_to_token = json.load(f)
        self.token_to_word = {v: k for k, v in self.word_to_token.items()}
