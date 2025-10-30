"""
Simple tests for tokenizer
"""

import unittest
import tempfile
import os
from tokenizer import Tokenizer


class TestTokenizer(unittest.TestCase):
    """Test cases for the Tokenizer class"""

    def setUp(self):
        """Set up tokenizer with sample data"""
        self.tokenizer = Tokenizer(vocab_size=100)

        # Sample texts for building vocabulary
        self.texts = [
            "Hello world! This is a test.",
            "The quick brown fox jumps over the lazy dog.",
            "Machine learning is fascinating."
        ]
        self.tokenizer.build_vocab(self.texts)

    def test_initialization(self):
        """Test tokenizer has correct special tokens"""
        self.assertEqual(self.tokenizer.special['<PAD>'], 0)
        self.assertEqual(self.tokenizer.special['<UNK>'], 1)
        self.assertEqual(self.tokenizer.special['<BOS>'], 2)
        self.assertEqual(self.tokenizer.special['<EOS>'], 3)

    def test_preprocess(self):
        """Test text preprocessing"""
        self.assertEqual(self.tokenizer.preprocess(
            "Hello World!"), "hello world")
        self.assertEqual(self.tokenizer.preprocess(
            "  Multiple   spaces  "), "multiple spaces")
        self.assertEqual(self.tokenizer.preprocess(
            "Hello, world!"), "hello world")

    def test_build_vocab(self):
        """Test vocabulary building"""
        # Should have more than just special tokens
        self.assertGreater(self.tokenizer.get_vocab_size(), 4)

        # Special tokens should be in vocabulary
        self.assertIn('<PAD>', self.tokenizer.word_to_token)
        self.assertIn('<UNK>', self.tokenizer.word_to_token)

    def test_encode_decode(self):
        """Test encoding and decoding"""
        text = "hello world test"

        # Encode text to token IDs
        token_ids = self.tokenizer.encode(text)
        self.assertIsInstance(token_ids, list)
        self.assertTrue(all(isinstance(tid, int) for tid in token_ids))

        # Decode back to text
        decoded = self.tokenizer.decode(token_ids)
        self.assertEqual(decoded, text)

    def test_unknown_words(self):
        """Test handling of unknown words"""
        text = "hello xyzunknownword test"
        token_ids = self.tokenizer.encode(text)
        # Unknown words should get <UNK> token ID
        self.assertIn(self.tokenizer.special['<UNK>'], token_ids)

    def test_encode_with_special_tokens(self):
        """Test encoding with BOS and EOS"""
        text = "hello world"
        token_ids = self.tokenizer.encode_with_special_tokens(text)

        # Should start with BOS and end with EOS
        self.assertEqual(token_ids[0], self.tokenizer.special['<BOS>'])
        self.assertEqual(token_ids[-1], self.tokenizer.special['<EOS>'])

        # Should be 2 tokens longer than regular encode
        regular_ids = self.tokenizer.encode(text)
        self.assertEqual(len(token_ids), len(regular_ids) + 2)

    def test_save_and_load_vocab(self):
        """Test saving and loading vocabulary"""
        # Save to temp file
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            temp_file = f.name

        try:
            self.tokenizer.save_vocab(temp_file)

            # Load into new tokenizer
            new_tokenizer = Tokenizer()
            new_tokenizer.load_vocab(temp_file)

            # Vocabularies should match
            self.assertEqual(self.tokenizer.word_to_token,
                             new_tokenizer.word_to_token)
        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)


if __name__ == '__main__':
    unittest.main()
