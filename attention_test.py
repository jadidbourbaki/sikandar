""" tests for the multi-head self-attention mechanism """

import unittest
import logging
import torch

from attention import MultiHeadAttention


class TestMultiHeadAttention(unittest.TestCase):
    """ tests for the multi-head self-attention mechanism """

    def setUp(self):
        """ set up the test environment """
        logging.basicConfig(level=logging.INFO)
        torch.manual_seed(42)

    def test_small_example(self):
        """ test the multi-head self-attention mechanism with a small example """
        d_model = 4
        num_heads = 2
        # we will use a batch size of 1 for the test

        attention = MultiHeadAttention(d_model, num_heads)

        # input tensor needs shape (batch_size, seq_len, d_model) = (1, 4, 4)
        x = torch.tensor([[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [
                         13, 14, 15, 16]]], dtype=torch.float32)

        logging.info("input: %s", x)

        output = attention(x)  # (1, 4, 4)

        logging.info("attention output: %s", output)

        self.assertEqual(output.shape, (1, 4, 4))

        expected_output = torch.tensor([[[-2.7127, -4.3373, -3.9432,  3.5711],
                                         [-3.2189, -5.5754, -4.6854,  4.1942],
                                         [-3.3166, -5.8078, -4.8310,  4.3139],
                                         [-3.3383, -5.8588, -4.8635,  4.3404]]],
                                       dtype=torch.float32)

        self.assertTrue(torch.allclose(output, expected_output, atol=1e-4))

    def test_causal_mask(self):
        """Test the causal mask in the multi-head attention mechanism"""
        d_model = 4
        num_heads = 2
        attention = MultiHeadAttention(d_model, num_heads)

        seq_len = 3

        x = torch.randn(1, seq_len, d_model)
        logging.info("input: %s", x)

        # this is not the mos rigorous test on the planet, all
        # we test here is whether adding a causal mask changes the output.
        output_no_mask = attention(x, mask=None)  # (1, 3, 4)

        logging.info("output no mask: %s", output_no_mask)

        self.assertEqual(output_no_mask.shape, (1, 3, 4))

        # Create causal mask (lower triangular matrix)
        mask = torch.tril(torch.ones(seq_len, seq_len))
        mask = mask.unsqueeze(0)  # (1, seq_len, seq_len)

        logging.info("causal mask: %s", mask)

        output_with_mask = attention(x, mask=mask)

        logging.info("output with mask: %s", output_with_mask)

        # Shape should still be correct
        self.assertEqual(output_with_mask.shape, (1, 3, 4))

        self.assertFalse(torch.allclose(
            output_no_mask, output_with_mask, atol=1e-6))

    def test_different_num_heads(self):
        """Test that attention works with different numbers of heads"""

        d_model = 8  # Must be divisible by num_heads
        x = torch.randn(1, 4, d_model)

        # Test with 1, 2, 4, and 8 heads
        for num_heads in [1, 2, 4, 8]:
            attention = MultiHeadAttention(d_model, num_heads)
            output = attention(x)

            self.assertEqual(output.shape, (1, 4, d_model))
            logging.info("tested with %d heads, got output shape: %s",
                         num_heads, output.shape)

        self.assertTrue(d_model % num_heads == 0,
                        "d_model must be divisible by num_heads")

    def test_output_shape_invariance(self):
        """Test that output shape matches input shape (batch, seq, d_model)"""
        attention = MultiHeadAttention(d_model=4, num_heads=2)

        # Test different batch sizes and sequence lengths
        test_cases = [
            (1, 5, 4),   # batch=1, seq=5, d_model=4
            (2, 3, 4),   # batch=2, seq=3, d_model=4
            (4, 10, 4),  # batch=4, seq=10, d_model=4
        ]

        for batch_size, seq_len, d_model in test_cases:
            x = torch.randn(batch_size, seq_len, d_model)
            output = attention(x)

            # Output should have same shape as input
            self.assertEqual(output.shape, (batch_size, seq_len, d_model))
            logging.info("shape test: input %s -> output %s",
                         x.shape, output.shape)

    def test_attention_weights_properties(self):
        """Test that attention weights have correct properties 
            The first two axioms of discrete probability theory:
            1. Non-negativity: $P(A) \\geq 0$ for all events $A$
            2. Normalization: $\\sum_{A} P(A) = 1$
        """
        d_model = 4
        num_heads = 2
        attention = MultiHeadAttention(d_model, num_heads)

        seq_len = 3
        x = torch.randn(1, seq_len, d_model)

        # Manually compute attention weights to verify properties
        q = attention.q_linear(x)
        k = attention.k_linear(x)

        batch_size = x.size(0)

        # reshape the query and key vectors to the shape (batch_size, seq_len, num_heads, d_k)
        q = q.view(batch_size, seq_len, num_heads,
                   d_model // num_heads).transpose(1, 2)

        k = k.view(batch_size, seq_len, num_heads,
                   d_model // num_heads).transpose(1, 2)

        # compute the attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(
            torch.tensor(d_model // num_heads, dtype=torch.float32))

        # compute the attention weights (i.e the probabilities of the attention heads)
        attn_weights = torch.softmax(scores, dim=-1)

        # each row of the attention weights should sum to 1 (softmax property)
        sums = attn_weights.sum(dim=-1)
        self.assertTrue(torch.allclose(sums, torch.ones_like(sums), atol=1e-5))

        # all values of the attention weights should be between 0 and 1
        self.assertTrue((attn_weights >= 0).all()
                        and (attn_weights <= 1).all())

        logging.info("attention weights sum to 1 and are in [0, 1]")
        logging.info("attention weights shape: %s", attn_weights.shape)

    def test_single_head_vs_multi_head(self):
        """Compare single head vs multi-head to see the difference"""
        d_model = 8
        x = torch.randn(1, 4, d_model)

        single_head = MultiHeadAttention(d_model, num_heads=1)
        multi_head = MultiHeadAttention(d_model, num_heads=4)

        output_single = single_head(x)
        output_multi = multi_head(x)

        # Both should have same output shape
        self.assertEqual(output_single.shape, output_multi.shape)
        self.assertEqual(output_single.shape, (1, 4, d_model))

        # But outputs should be different (different learned weights)
        self.assertFalse(torch.allclose(
            output_single, output_multi, atol=1e-6))

        logging.info(
            "single head vs multi-head: same shape, different outputs")
        logging.info("single head output norm: %.4f", output_single.norm())
        logging.info("multi head output norm: %.4f", output_multi.norm())

    def test_edge_case_single_token(self):
        """Test attention with a sequence of length 1"""
        attention = MultiHeadAttention(d_model=4, num_heads=2)
        x = torch.randn(1, 1, 4)  # batch=1, seq_len=1, d_model=4

        output = attention(x)

        self.assertEqual(output.shape, (1, 1, 4))
        logging.info("edge case: single token sequence works")


if __name__ == "__main__":
    unittest.main()
