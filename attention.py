"""
This file implements the multi-head self-attention mechanism for sikandar.
The multi-head self-attention mechanism is the core of the transformer architecture.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention: the core of transformers"""

    def __init__(self, d_model: int, num_heads: int) -> None:
        """Initialize the multi-head attention mechanism.
        Args:
            d_model (int): the dimension of the model
            num_heads (int): the number of attention heads in the transformer blocks
        """
        super().__init__()

        # -- attention mechanism --
        # Self-attention captures dependencies between tokens by comparing queries (Q) and keys (K)
        # and using the resulting weights to mix values (V).
        # For one head, the similarity function is
        # $s(q, k) = (q^T k) / sqrt(d_k)$ where $q, k \in \mathbb{R}^{d_k}$
        # and $d_k = d_model // num_heads$.
        # In code, we first project x -> Q, K, V of shape
        # (batch_size, seq_len, d_model), then reshape to
        # (batch_size, num_heads, seq_len, d_k) with $d_k = d_model // num_heads$.
        # The score tensor has shape $(batch_size, num_heads, seq_len, seq_len)$,
        # softmax over the last dim gives attention weights $a(q, k)$, which weight V to
        # produce the head outputs.

        # assert that the dimension of the model is divisible by the number of attention heads
        # this is necessary for the multi-head attention mechanism to work
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads

        # the dimension of the key vectors for each attention head
        self.d_k = d_model // num_heads

        # Linear layers to compute Q, K, V from input
        # Q is the query vector, K is the key vector, V is the value vector.
        # Q, K, V are all 3D tensors of shape (batch_size, seq_len, d_model).
        # Q, K, V are computed by projecting the input tensor x through 3 linear layers:
        # Q = W_q * x, K = W_k * x, V = W_v * x.
        # The linear layers are used to project the input tensor x
        # into the query, key, and value vectors.
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)

        self.out = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        """Forward pass for the multi-head attention mechanism"""
        batch_size, seq_len = x.size(0), x.size(1)

        # Compute Q, K, V: $(batch_size, seq_len, d_model)$
        # Recall that Q, K, V are the query, key, and value vectors respectively.
        q = self.q_linear(x)
        k = self.k_linear(x)
        v = self.v_linear(x)

        # Reshape to (batch, num_heads, seq_len, d_k)
        # the view method is used to reshape the tensor into a new shape.
        # the transpose method is used to transpose the tensor.
        q = q.view(batch_size, seq_len, self.num_heads,
                   self.d_k)  # (batch_size, seq_len, num_heads, d_k)
        # the parameters to the transpose method are the dimensions to transpose.
        # in our case 1, 2 means transpose the tensor from the 1st to the 2nd dimension.
        q = q.transpose(1, 2)  # (batch_size, num_heads, seq_len, d_k)

        # we repeat the same process for the key and value vectors.
        k = k.view(batch_size, seq_len, self.num_heads,
                   self.d_k)  # (batch_size, seq_len, num_heads, d_k)
        k = k.transpose(1, 2)  # (batch_size, num_heads, seq_len, d_k)

        v = v.view(batch_size, seq_len, self.num_heads,
                   self.d_k)  # (batch_size, seq_len, num_heads, d_k)
        v = v.transpose(1, 2)  # (batch_size, num_heads, seq_len, d_k)

        # Attention scores: $(batch_size, num_heads, seq_len, seq_len)$
        # we compute the attention scores by taking the dot product of the query and key vectors.
        # the key vector is transposed with respect to the last two dimensions i.e
        # the last two dimensions are swapped.
        # by the rules of matrix multiplication the final shape of the scores tensor is:
        # $(batch_size, num_heads, seq_len, seq_len) =
        # (batch_size, num_heads, seq_len, d_k) \times (batch_size, num_heads, d_k, seq_len)$.
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)

        # Apply causal mask (prevent seeing future tokens)
        if mask is not None:
            # the future tokens are masked out by setting the scores to -infinity.
            # when the softmax function is applied to the scores,
            # the -infinity values will become 0.

            # we need to unsqueeze the mask to add a dimension for the number of heads
            # this is because the mask is a 2D tensor and we need to add a dimension
            # for the number of attention heads so that the mask can be broadcasted
            # to the scores tensor
            mask = mask.unsqueeze(1)  # (batch, 1, seq_len, seq_len)
            scores = scores.masked_fill(mask == 0, float('-inf'))

        # Softmax to get attention weights
        attn_weights = F.softmax(scores, dim=-1)

        # Apply attention to values
        # $(batch_size, num_heads, seq_len, d_k) =
        # (batch_size, num_heads, seq_len, seq_len) \times (batch_size, num_heads, seq_len, d_k)$.
        attn_output = torch.matmul(attn_weights, v)

        # Concatenate heads: (batch, seq_len, d_model)
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model
        )

        return self.out(attn_output)
