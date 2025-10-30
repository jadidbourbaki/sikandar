"""
Minimal GPT-style transformer for learning
"""

import torch
import torch.nn as nn

from transformer_block import TransformerBlock


class SikandarModel(nn.Module):
    """Decoder-only transformer (GPT-style)"""

    def __init__(self, vocab_size: int, d_model: int, num_heads: int,
                 num_layers: int, max_len: int) -> None:
        """Initialize the Sikandar model.
        Args:
            vocab_size (int): the size of the vocabulary
            d_model (int): the dimension of the model
            num_heads (int): the number of attention heads in the transformer blocks
            num_layers (int): the number of transformer blocks
            max_len (int): the maximum length of the input sequence
        """
        super().__init__()

        self.vocab_size = vocab_size
        self.d_model = d_model

        # -- token embeddings --
        # convert token IDs to vectors using an embedding layer
        # the embedding layer is a simple lookup table that maps each
        # token ID to a vector of dimension d_model
        # the token embeddings are used to represent the input tokens in the model
        # for example, if the token ID is 1, the embedding vector might be
        # something like [0.1, 0.2, 0.3, ...]
        # and if the token ID is 2, the embedding vector might be something
        # like [0.4, 0.5, 0.6, ...] and so on.
        # this is mathematically f: {0, 1, ..., vocab_size-1} -> R^{d_model}
        # where the domain is the set of token IDs (integers from 0 to vocab_size-1)
        # and R^{d_model} is the set of real-valued vectors of dimension d_model
        self.token_embedding = nn.Embedding(vocab_size, d_model)

        # -- position embeddings --
        # convert position indices to vectors using an embedding layer
        # the position embeddings are used to represent the position of the tokens in the sequence
        # similar to the token embeddings,
        # this is mathematically f: {0, 1, ..., max_len-1} -> R^{d_model}
        # (note the max_len instead of vocab_size because the position indices
        # are not limited to the vocabulary size)
        self.pos_embedding = nn.Embedding(max_len, d_model)

        # Stack of transformer blocks
        # the transformer blocks are the main building blocks of the model
        # each transformer block consists of a multi-head attention mechanism
        # and a feed-forward network.
        # the multi-head attention mechanism is used to capture the dependencies
        # between the tokens in the sequence.
        # the feed-forward network is used to capture the non-linear dependencies
        # between the tokens in the sequence.
        # the transformer blocks are stacked to form the main architecture of the model
        # the output of the last transformer block is used to generate the next token predictions
        self.blocks = nn.ModuleList([
            # each transformer block has d_model hidden units, num_heads attention heads,
            # and a feed-forward network with d_model * 4 hidden units
            # note: there is no deep reason for the factor of 4, it is just a common choice
            TransformerBlock(d_model, num_heads, d_model * 4)
            for _ in range(num_layers)
        ])

        # -- layer normalization --
        # normalize the hidden units using a layer normalization layer
        # the layer normalization layer is used to stabilize the training process
        # by subtracting the mean of the hidden unit vectors and dividing by the standard deviation
        # this ensures that the hidden units have a mean of 0 and a standard deviation of 1
        self.norm = nn.LayerNorm(d_model)

        # -- output layer --
        # convert the hidden units to token predictions using a linear layer
        # the linear layer is a simple lookup table that maps each vector to a token ID
        self.output = nn.Linear(d_model, vocab_size)

        # -- initializing weights --
        # initialize the weights of the model using a Xavier uniform initialization
        # Xavier uniform initialization is a common initialization technique for neural networks
        # it is used to initialize the weights of the model in a way that ensures that the model
        # is not too sensitive to the initial weights
        # this is done by initializing the weights to random values from a uniform distribution
        # between $-\sqrt{6/(n_{in} + n_{out})}$ and $\sqrt{6/(n_{in} + n_{out})}$
        # where $n_{in}$ is the number of input units and $n_{out}$ is the number of output units
        # conceptually, this is similar to the idea of just initializing the weights to
        # random values from a uniform distribution between -1 and 1.
        self._init_weights()

    def _init_weights(self):
        """Initialize weights using Xavier uniform initialization"""
        for p in self.parameters():
            if p.dim() > 1:
                # initialize the weights of the model using a Xavier uniform initialization
                nn.init.xavier_uniform_(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for the Sikandar model
        Args:
            x (torch.Tensor): the input tensor of shape (batch_size, seq_len)
        Returns:
            torch.Tensor: the output tensor of shape (batch_size, seq_len, vocab_size)
        """
        # Note: the input tensor x is a tensor of shape (batch_size, seq_len)
        # each row is a sequence of token ids which represent the input sequence for the model
        # there are batch_size such sequences in the tensor which are taken from the
        # training dataset

        # get the batch size and sequence length from the input tensor
        batch_size, seq_len = x.size()

        # -- causal mask --
        # create a causal mask using a lower triangular matrix
        # A causal mask prevents the model from attending to future tokens in the sequence.
        # It does this by masking the attention scores for future tokens.
        # This is done by creating a lower triangular matrix where:
        # - Lower triangular part = 1 (past/present tokens can attend to each other)
        # - Upper triangular part = 0 (future tokens are masked out)
        # When applied, positions where mask == 0 get their attention scores set to -infinity,
        # which after softmax become 0 probability, effectively preventing
        # attention to future tokens.

        # create a ones matrix of the right size, i.e., (seq_len, seq_len)
        ones_matrix = torch.ones(seq_len, seq_len, device=x.device)
        # create the lower triangular matrix of ones and zeros
        mask = torch.tril(ones_matrix)
        # expand the mask to the batch size
        mask = mask.unsqueeze(0).expand(
            batch_size, -1, -1)  # (batch_size, seq_len, seq_len)

        # convert the token ids in the input tensor to token embeddings
        # using the token embedding layer
        token_emb = self.token_embedding(x)  # (batch_size, seq_len, d_model)

        # get the position indices for the input tensor
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)

        # convert the position indices in the input tensor to position embeddings
        # using the position embedding layer
        # note: there do not need to be batch_size of these position embeddings,
        # because the position embeddings are the same for all sequences in the batch
        # so we can just create a single tensor of shape (1, seq_len, d_model)
        pos_emb = self.pos_embedding(positions)  # (1, seq_len, d_model)

        # add the token embeddings and the position embeddings
        # to get the input for the transformer blocks
        x = token_emb + pos_emb  # (batch_size, seq_len, d_model)

        # Apply transformer blocks
        for block in self.blocks:
            x = block(x, mask)

        # normalize the hidden units using the normalization layer
        # we created earlier.
        x = self.norm(x)

        # convert the normalized hidden units to token predictions using the
        # output linear layer we created earlier.
        return self.output(x)  # (batch_size, seq_len, vocab_size)
