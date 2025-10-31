"""
Minimal training script for learning LLM implementation
"""

import argparse
import logging
import pathlib
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from model import SikandarModel
from tokenizer import Tokenizer


class TextDataset(Dataset):
    """Simple dataset that converts text to token sequences"""

    def __init__(self, texts, tokenizer, max_length):
        """Initialize the TextDataset"""
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_id = tokenizer.get_pad_id()
        self.data = []

        logging.info("tokenizing texts and creating chunks...")
        for text in tqdm(texts, desc="processing texts"):
            tokens = self.tokenizer.encode_with_special_tokens(text)
            self._chunkify(tokens)

        logging.info("dataset created: %d chunks from %d texts",
                     len(self.data), len(texts))

    def _chunkify(self, tokens):
        """Chunkify tokens into max length sequences"""
        for i in range(0, len(tokens), self.max_length):
            chunk = tokens[i:i + self.max_length]
            if len(chunk) < self.max_length:
                chunk = self._pad_sequence(chunk)
            assert len(chunk) == self.max_length
            self.data.append(chunk)

    def _pad_sequence(self, sequence):
        """pad sequence to max length"""
        return sequence + [self.pad_id] * (self.max_length - len(sequence))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """Get a tokenized chunk for training"""
        seq = torch.tensor(self.data[idx], dtype=torch.long)

        # for next-token prediction:
        # the input is [0:n-1] and the target is [1:n] (shifted by 1)
        # for example:
        # if the sequence is [1, 2, 3, 4, 5],
        # the input is [1, 2, 3, 4] and the target is [2, 3, 4, 5]
        # note that the integers in the sequence are token IDs, not words
        return seq[:-1], seq[1:]


def load_text_data(filepath: pathlib.Path) -> [str]:
    """Load text file, one line per sample"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]


def train_sikandar():
    """Train the Sikandar model"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-data', type=str, required=True)
    parser.add_argument('--val-data', type=str, required=True)
    parser.add_argument('--output-dir', type=str, default='./output')

    # model hyperparameters
    parser.add_argument('--vocab-size', type=int, default=10000,
                        help='max number of unique words (actually tokens) in the vocabulary')
    parser.add_argument('--d-model', type=int, default=512,
                        help='dimension of the model, which is the number of' +
                             'hidden units in the transformer blocks')
    parser.add_argument('--num-heads', type=int, default=8,
                        help='number of attention heads in the transformer blocks')
    parser.add_argument('--num-layers', type=int, default=6,
                        help='number of transformer blocks')
    parser.add_argument('--max-len', type=int, default=512,
                        help='max number of words (actually tokens) in a sequence')

    # training hyperparameters
    parser.add_argument('--batch-size', type=int, default=32,
                        help='the number of samples in one training batch')
    parser.add_argument('--learning-rate', type=float, default=1e-4,
                        help='the learning rate for the optimizer')
    parser.add_argument('--num-epochs', type=int, default=10,
                        help='the number of epochs to train for')

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info("using device: %s", device)

    # Load training data
    train_texts = load_text_data(args.train_data)
    logging.info("loaded %d train samples", len(train_texts))

    # Load validation data
    val_texts = load_text_data(args.val_data)
    logging.info("loaded %d val samples", len(val_texts))

    # Build the vocabulary
    tokenizer = Tokenizer(vocab_size=args.vocab_size)
    tokenizer.build_vocab(train_texts)
    logging.info("vocabulary size: %d", tokenizer.get_vocab_size())

    # Create the training dataset
    train_dataset = TextDataset(train_texts, tokenizer, args.max_len)
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True)

    logging.info("created training dataset with %d samples",
                 len(train_dataset))

    # Create the validation dataset
    val_dataset = TextDataset(val_texts, tokenizer, args.max_len)
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False)

    logging.info("created validation dataset with %d samples",
                 len(val_dataset))

    # Create model
    model = SikandarModel(
        vocab_size=tokenizer.get_vocab_size(),
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        max_len=args.max_len
    )

    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    logging.info("created sikandar model with %d parameters", total_params)

    # Using the cross entropy loss function
    # https://en.wikipedia.org/wiki/Cross-entropy
    # also helpful to look at the pytorch documentation on the nn.CrossEntropyLoss function
    # https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
    pad_id = tokenizer.get_pad_id()  # we ignore padding tokens in the loss calculation
    criterion = nn.CrossEntropyLoss(ignore_index=pad_id)

    # Using the Adam optimizer: a variant of stochastic gradient descent
    # https://en.wikipedia.org/wiki/Stochastic_gradient_descent#Adam
    # the original paper on Adam: https://arxiv.org/abs/1412.6980
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    # Training loop
    logging.info("starting training loop for %d epochs", args.num_epochs)
    for epoch in range(args.num_epochs):
        logging.info("starting epoch %d", epoch+1)

        # Train
        model.train()  # set the model to training mode
        train_loss = 0.0  # reset the training loss for this epoch

        # iterate over the training dataset in batches
        # take a look at the __getitem__ method in the TextDataset class to
        # understand the input and target ids
        for input_ids, target_ids in tqdm(train_loader, desc=f"epoch {epoch+1}"):
            # input_ids is a tensor of shape (batch_size, seq_len)
            # each row is a sequence of token ids which represent the input sequence for the model
            # there are batch_size such sequences in the tensor which are taken from the
            # training dataset
            input_ids = input_ids.to(device)

            # target_ids is also a tensor of shape (batch_size, seq_len)
            # each row is a sequence of token ids which represent the target sequence for the model
            # there are batch_size such sequences in the tensor which are taken from the
            # training dataset
            target_ids = target_ids.to(device)

            optimizer.zero_grad()  # reset the gradients for this batch

            # forward pass: get the logits for the next token predictions

            # -- note on logits --
            # here is some information on what a "logit" is: https://en.wikipedia.org/wiki/Logit
            # the logits are *not* probabilities because they are not normalized to sum to 1
            # but they are similar in that they are "scores" for the next token predictions

            # -- the logits tensor --
            # the shape of the logits tensor is (batch_size, seq_len, vocab_size)
            # each element in the tensor is the "score" for the next token prediction
            # for the corresponding sequence and the corresponding position in the sequence
            # the logits tensor is used to compute the loss for the next token predictions
            # the loss is computed by comparing the predicted "scores" with the target token ids
            # the loss is then used to update the model parameters using the optimizer

            # -- big picture --
            # for each sequence in the batch, we get a list of "scores" for vocabulity word
            # we can then use the softmax function to convert the scores to probabilities
            # we can then compare the word with the highest probability to the target token id
            # and compute the loss for that sequence
            # we then backpropagate the loss to update the model parameters
            logits = model(input_ids)  # (batch, seq_len, vocab_size)

            # -- flattening the logits tensor for the loss calculation --
            # the logits tensor is a 3D tensor with shape (batch_size, seq_len, vocab_size)
            # reshaped_logits is a 2D tensor with shape (batch_size * seq_len, vocab_size)
            # each row in the reshaped_logits tensor is a sequence of "scores"
            # for the next token predictions
            # the reason we need to flatten it is because the loss function expects a 2D tensor
            # there is not much conceptually to learn here, but it is a common pattern in PyTorch
            reshaped_logits = logits.reshape(-1, logits.size(-1))

            # -- flattening the target_ids tensor for the loss calculation --
            # recall that target_ids is a tensor of shape (batch_size, seq_len)
            # reshaped_target_ids is a 1D tensor with shape (batch_size * seq_len)
            # each element in the reshaped_target_ids tensor is the target token id
            # for the corresponding sequence and the corresponding position in the sequence
            # again, the reason we need to flatten it is because the loss
            # function expects a 1D tensor
            reshaped_target_ids = target_ids.reshape(-1)

            # -- calculating the loss --
            # In row r and column c in reshaped_logits, we have the "score"
            # for the next token prediction
            # for the r-th sequence and the c-th position in the sequence
            # In row r and column c in reshaped_target_ids, we have the target token id
            # for the r-th sequence and the c-th position in the sequence
            # we compare these i.e the array of length vocab_size and the single target token id
            # and compute the loss for that sequence
            # we then backpropagate the loss to update the model parameters using the optimizer
            loss = criterion(reshaped_logits, reshaped_target_ids)

            # -- backpropagation --
            # we compute the gradient of the loss with respect to the model parameters
            # we then update the model parameters using the optimizer
            # this is just a fancy way of saying we use stochastic gradient descent
            # to update the model parameters
            loss.backward()

            # -- updating the model parameters --
            # we update the model parameters using the optimizer
            optimizer.step()

            # -- updating the training loss --
            # we add the loss for this batch to the training loss
            # this is done by using the item method to get the scalar value of the loss
            # we then add it to the training loss for this epoch
            train_loss += loss.item()

        # calculate the average training loss for this epoch
        train_loss /= len(train_loader)
        logging.info("training loss for epoch %d is %f", epoch+1, train_loss)

        logging.info("starting validation for epoch %d", epoch+1)

        # set the model to evaluation mode
        model.eval()

        # reset the validation loss for this epoch
        val_loss = 0.0

        # iterate over the validation dataset in batches
        with torch.no_grad():  # no need to compute gradients for validation
            # we already discussed what input_ids and target_ids are in the training loop
            # so we won't repeat it here
            for input_ids, target_ids in tqdm(val_loader, desc=f"epoch {epoch+1}"):
                input_ids = input_ids.to(device)
                target_ids = target_ids.to(device)

                # forward pass: get the logits for the next token predictions
                logits = model(input_ids)

                # flatten the logits tensor for the loss calculation
                reshaped_logits = logits.reshape(-1, logits.size(-1))

                # flatten the target_ids tensor for the loss calculation
                reshaped_target_ids = target_ids.reshape(-1)

                # calculate the loss
                loss = criterion(reshaped_logits, reshaped_target_ids)

                # add the loss for this batch to the validation loss
                # this is done by using the item method to get the scalar value of the loss
                # we then add it to the validation loss for this epoch
                val_loss += loss.item()

        # calculate the average validation loss for this epoch
        val_loss /= len(val_loader)
        logging.info("validation loss for epoch %d is %f", epoch+1, val_loss)

        logging.info("ending epoch %d", epoch+1)

    logging.info("training completed for %d epochs", args.num_epochs)

    # Save model and vocabulary
    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    model_path = output_dir / "model.pt"
    vocab_path = output_dir / "vocab.json"

    logging.info("saving model to %s", model_path)
    torch.save({
        'model_state_dict': model.state_dict(),
        'd_model': args.d_model,
        'num_heads': args.num_heads,
        'num_layers': args.num_layers,
        'max_len': args.max_len,
        'vocab_size': tokenizer.get_vocab_size(),
    }, model_path)

    logging.info("saving vocabulary to %s", vocab_path)
    tokenizer.save_vocab(vocab_path)

    logging.info("model and vocabulary saved!")


if __name__ == '__main__':
    train_sikandar()
