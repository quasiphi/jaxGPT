#!/usr/bin/env python3

import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import jax.random as random
from flax.training import train_state
from optax import adamw, softmax_cross_entropy

from jaxgpt.utils.checkpoints import save_checkpoint
from jaxgpt.models.gpt import GPT
from jaxgpt.data.sanitize import prepare_file
from jaxgpt.data.data_loader import DataLoader

def create_train_state(rng, model, learning_rate):
    # variables?
    params = model.init(rng, jnp.ones((1, 512), dtype=jnp.int32))
    tx = adamw(learning_rate)
    return train_state.TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx
    )

def loss_fn(model, params, inputs, labels):
    logits = model.apply(params, inputs)
    loss = jnp.mean(softmax_cross_entropy(logits, labels))
    return loss

def train_epoch(model, state, train_loader, rng):
    epoch_loss = 0.0
    for batch in train_loader:
        inputs, labels = batch, batch
        loss, grads = jax.value_and_grad(loss_fn)(model, state.params, inputs, labels)
        state = state.apply_gradients(grads=grads)
        epoch_loss += loss
    return state, epoch_loss / len(train_loader)

def train_model(rng, train_loader, model, epochs, learning_rate):
    state = create_train_state(rng, model, learning_rate)
    # TODO: tqdm
    for epoch in range(epochs):
        state, avg_loss = train_epoch(model, state, train_loader, rng)
        print(f'Epoch {epoch}, loss: {avg_loss}')
        save_checkpoint(state, epoch)

if __name__ == '__main__':
    rng = random.PRNGKey(80085)

    file_name = sys.argv[1]
    path = Path(file_name).absolute().resolve()
    assert path.exists(), f'Object at supplied path does not exist: {path}'
    assert path.is_file(), f'Object at supplied path is not a file: {path}'
    train_data = prepare_file(path)

    with open(path, 'r') as f:
        data = f.read()

    train_loader = DataLoader(data, batch_size=32, seq_len=2)
    train_ids, val_ids = train_loader.create_batches()

    # Define model hyperparameters
    num_layers = 6
    num_heads = 8
    d_model = 512
    d_ff = 2048
    vocab_size = train_loader.tokenizer.n_vocab
    print(vocab_size)
    max_seq_len = train_loader.seq_len
    dropout_rate = 0.1

    # Initialize the model
    model = GPT(
        num_layers=num_layers,
        num_heads=num_heads,
        d_model=d_model,
        d_ff=d_ff,
        vocab_size=vocab_size,
        max_seq_len=max_seq_len,
        dropout_rate=dropout_rate
    )

    # Set training parameters
    epochs = 10
    learning_rate = 1e-4

    # Train the model
    train_model(rng, train_ids, model, epochs, learning_rate)

    # Optionally, you can add validation here
    # val_loss = evaluate_model(model, val_ids)
    # print(f'Validation loss: {val_loss}')