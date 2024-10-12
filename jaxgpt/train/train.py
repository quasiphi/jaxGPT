import jax
import jax.numpy as jnp
from flax.training import train_state
from optax import adamw, softmax_cross_entropy

from jaxgpt.utils.checkpoints import save_checkpoint
from jaxgpt.models.gpt import GPT

def create_train_state(rng, model, learning_rate):
    # variables?
    params = model.init(rng, jnp.ones((1, 512)))
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