import time
from collections import defaultdict

import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState
from flax import linen as nn
import numpy as np

from jaxgpt.utils import CfgNode as CN

class Trainer:
    @staticmethod
    def get_default_config() -> CN:
        C = CN()
        C.num_workers = 4
        C.max_iters = None
        C.batch_size = 64
        C.block_size = 128
        C.learning_rate = 3e-4
        C.betas = (0.9, 0.95)
        C.weight_decay = 0.1
        C.grad_norm_clip = 1.0
        return C

    def __init__(self, config, model, train_dataset):
        self.config = config
        self.model = model
        self.optim = None
        self.train_dataset = train_dataset
        self.callbacks = defaultdict(list)

        self.device = jax.default_device
        print('running on device', self.device)
        self.state = None

        self.iter_num = 0
        self.iter_time = .0
        self.iter_dt = .0

    def add_callback(self, onevent: str, callback):
        self.callbacks[onevent].append(callback)

    def set_callback(self, onevent: str, callback):
        self.callbacks[onevent] = [callback]

    def trigger_callbacks(self, onevent: str):
        for callback in self.callbacks.get(onevent, []):
            callback(self)

    def create_train_state(self, rng, model, config):
        # Debug print
        print("Train dataset shape:", self.train_dataset[0][0].shape)
        print("Train dataset type:", self.train_dataset[0][0].dtype)
        
        # Use the first batch of actual training data for initialization
        init_data = jax.tree_map(lambda x: x[:config.batch_size], self.train_dataset[0])
        
        # Debug print
        print("Init data shape:", init_data[0].shape)
        print("Init data type:", init_data[0].dtype)
        
        params = model.init(rng, init_data[0], train=True)
        tx = optax.chain(
            optax.clip_by_global_norm(config.grad_norm_clip),
            optax.adamw(config.learning_rate, b1=config.betas[0], b2=config.betas[1], weight_decay=config.weight_decay),
        )
        return TrainState.create(apply_fn=model.apply, params=params, tx=tx)

    def run(self):
        config = self.config
        rng = jax.random.PRNGKey(config.seed)
        self.state = self.create_train_state(rng, self.model, config)

        def data_generator(dataset, batch_size):
            while True:
                np.random.shuffle(dataset)
                for i in range(0, len(dataset), batch_size):
                    batch = dataset[i:i+batch_size]
                    yield jax.device_put(batch)

        train_loader = data_generator(self.train_dataset, config.batch_size)

        model = self.model 
        self.iter_num = 0
        self.iter_time = time.time()

        @jax.jit
        def loss_fn(params, x, y):
            logits, _ = model.apply({'params': params}, x, train=True)
            loss = jnp.mean(optax.softmax_cross_entropy(logits=logits, labels=jax.nn.one_hot(y, logits.shape[-1])))
            return loss

        grad_fn = jax.jit(jax.value_and_grad(loss_fn))

        while True:
            batch = next(train_loader)
            x, y = batch

            # Add debugging information here
            print("Input shape:", x.shape)
            print("Input data sample:", x[:5])
            print("Target shape:", y.shape)
            print("Target data sample:", y[:5])

            loss, grads = grad_fn(self.state.params, x, y)
            self.state = self.state.apply_gradients(grads=grads)
            self.trigger_callbakcs('on_batch_end')
            self.iter_num += 1
            tnow = time.time()
            self.iter_dt = tnow - self.iter_time
            self.iter_time = tnow

            if config['max_iters'] is not None and self.iter_num >= config['max_iters']:
                break

