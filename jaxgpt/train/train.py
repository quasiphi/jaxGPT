import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

import random
import os
import numpy as np

os.environ["XLA_FLAGS"] = (
    "--xla_backend_optimization_level=3 "
    "--xla_gpu_enable_triton_softmax_fusion=true "
    "--xla_gpu_triton_gemm_any=True "
    "--xla_gpu_enable_custom_fusions=true "
)


SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
np.set_printoptions(precision=4, suppress=True, floatmode="fixed")

import sys
import time
import pickle

import jax
import jax.numpy as jnp
import optax
import chex
import orbax.checkpoint as ocp
import tiktoken


from tqdm import tqdm
from pathlib import Path
from pprint import pprint
from functools import partial

from flax import serialization

from jaxgpt.models.gpt import GPT, GPTConfig
from jaxgpt.utils.common import colored


root_key = jax.random.key(SEED)


out_dir = "out"
eval_interval = 2000
log_interval = 1
eval_iters = 100
eval_only = False
always_save_checkpoint = True
init_from = "scratch"  # or 'scratch' or 'resume' or 'gpt2' for pretrained

# data
dataset = "shakespeare"
batch_size = 8
block_size = 128

# model
n_layer = 6
n_head = 8
n_embd = 384
dropout = 0.0

# optim
learning_rate = 1e-4
max_iters = 600000
weight_decay = 1e-2
beta1 = 0.9
beta2 = 0.95

decay_lr = True
warmup_iters = 2000
lr_decay_iters = 600000  # set it ~ max_iters
min_lr = 6e-5

# i think these are unused
device = "cpu"
dtype = "bfloat16"

max_new_tokens = 100
temperature = 0.8
top_k = 200

config_keys = [
    k
    for k, v in globals().items()
    if not k.startswith("_") and isinstance(v, (int, float, bool, str))
]
print(config_keys)

# check if the config has been supplied
for arg in sys.argv[1:]:
    if "=" not in arg:
        continue
    kv = arg.split("=")
    assert len(kv) == 2, f"expecting each arg to be of form --arg=value, got {arg}"
    k, v = kv
    assert k[:2] == "--"
    k = k[2:]
    if k == "config":
        config_file = Path(v)
        if not config_file.exists():
            print(f"Config file {v} not found")
            continue
        config = {}
        with open(config_file, "r") as f:
            for line in f:
                key, value = line.strip().split("=")
                key = key.strip()
                value = value.strip()
                if key in config_keys:
                    config[key] = eval(value)
        globals().update(config)

config = {k: globals()[k] for k in config_keys}
pprint(config)

checkpoint_path = Path(out_dir) / "checkpoints"
checkpoint_manager = ocp.CheckpointManager(
    checkpoint_path,
    options=ocp.CheckpointManagerOptions(max_to_keep=2, save_interval_steps=1),
    item_names=("params", "opt_state"),
)

data_dir = os.path.join("../data/data", dataset)
train_data = np.memmap(os.path.join(data_dir, "train.bin"), dtype=np.uint16, mode="r")
test_data = np.memmap(os.path.join(data_dir, "test.bin"), dtype=np.uint16, mode="r")


def get_batch(split):
    data = train_data if split == "train" else test_data
    ix = np.random.randint(len(data) - block_size, size=(batch_size,))
    x = np.stack([data[i : i + block_size].astype(np.int32) for i in ix])
    y = np.stack([data[i + 1 : i + 1 + block_size].astype(np.int32) for i in ix])
    return jnp.array(x), jnp.array(y)


iter_num = 0
best_val_loss = 1e9

meta_path = os.path.join(data_dir, "meta.pkl")
if os.path.exists(meta_path):
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)
    vocab_size = meta["vocab_size"]
    print(f'vocab_size = {colored(vocab_size, "blue")} (from {meta_path})')
else:
    vocab_size = 50267
    print(
        f'vocab_size not found in {meta_path}, using GPT-2 default of {colored(str(vocab_size), "blue")}'
    )

# model init, config is globalized
model_args = dict(
    n_layer=n_layer,
    n_head=n_head,
    n_embd=n_embd,
    block_size=block_size,
    dropout=dropout,
    vocab_size=vocab_size,
)
if init_from == "scratch":
    print(colored("*** initializing model from scratch ***", "green"))
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    state = model.create_train_state(**config)
    params = state.params
elif init_from == "resume":
    print(colored("*** resuming training from last checkpoint ***", "green"))
    latest_step = checkpoint_manager.latest_step()
    assert latest_step is not None, colored("no checkpoints found", "red")
    checkpoint = checkpoint_manager.restore(latest_step)
    checkpoint_model_args = checkpoint["model_args"]
    for k, v in model_args.items():
        assert checkpoint_model_args[k] == v
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    empty_state = jax.eval_shape(lambda: model.create_train_state(**config))
    state = serialization.from_state_dict(empty_state, checkpoint["state"])
    iter_num = checkpoint["iter_num"] + 1
    best_val_loss = checkpoint["best_val_loss"]
elif init_from.startswith("gpt2"):
    print(
        colored(
            f'*** initializing from pretrained GPT-2 model "{init_from}" ***', "green"
        )
    )
    override_args = dict(dropout=dropout)
    model, params = GPT.from_pretrained(init_from, override_args=override_args)
    state = model.create_train_state(**config, params=params)
    model_args["n_layer"] = model.config.n_layer
    model_args["n_head"] = model.config.n_head
    model_args["n_embd"] = model.config.n_embd
else:
    raise RuntimeError(f'unknown init_from value "{init_from}"')


gradient_clipping = 1.0
root_key, model_init_param_key = jax.random.split(root_key)

model_params = model.init(
    model_init_param_key,
    jnp.ones((batch_size, block_size), dtype=jnp.int32),
    train=False,
)

# TODO: Leave sharding for later
# if number_of_devices > 1:
#     # ? Place the parameters on every device to reduce communication
#     model_params = jax.device_put(
#         model_params, sharding.replicate(axis=0, keepdims=True)
#     )

# TODO: Choose a lr schedule based on config
# lr_schedule = optax.warmup_cosine_decay_schedule(
#     init_value=1e-7,
#     peak_value=learning_rate,
#     warmup_steps=warmup_iters,
#     decay_steps=lr_decay_iters,
#     end_value=min_lr,
# )
lr_schedule = learning_rate

optimizer = optax.adamw(
    learning_rate=lr_schedule,
    b1=beta1,
    b2=beta2,
    weight_decay=weight_decay,
)
optimizer = optax.chain(optax.clip_by_global_norm(gradient_clipping), optimizer)
optimizer_state = optimizer.init(model_params)

param_count = sum(x.size for x in jax.tree.leaves(model_params))
print(colored(f"Model Parameters: {param_count}", "blue"))


@chex.dataclass
class Model_State:
    model_params: chex.Array
    opt_state: chex.Array


model_state = Model_State(model_params=model_params, opt_state=optimizer_state)


# @jax.jit
def train_step(rnd_key, model_state: Model_State, batch):
    def loss_fn(params):
        inputs, targets = batch

        logits = model.apply(params, inputs, train=True, rngs=rnd_key)

        loss = optax.softmax_cross_entropy_with_integer_labels(logits, targets).mean()

        return loss

    params = model_state.model_params

    loss, grads = jax.value_and_grad(loss_fn)(params)
    updates, opt_state = optimizer.update(grads, model_state.opt_state, params)
    params = optax.apply_updates(params, updates)

    model_state.model_params = params
    model_state.opt_state = opt_state

    return loss, model_state


def estimate_loss(model_state: Model_State):
    out = {}

    for split in ["train", "test"]:
        losses = np.zeros(eval_iters)

        progress_bar = tqdm(total=eval_iters)
        progress_bar.set_description(f"{split.capitalize()} Loss: {0.0:.3f}")

        for k in range(eval_iters):
            batch = get_batch(split)
            inputs, labels = batch

            logits = model.apply(
                model_state.model_params, inputs, train=True, rngs=jax.random.key(0)
            )

            loss = optax.softmax_cross_entropy_with_integer_labels(
                logits, labels
            ).mean()
            losses[k] = float(loss)

            progress_bar.update(1)
            progress_bar.set_description(
                f"{split.capitalize()} Loss: {float(loss):.3f}"
            )

        out[split] = losses.mean()
        progress_bar.close()

    return out


@jax.jit
def _sample(params, key, tokens) -> jax.Array:
    return model.generate(
        key,
        params,
        tokens,
        max_new_tokens=max_new_tokens,
        top_k=top_k,
        temperature=temperature,
    )


tokenizer = tiktoken.get_encoding("gpt2")


def sample(params, key, tokens) -> str:
    tokens = _sample(params, key, tokens)
    return tokenizer.decode(tokens[0])


test_batch = get_batch("test")
train_batch = get_batch("train")
st = time.time()
while True:
    if iter_num % eval_interval == 0:
        print("Validation", "- " * 20)

        root_key, sample_key = jax.random.split(root_key)

        sample_str = colored(
            sample(state.params, sample_key, tokens=test_batch[0][0:1, :5]), "yellow"
        )
        print(f'Generation Sample: "{sample_str}"')

        losses = estimate_loss(model_state)

        print(
            f"step {iter_num}: train loss {losses['train']:.4f}, test loss {losses['test']:.4f}"
        )

        if iter_num > 0:
            print(f"saving checkpoint to {out_dir}")

            checkpoint = {
                "state": state,
                "model_args": model_args,
                "iter_num": iter_num,
                "val_loss": losses["test"],
                "config": config,
            }

            # * Arrays are sharded by default
            gathered_opt_state = jax.tree.map(
                lambda x: jax.device_get(x), model_state.opt_state
            )
            gathered_model_params = jax.tree.map(
                lambda x: jax.device_get(x), model_state.model_params
            )
            checkpoint_manager.save(
                iter_num,
                args=ocp.args.Composite(
                    params=ocp.args.StandardSave(gathered_model_params),
                    opt_state=ocp.args.StandardSave(gathered_opt_state),
                ),
            )
            checkpoint_manager.wait_until_finished()

        print("Training", "- " * 20)

    if iter_num == 0 and eval_only:
        break

    root_key, train_key = jax.random.split(root_key)
    loss, model_state = train_step(train_key, model_state, get_batch("train"))

    # * NaN Check
    if any(jnp.isnan(p).any() for p in jax.tree.leaves(state.params)):
        raise RuntimeError("NaN params")

    et = time.time()
    dt = et - st
    if iter_num % log_interval == 0:
        print(f"iter {iter_num}: loss {loss:.4f}, time {dt*1000:.2f}ms")

    iter_num += 1

    if iter_num >= max_iters:
        break
