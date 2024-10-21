import jax.numpy as jnp


def generate_text(model, params, prompt, tokenizer, max_tokens=50):
    input_ids = tokenizer(prompt, return_tensors='jax')[0]
    for _ in range(max_tokens):
        logits = model.apply(params, input_ids)
        next_id = jnp.argmax(logits[:, -1, :], axis=-1)
        input_ids = jnp.concatenate([input_ids, next_id[None, :]], axis=1)
    return tokenizer.decode(input_ids[0].tolist())