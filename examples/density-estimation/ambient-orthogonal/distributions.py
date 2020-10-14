import jax.numpy as jnp


def log_unimodal(x: jnp.ndarray) -> jnp.ndarray:
    """Unimodal distribution on O(n) centered at the identity matrix."""
    num_dims = x.shape[-1]
    Id = jnp.eye(num_dims)
    lp = -0.5 * jnp.square(x - Id).sum(axis=(-1, -2))
    lp = lp / 0.5
    return lp

def log_multimodal(x):
    """Multimodal distribution on O(n) with components centered at the identity
    matrix and the pure reflection.

    """
    num_dims = x.shape[-1]
    Id = jnp.eye(num_dims)
    scale = 0.5
    lp = 0.
    lp += jnp.exp(-0.5 * jnp.square(x - Id).sum((-1, -2)) / jnp.square(scale))
    lp += jnp.exp(-0.5 * jnp.square(x + Id).sum((-1, -2)) / jnp.square(scale))
    return jnp.log(lp)
