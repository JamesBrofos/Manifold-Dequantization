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
    Id = jnp.eye(3)
    Ra = jnp.diag(jnp.array([-1., -1., 1]))
    Rb = jnp.diag(jnp.array([-1., 1., -1]))
    scale = 0.5
    lp = 0.
    lp += jnp.exp(-0.5 * jnp.square(x - Id).sum((-1, -2)) / jnp.square(scale))
    lp += jnp.exp(-0.5 * jnp.square(x - Ra).sum((-1, -2)) / jnp.square(scale))
    lp += jnp.exp(-0.5 * jnp.square(x - Rb).sum((-1, -2)) / jnp.square(scale))
    return jnp.log(lp)
