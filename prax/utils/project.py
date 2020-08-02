import jax.numpy as jnp


def project(x: jnp.ndarray) -> jnp.array:
    """Project a tensor to the sphere by dividing by the norm along the last
    dimension.

    Args:
        x: The tensor to project to the sphere.

    Returns:
        The projection of `x` to the sphere.

    """
    return x / jnp.linalg.norm(x, axis=-1)[..., jnp.newaxis]

