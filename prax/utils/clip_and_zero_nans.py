import jax.numpy as jnp


def clip_and_zero_nans(g: jnp.ndarray, clip_value: float) -> jnp.ndarray:
    """Clip the input to within a certain range and remove the NaNs in a matrix by
    replacing them with zeros. This function is useful for ensuring stability
    in gradient descent.

    Args:
        g: Matrix whose elements should be clipped to within a certain range
            and whose NaN elements should be replaced by zeros.
        clip_value: Value at which to truncate the elements.

    Returns:
        out: The input matrix but with clipped values and NaN elements replaced
            by zeros.

    """
    g = jnp.where(jnp.isnan(g), jnp.zeros_like(g), g)
    g = jnp.clip(g, -clip_value, clip_value)
    return g

