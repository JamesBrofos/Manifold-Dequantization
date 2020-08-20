import jax.numpy as jnp


def forward(x: jnp.ndarray, shift: jnp.ndarray, scale: jnp.ndarray) -> jnp.ndarray:
    """Compute the affine linear transformation of the input given a shift and
    scale.

    Args:
        x: The input on which to apply the affine transformation.
        shift: The shift to apply to the input.
        scale: The scaling factor to apply to the input.

    Returns:
        out: The input that has undergone an affine transformation according to
            the specified shift and scale.
    """
    return x * scale + shift

def inverse(y: jnp.ndarray, shift: jnp.ndarray, scale: jnp.ndarray) -> jnp.ndarray:
    """Applies the inverse affine transformation given the shift and scale.

    Args:
        y: The input on which to undo an affine transformation.
        shift: The shift to undo to the input.
        scale: The scale to undo to the input

    Returns:
        out: The input under an the inverse of an affine transformation.

    """
    return (y - shift) / scale

def forward_log_det_jacobian(scale: jnp.ndarray) -> jnp.ndarray:
    """Computes the Jacobian correction for a random variable that is transformed
    in the forward direction by an affine transformation. The Jacobian
    correction is just the absolute value of the log-determinant of the scale.

    Args:
        scale: The scale factor of the affine transformation.

    Returns:
        out: The absolute value of the log-determinant of the scale, which is the
            Jacobian correction for a random variable transformed by an affine
            transformation in the forward direction.

    """
    abslogdet = jnp.sum(jnp.log(jnp.abs(scale)), axis=-1)
    return abslogdet
