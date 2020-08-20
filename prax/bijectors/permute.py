import jax.numpy as jnp


def forward(x: jnp.ndarray, perm: jnp.ndarray) -> jnp.ndarray:
    """Permute the variables along the last dimension according to the input
    permutation.

    Args:
        x: An array of values to permute along the last dimension.
        perm: The permutation of the last dimension.

    Returns:
        out: The permutation of the input along the last dimension by the given
            permutation.

    """
    return x[..., perm]

def inverse(y: jnp.ndarray, perm: jnp.ndarray) -> jnp.ndarray:
    """Undo the permutation along the last dimension. The inverse of a permutation
    can be found by identifying the arguments that sort it from lowest to
    highest.

    Args:
        y: The permuted matrix whose permutation is to be undone.
        perm: The original permutation; not the inverse permutation. The inverse
            permutation is computed by identifying the arguments that sort the
            original permutation.

    Returns:
        out: The unpermuted array.

    """
    return y[..., jnp.argsort(perm)]

def forward_log_det_jacobian() -> float:
    """The permutation operation is volume preserving and hence the determinant of
    its Jacobian is one; the log determinant is zero.

    Returns:
        out: The log-determinant of the Jacobian is zero; returns zero.

    """
    return 0.
