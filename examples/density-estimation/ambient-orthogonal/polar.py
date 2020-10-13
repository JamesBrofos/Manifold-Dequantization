from typing import Tuple

import jax.numpy as jnp


transp = lambda x: jnp.swapaxes(x, -1, -2)

def polar(A: jnp.ndarray) -> Tuple[jnp.ndarray]:
    """Compute the polar decomposition of a matrix.

    Args:
        A: Matrix whose polar decomposition should be computed.

    Returns:
        o: Orthogonal matrix component of the decomposition.
        l: Cholesky decomposition of the positive definite component of the
            decomposition.

    """
    u, s, vT = jnp.linalg.svd(A, full_matrices=False)
    v = transp(vT)
    # Positive definite component of the decomposition.
    p = v@(s[..., jnp.newaxis] * vT)
    o = u@vT
    chol = jnp.linalg.cholesky(p)
    return o, chol

def vecpolar(a):
    n = jnp.sqrt(a.shape[-1]).astype(jnp.int32)
    A = a.reshape((n, n))
    o, chol = polar(A)
    return jnp.hstack((o, chol)).ravel()
