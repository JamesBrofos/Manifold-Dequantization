from typing import Tuple

import jax.numpy as jnp
import jax.scipy.special as jspsp
from jax import random
from jax import vmap


def uqr(A: jnp.ndarray) -> Tuple[jnp.ndarray]:
    """This is the implementation of the unique QR decomposition as proposed in
    [1], modified for JAX compatibility.

    [1] https://github.com/numpy/numpy/issues/15628

    Args:
        A: Matrix for which to compute the unique QR decomposition.

    Returns:
        Q: Orthogonal matrix factor.
        R: Upper triangular matrix with positive elements on the diagonal.

    """
    Q, R = jnp.linalg.qr(A)
    signs = 2 * (jnp.diag(R) >= 0) - 1
    Q = Q * signs[:, jnp.newaxis]
    R = R * signs[..., jnp.newaxis]
    return Q, R

def sample(rng: random.PRNGKey, num_samples: int, num_dims: int) -> jnp.ndarray:
    """Draw samples from the uniform distribution on the O(n).

    Args:
        rng: Pseudo-random number generator seed.
        num_samples: Number of uniform samples to generate.

    Returns:
        q: A matrix of uniformly distributed elements on O(n).

    """
    xo = random.normal(rng, [num_samples, num_dims, num_dims])
    q, r = vmap(uqr)(xo)
    return q

def logpdf(xon: float) -> float:
    """Computes the log-density of the uniform distribution on O(n). This is the
    negative logarithm of the volume of O(n), which is twice the volume of
    SO(n).

    Args:
        xon: The observations on O(n) at which to compute the uniform density.

    Returns:
        logpdf: The log-density of the uniform distribution on O(n).

    """
    num_dims = xon.shape[-1]
    logvol = (
        jnp.log(2.) + (num_dims-1) * jnp.log(2.) +
        (num_dims - 1)*(num_dims + 2) / 4. * jnp.log(jnp.pi) -
        jspsp.gammaln(jnp.arange(2, num_dims + 1) / 2).sum())
    logpdf = -logvol
    return logpdf * jnp.ones((len(xon), ))
