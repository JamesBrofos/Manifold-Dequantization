from typing import Sequence

import jax.numpy as jnp
import jax.scipy.special as jspsp
from jax import random


def _project(x: jnp.ndarray) -> jnp.array:
    """Project a tensor to the sphere by dividing by the norm along the last
    dimension.

    Args:
        x: The tensor to project to the sphere.

    Returns:
        The projection of `x` to the sphere.

    """
    return x / jnp.linalg.norm(x, axis=-1)[..., jnp.newaxis]

def rvs(rng: jnp.ndarray, shape: Sequence[int]) -> jnp.ndarray:
    """Sample from the uniform (Haar) measure on the sphere. This is achieved by
    sampling from a rotationally invariant distribution (such as the
    multivariate normal) and projecting to the sphere. The last dimension of
    shape is taken as the dimensionality of the Euclidean embedding space of
    the sphere.

    Args:
        rng: A PRNGKey used as the random key.
        shape: A tuple of nonnegative integers representing the result shape.

    Returns:
        Samples from the uniform distribution on a sphere.

    """
    x = random.normal(rng, shape)
    return _project(x)

def logpdf(x: jnp.ndarray) -> jnp.ndarray:
    """Log-density of the uniform distribution on the sphere. Since the Haar
    distribution is uniform, the density is just the inverse of the surface
    area of the sphere. Therefore, the log-density is the negative logarithm of
    the surface area.

    Args:
        x: The locations on the sphere at which to compute the log-density of
            the Haar distribution on the sphere.

    Returns:
        out: The log-density of the Haar distribution of the sphere.

    """
    n = x.shape[-1]
    halfn = 0.5*n
    logsa = jnp.log(2.) + halfn*jnp.log(jnp.pi) - jspsp.gammaln(halfn)
    return -logsa*jnp.ones(x.shape[:-1])
