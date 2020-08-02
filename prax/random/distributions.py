from typing import Sequence

import jax.numpy as jnp
from jax import ops, random

from ..utils import project


def haarsph(rng: jnp.ndarray, shape: Sequence[int]) -> jnp.ndarray:
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
    return project(x)

def powsph(rng: jnp.ndarray, kappa: float, mu: jnp.ndarray, shape:
           Sequence[int]) -> jnp.ndarray:
    """Power spherical distribution. See [1] for details.

    [1] https://arxiv.org/pdf/2006.04437.pdf

    Args:
        rng: A PRNGKey used as the random key.
        kappa: Concentration parameter.
        mu: Mean direction on the sphere. The dimensionality of the sphere is
            determined from this paramter.
        shape: A tuple of nonnegative integers representing the result shape.

    Returns:
        Samples from the power spherical distribution.
    """
    dims = mu.size
    sphere_rng, beta_rng = random.split(rng)
    sphereshape = shape + [dims-1]
    unif = haarsph(sphere_rng, sphereshape)
    alpha = (dims - 1.) / 2. + kappa
    beta = (dims - 1.) / 2.
    z = random.beta(beta_rng, alpha, beta, shape)
    t = (2.*z - 1.)[..., jnp.newaxis]
    y = jnp.concatenate((t, jnp.sqrt(1 - jnp.square(t)) * unif), axis=-1)
    e = jnp.zeros([dims])
    e = ops.index_update(e, 0, 1.)
    u = e - mu
    u /= jnp.linalg.norm(u)
    Id = jnp.eye(dims)
    H = Id - 2.*jnp.outer(u, u)
    x = jnp.einsum('ij,...j->...i', H, y)
    return x
