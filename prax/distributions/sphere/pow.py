from typing import Sequence

import jax.numpy as jnp
import jax.scipy.special as jspsp
from jax import ops, random


def rvs(rng: jnp.ndarray, kappa: float, mu: jnp.ndarray, shape: Sequence[int]) -> jnp.ndarray:
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

def logpdf(x: jnp.ndarray, kappa: float, mu: jnp.ndarray) -> jnp.ndarray:
    """Log-density function of the power spherical distribution.

    Args:
        x: The set of points at which to evaluate the power spherical density.
        kappa: Concentration parameter.
        mu: Mean direction on the sphere. The dimensionality of the sphere is
            determined from this paramter.

    Returns:
        out: The log-density of the power spherical distribution with the
            specified concentration and mean parameter at the desired points.

    """
    d = mu.size
    alpha = (d - 1.) / 2. + kappa
    beta = (d - 1.) / 2.
    lognormalizer = (
        (alpha + beta) * jnp.log(2.) + beta * jnp.log(jnp.pi) +
        jspsp.gammaln(alpha) - jspsp.gammaln(alpha + beta))
    unlogprob = kappa * jnp.log(1. + x.dot(mu))
    return unlogprob - lognormalizer

def mean(kappa: float, mu: jnp.ndarray) -> jnp.ndarray:
    """Compute the expectation of the power spherical distribution.

    Args:
        kappa: Concentration parameter.
        mu: Mean direction on the sphere. The dimensionality of the sphere is
            determined from this paramter.

    Returns:
        out: The expectation of the power spherical distribution.

    """
    d = mu.size
    alpha = (d - 1.) / 2. + kappa
    beta = (d - 1.) / 2.
    return mu * (alpha - beta) / (alpha + beta)

def var(kappa: float, mu: jnp.ndarray) -> jnp.ndarray:
    """Compute the variance of the power spherical distribution.

    Args:
        kappa: Concentration parameter.
        mu: Mean direction on the sphere. The dimensionality of the sphere is
            determined from this paramter.

    Returns:
        out: The variance of the power spherical distribution.

    """
    d = mu.size
    alpha = (d - 1.) / 2. + kappa
    beta = (d - 1.) / 2.
    return (
        2*alpha / ((alpha+beta)**2 * (alpha + beta + 1.)) *
        ((beta - alpha) * jnp.outer(mu, mu) + (alpha + beta) * jnp.eye(d)))
