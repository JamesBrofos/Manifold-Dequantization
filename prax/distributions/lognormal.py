from typing import Sequence

import jax.numpy as jnp
import jax.scipy.stats as jspst
from jax import random


def logpdf(x: jnp.ndarray, mu:jnp.ndarray, sigma: jnp.ndarray) -> jnp.ndarray:
    """Compute the log-density of a log-normal ranomd variable.

    Args:
        x: Array of values at which to compute the log-density of the log-normal
            random variable.
        mu: The log-normal random variable is the exponential of a normal random
            variable with this mean.
        sigma: The log-normal random variable is the exponential of a normal
            random variable with this standard deviation.

    Returns:
        out: The value of the log-density of a log-normal random variable at the
            specified locations and with the mean and standard deviation
            parameters.

    """
    log_x = jnp.log(x)
    return jspst.norm.logpdf(log_x, mu, sigma) - log_x

def sample(rng: jnp.ndarray, mu: jnp.ndarray, sigma: jnp.ndarray, shape:
           Sequence[int]) -> jnp.ndarray:
    """Compute samples from a log-normal distributon via reparameterization by
    expressing the log-normal as the exponentiation of an affine transformation
    of a standard normal.

    Args:
        rng: Pseudo-random number generator seed.
        mu: The log-normal random variable is the exponential of a normal random
            variable with this mean.
        sigma: The log-normal random variable is the exponential of a normal
            random variable with this standard deviation.
        shape: The shape of the random sample.

    Returns:
        out: Samples from a log-normal distribution with the desired mean and
            standard deviation parameters of the exponentiated normal
            distribution.

    """
    z = random.normal(rng, shape)
    x = sigma * z + mu
    return jnp.exp(x)
