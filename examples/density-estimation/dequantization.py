from typing import Callable, Sequence, Tuple

import jax.numpy as jnp
from jax import nn

import prax.distributions as pd

from ambient import ambient_flow_log_prob


def dequantize(rng: jnp.ndarray, deq_params: Sequence[jnp.ndarray],
               deq_fn: Callable, y: jnp.ndarray, num_samples: int) -> jnp.ndarray:
    """Dequantize the observations using a log-normal multiplicative
    dequantizer.

    Args:
        rng: Pseudo-random number generator seed.
        deq_params: Parameters of the mean and scale functions used in
            the log-normal dequantizer.
        deq_fn: Function that computes the mean and scale given its
            parameterization and input.
        y: Observations on the sphere to dequantize.

    Returns:
        x, qcond: A tuple containing observations that are dequantized according
            to multiplication by a log-normal random variable and the
            log-density of the conditional dequantizing distribution.

    """
    mu, sigma = deq_fn(deq_params, y)
    mu = nn.softplus(mu)
    ln = pd.lognormal.sample(rng, mu, sigma, [num_samples] + list(mu.shape))
    x = ln * y
    qcond = jnp.squeeze(pd.lognormal.logpdf(ln, mu, sigma), -1) - 2.*jnp.log(ln[..., 0])
    return x, qcond

def negative_elbo_per_example(deq_params: Sequence[jnp.ndarray],
                              bij_params: Sequence[jnp.ndarray],
                              deq_fn: Callable, bij_fns:
                              Sequence[Callable], rng: jnp.ndarray, y:
                              jnp.ndarray, num_samples: int) -> jnp.ndarray:
    """Compute the negative evidence lower bound of the dequantizing distribution.
    This is the loss function for learning parameters of the dequantizing
    distribution.

    Args:
        deq_params: Parameters of the mean and scale functions used in
            the log-normal dequantizer.
        bij_params: List of arrays parameterizing the RealNVP bijectors.
        deq_params: Parameters of the mean and scale functions used in the
            log-normal dequantizer.
        bij_fns: List of functions that compute the shift and scale of the RealNVP
            affine transformation.
        rng: Pseudo-random number generator seed.
        y: Observations on the sphere to dequantize.
        num_samples: Number of dequantization samples to compute.

    Returns:
        nelbo: The negative evidence lower bound for each example.

    """
    x, qcond = dequantize(rng, deq_params, deq_fn, y,
                          num_samples)
    pamb = ambient_flow_log_prob(bij_params, bij_fns, x)
    elbo = jnp.mean(pamb - qcond, 0)
    nelbo = -elbo
    return nelbo

def negative_elbo(deq_params: Sequence[jnp.ndarray], bij_params:
                  Sequence[jnp.ndarray], deq_fn: Callable, bij_fns:
                  Sequence[Callable], rng: jnp.ndarray, y: jnp.ndarray,
                  num_samples: int) -> float:
    """Compute the evidence lower bound averaged over all examples.

    Args:
        deq_params: Parameters of the mean and scale functions used in the
            log-normal dequantizer.
        bij_params: List of arrays parameterizing the RealNVP bijectors.
        deq_fn: Function that computes the mean and scale given its
            parameterization and input.
        bij_fns: List of functions that compute the shift and scale of the RealNVP
            affine transformation.
        rng: Pseudo-random number generator seed.
        y: Observations on the sphere to dequantize.
        num_samples: Number of dequantization samples to compute.

    Returns:
        out: The negative evidence lower bound averaged over all examples.

    """
    return negative_elbo_per_example(deq_params, bij_params,
                                     deq_fn, bij_fns, rng, y,
                                     num_samples).mean()
