from typing import Callable, Sequence, Tuple

import jax.numpy as jnp
from jax import nn, random

import prax.distributions as pd

from ambient import ambient_flow_log_prob


def dequantize(rng: jnp.ndarray, deq_params: Sequence[jnp.ndarray], deq_fn:
               Callable, axis: jnp.ndarray, angle: jnp.ndarray, num_samples: int) -> jnp.ndarray:
    """Dequantize the observations using a log-normal multiplicative
    dequantizer.

    Args:
        rng: Pseudo-random number generator seed.
        deq_params: Parameters of the mean and scale functions used in
            the log-normal dequantizer.
        deq_fn: Function that computes the mean and scale given its
            parameterization and input.
        axis: Observations on the sphere to dequantize.
        angle: Observations on the circle to dequantize.
        num_samples: Number of dequantization samples to compute.

    Returns:
        x, qcond: A tuple containing observations that are dequantized according
            to multiplication by a log-normal random variable and the
            log-density of the conditional dequantizing distribution.

    """
    rng, rng_axis, rng_angle = random.split(rng, 3)
    y = jnp.hstack((axis, angle))
    mu, sigma = deq_fn(deq_params, y)
    mu = nn.softplus(mu)
    mu_axis, mu_angle = mu[..., [0]], mu[..., [1]]
    sigma_axis, sigma_angle = sigma[..., [0]], sigma[..., [1]]
    ln_axis = pd.lognormal.sample(rng_axis, mu_axis, sigma_axis, [num_samples] + list(mu_axis.shape))
    ln_angle = pd.lognormal.sample(rng_angle, mu_angle, sigma_angle, [num_samples] + list(mu_angle.shape))
    x_axis = ln_axis * axis
    x_angle = ln_angle * angle
    qcond_axis = pd.lognormal.logpdf(ln_axis, mu_axis, sigma_axis) - 2.*jnp.log(ln_axis)
    qcond_angle = pd.lognormal.logpdf(ln_angle, mu_angle, sigma_angle) - jnp.log(ln_angle)
    qcond = jnp.squeeze(qcond_axis + qcond_angle)
    return (x_axis, x_angle), qcond

def negative_elbo_per_example(deq_params: Sequence[jnp.ndarray], bij_params:
                              Sequence[jnp.ndarray], deq_fn: Callable, bij_fns:
                              Sequence[Callable], rng: jnp.ndarray, axis:
                              jnp.ndarray, angle: jnp.ndarray, num_samples:
                              int) -> jnp.ndarray:
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
        axis: Observations on the sphere to dequantize.
        angle: Observations on the circle to dequantize.
        num_samples: Number of dequantization samples to compute.

    Returns:
        nelbo: The negative evidence lower bound for each example.

    """
    (x_axis, x_angle), qcond = dequantize(rng, deq_params, deq_fn, axis, angle, num_samples)
    x = jnp.concatenate((x_axis, x_angle), axis=-1)
    pamb = ambient_flow_log_prob(bij_params, bij_fns, x)
    elbo = jnp.mean(pamb - qcond, 0)
    nelbo = -elbo
    return nelbo

def negative_elbo(deq_params: Sequence[jnp.ndarray], bij_params:
                  Sequence[jnp.ndarray], deq_fn: Callable, bij_fns:
                  Sequence[Callable], rng: jnp.ndarray, axis: jnp.ndarray,
                  angle: jnp.ndarray, num_samples: int) -> float:
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
        axis: Observations on the sphere to dequantize.
        angle: Observations on the circle to dequantize.
        num_samples: Number of dequantization samples to compute.

    Returns:
        out: The negative evidence lower bound averaged over all examples.

    """
    return negative_elbo_per_example(deq_params, bij_params, deq_fn, bij_fns,
                                     rng, axis, angle, num_samples).mean()


if __name__ == '__main__':
    from jax import random
    from network import network_factory

    rng = random.PRNGKey(0)
    rng, rng_deq, rng_axis, rng_angle = random.split(rng, 4)
    rng, rng_bij = random.split(rng, 2)

    deq_params, deq_fn = network_factory(rng_deq, 5, 2)
    axis = pd.sphere.haarsph(rng_axis, [10, 3])
    angle = pd.sphere.haarsph(rng_angle, [10, 2])

    num_samples = 2
    (x_axis, x_angle), qcond = dequantize(rng, deq_params, deq_fn, axis, angle, 5)

    bij_params, bij_fns = [], []
    for i in range(5):
        p, f = network_factory(random.fold_in(rng_bij, i), 2, 3)
        bij_params.append(p)
        bij_fns.append(f)

    elbo = negative_elbo_per_example(deq_params, bij_params, deq_fn, bij_fns, rng, axis, angle, 20)
    print('evidence lower bound:')
    print(elbo)
