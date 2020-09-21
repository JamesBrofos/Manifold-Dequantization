from typing import Callable, Sequence

import jax.numpy as jnp
import jax.scipy.special as jspsp

import prax.distributions as pd

from ambient import ambient_flow_log_prob
from dequantization import dequantize


def log_importance_sample_density(obs: jnp.ndarray, num_is: int,
                                  deq_params: Sequence[jnp.ndarray],
                                  deq_fn: Callable, bij_params:
                                  Sequence[jnp.ndarray], bij_fns: Callable, rng:
                                  jnp.ndarray) -> jnp.ndarray:
    """Use importance samples to estimate the log-density on the sphere.

    Args:
        obs: Points on the sphere at which to compute the log-density, estimated
            by importance sampling.
        num_is: Number of importance samples.
        deq_params: Parameters of the mean and scale functions used in the
            log-normal dequantizer.
        deq_fn: Function that computes the mean and scale given its
            parameterization and input.
        bij_params: List of arrays parameterizing the RealNVP bijectors.
        bij_fns: List of functions that compute the shift and scale of the RealNVP
            affine transformation.
        rng: Pseudo-random number generator seed.

    Returns:
        log_isdens: The estimated log-density on the manifold, computed via
            importance sampling.

    """
    x, ln, qcond = dequantize(rng, deq_params, deq_fn, obs, num_is)
    pamb = ambient_flow_log_prob(bij_params, bij_fns, x)
    log_isdens = jspsp.logsumexp(pamb - qcond, axis=0) - jnp.log(num_is)
    return log_isdens

def mixture_density(obs: jnp.ndarray, kappa: float, muspha: jnp.ndarray,
                    musphb: jnp.ndarray) -> jnp.ndarray:
    """Compute the density of the power spherical mixture distribution.

    Args:
        obs: Points on the sphere at which to compute the mixture density.
        kappa: Concentration parameter of both modes
        muspha: Mean parameter of first mode.
        musphb: Mean parameter of second mode.

    Returns:
        out: The density of the power spherical mixture distribution.

    """
    logdens = log_mixture_density(obs, kappa, muspha, musphb)
    return jnp.exp(logdens)

def log_mixture_density(obs: jnp.ndarray, kappa: float, muspha: jnp.ndarray,
                        musphb: jnp.ndarray) -> jnp.ndarray:
    """Compute the log-density of the power spherical mixture distribution.

    Args:
        obs: Points on the sphere at which to compute the mixture density.
        kappa: Concentration parameter of both modes
        muspha: Mean parameter of first mode.
        musphb: Mean parameter of second mode.

    Returns:
        out: The log-density of the power spherical mixture distribution.

    """
    densa = pd.sphere.powsphlogdensity(obs, kappa, muspha)
    densb = pd.sphere.powsphlogdensity(obs, kappa, musphb)
    return jspsp.logsumexp(jnp.array([densa, densb]), axis=0) - jnp.log(2.)
