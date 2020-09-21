from functools import partial
from typing import Callable

import jax.numpy as jnp
from jax import lax, random

from coordinates import ang2euclid, euclid2ang


def correlated_torus_density(theta: jnp.ndarray) -> jnp.ndarray:
    """Unnormalized correlated density on the torus.

    Args:
        theta: An array containing the two angular coordinates of the torus.

    Returns:
        out: The unnormalized density at the provided angular coordinates.

    """
    thetaa, thetab = theta[..., 0], theta[..., 1]
    return jnp.squeeze(jnp.exp(jnp.cos(thetaa + thetab - 1.94)))

def unimodal_torus_density(theta: jnp.ndarray) -> jnp.ndarray:
    """Unnormalized multimodal density on the torus.

    Args:
        theta: An array containing the two angular coordinates of the torus.

    Returns:
        out: The unnormalized density at the provided angular coordinates.

    """
    p = lambda thetaa, thetab, phia, phib: jnp.exp(jnp.cos(thetaa - phia) + jnp.cos(thetab - phib))
    thetaa, thetab = theta[..., 0], theta[..., 1]
    return jnp.squeeze(p(thetaa, thetab, 4.18, 5.96))

def multimodal_torus_density(theta: jnp.ndarray) -> jnp.ndarray:
    """Unnormalized multimodal density on the torus.

    Args:
        theta: An array containing the two angular coordinates of the torus.

    Returns:
        out: The unnormalized density at the provided angular coordinates.

    """
    p = lambda thetaa, thetab, phia, phib: jnp.exp(jnp.cos(thetaa - phia) + jnp.cos(thetab - phib))
    thetaa, thetab = theta[..., 0], theta[..., 1]
    uprob = (p(thetaa, thetab, 0.21, 2.85) +
             p(thetaa, thetab, 1.89, 6.18) +
             p(thetaa, thetab, 3.77, 1.56)) / 3.
    return jnp.squeeze(uprob)

def embedded_torus_density(xtor: jnp.ndarray, torus_density: Callable) -> jnp.ndarray:
    """Embed the torus density in an ambient Euclidean space. Points on the torus
    are considered as points on two circles.

    Args:
        xtor: Points on the surface of two circles; these must be represented as
            angles for the torus density function.
        torus_density: The density on the torus from which to generate samples.

    Returns:
        out: The density function at the requested locations on the torus.

    """
    # TODO: I don't think any Jacobian correction is required because the
    # Jacobian determinant will be one.
    return torus_density(euclid2ang(xtor))

def rejection_sampling(rng: jnp.ndarray, num_samples: int, torus_density: Callable) -> jnp.ndarray:
    """Sample from the torus via a uniform distribution on the angular coordinates.
    If we have access to an unnormalized density `f(x)`, we require a constant
    `M` such that `M * q(x) > f(x)` at every point. Since we are using a
    uniform proposal distribution, `q(x) = 1 / (2.*pi)^2` (squared because of
    two independent components).

    Args:
        rng: Pseudo-random number generator seed.
        num_samples: Number of samples to attempt draw. The number of samples
            returned may be smaller since some samples will be rejected.
        torus_density: The density on the torus from which to generate samples.

    Returns:
        samples: Samples from the distribution on the torus.

    """
    # Precompute certain quantities used in rejection sampling.
    M = jnp.square(2.*jnp.pi) * 3.
    denom = M / jnp.square(2.*jnp.pi)

    def cond(val):
        """Check whether or not the proposal has been accepted.

        Args:
            val: A tuple containing the previous proposal, whether or not it was
                accepted (it wasn't), and the current iteration of the rejection
                sampling loop.

        Returns:
            out: A boolean for whether or not to continue sampling. If the sample
                was rejected, try again. Otherwise, return the accepted sample.

        """
        _, isacc, _ = val
        return jnp.logical_not(isacc)

    def sample_once(sample_iter, val):
        """Attempt to draw a single sample. If the sample is rejected, this function is
        called in a while loop until a sample is accepted.

        Args:
            sample_iter: Sampling iteration counter.
            val: A tuple containing the previous proposal, whether or not it was
                accepted (it wasn't), and the current iteration of the rejection
                sampling loop.

        Returns:
            out: A tuple containing the proposal, whether or not it was accepted, and
                the next iteration counter.

        """
        _, _, it = val
        rng_sample_once = random.fold_in(random.fold_in(rng, it), sample_iter)
        rng_prop, rng_acc = random.split(rng_sample_once, 2)
        theta = 2.*jnp.pi*random.uniform(rng_prop, [2]) - jnp.pi
        numer = torus_density(theta)
        alpha = numer / denom
        unif = random.uniform(rng_acc)
        isacc = unif < alpha
        return theta, isacc, it + 1

    def sample(_, it):
        """Samples in a loop so that the total number of samples has a predictable
        shape. The first argument is ignored. The second argument is the
        sampling iteration.

        Args:
            _: Ignored argument for `lax.scan` compatibility.
            it: Sampling iteration number.

        Returns:
            _, theta: A tuple containing the ignored input quantity and the
                accepted sample.

        """
        state = lax.while_loop(cond, partial(sample_once, it), (jnp.zeros(2), False, 0))
        theta, isacc, num_iters = state
        return _, theta
    _, theta = lax.scan(sample, None, jnp.arange(num_samples))
    return theta
