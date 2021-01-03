from functools import partial
from typing import Callable

import jax.numpy as jnp
from jax import lax, random

import prax.manifolds as pm


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
    # I don't think any Jacobian correction is required because the Jacobian
    # determinant will be one. This follows from the observation that the
    # circumference of a circle is 2pi and the length of the interval [-pi,
    # +pi] is also 2pi.
    return torus_density(pm.torus.euclid2ang(xtor))

def rejection_sampling(rng: jnp.ndarray, num_samples: int, torus_density: Callable, beta: float=1.0) -> jnp.ndarray:
    """Sample from the torus via a uniform distribution on the angular coordinates.
    If we have access to an unnormalized density `f(x)`, we require a constant
    `M` such that `M * q(x) > f(x)` at every point. Since we are using a
    uniform proposal distribution, `q(x) = 1 / (2.*pi)^2` (squared because of
    two angular dimensions, both uniformly sampled).

    Args:
        rng: Pseudo-random number generator seed.
        num_samples: Number of samples to attempt draw. The number of samples
            returned may be smaller since some samples will be rejected.
        torus_density: The density on the torus from which to generate samples.
        beta: Density concentration parameter.

    Returns:
        samples: Samples from the distribution on the torus.

    """
    # Precompute certain quantities used in rejection sampling.
    M = jnp.square(2.*jnp.pi) * jnp.exp(beta * jnp.log(3.))
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
        theta = 2.*jnp.pi*random.uniform(rng_prop, [2])
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
