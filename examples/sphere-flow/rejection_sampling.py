from functools import partial
from typing import Callable

import jax.numpy as jnp
from jax import lax, random

import prax.distributions as pd


def rejection_sampling(rng: jnp.ndarray, num_samples: int, num_dims: int, sphere_density: Callable) -> jnp.ndarray:
    """Samples from the sphere in embedded coordinates using the uniform
    distribution on the sphere as a proposal density.

    Args:
        rng: Pseudo-random number generator seed.
        num_samples: Number of samples to attempt draw. The number of samples
            returned may be smaller since some samples will be rejected.
        num_dims: Dimensionality of the samples.
        sphere_density: The density on the sphere from which to generate samples.

    Returns:
        samples: Samples from the distribution on the sphere.

    """
    # Precompute certain quantities used in rejection sampling. The upper bound
    # on the density was found according to large-scale simulation.
    # >>> rng = random.PRNGKey(0)
    # >>> embedded_sphere_density(pd.sphere.haarsph(rng, [10000000, 3])).max()
    prop_dens = jnp.exp(pd.sphere.haarsphlogdensity(jnp.array([0., 0., 1.])))
    M = 25000. / prop_dens
    denom = M * prop_dens

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
        xsph = pd.sphere.haarsph(rng_prop, [num_dims])
        numer = sphere_density(xsph)
        alpha = numer / denom
        unif = random.uniform(rng_acc)
        isacc = unif < alpha
        return xsph, isacc, it + 1

    def sample(_, it):
        """Samples in a loop so that the total number of samples has a predictable
        shape. The first argument is ignored. The second argument is the
        sampling iteration.

        Args:
            _: Ignored argument for `lax.scan` compatibility.
            it: Sampling iteration number.

        Returns:
            _, xsph: A tuple containing the ignored input quantity and the
                accepted sample.

        """
        state = lax.while_loop(cond, partial(sample_once, it), (jnp.zeros(num_dims), False, 0))
        xsph, isacc, num_iters = state
        return _, xsph

    _, xsph = lax.scan(sample, None, jnp.arange(num_samples))
    return xsph
