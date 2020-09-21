from functools import partial
from typing import Callable

import jax.numpy as jnp
from jax import lax, random

import prax.distributions as pd

from coordinates import sph2euclid, hsph2euclid
from globe import is_land

def embedded_sphere_density(xsph: jnp.ndarray) -> jnp.ndarray:
    """Unnormalized correlated density on the sphere.

    Args:
        xsph: Observations on the sphere at which to compute the unnormalized
            density.

    Returns:
        out: The unnormalized density at the provided points on the sphere.

    """
    p = lambda x, mu: jnp.exp(10.*x.dot(mu))
    mua = sph2euclid(0.7, 1.5)
    mub = sph2euclid(-1., 1.)
    muc = sph2euclid(0.6, 0.5)
    mud = sph2euclid(-0.7, 4.)
    return p(xsph, mua) + p(xsph, mub) + p(xsph, muc) + p(xsph, mud)

def embedded_hypersphere_density(xsph: jnp.ndarray) -> jnp.ndarray:
    """Unnormalized correlated density on the four-dimensional hypersphere.

    Args:
        xsph: Observations on the sphere at which to compute the unnormalized
            density.

    Returns:
        out: The unnormalized density at the provided points on the sphere.

    """
    p = lambda x, mu: jnp.exp(10.*x.dot(mu))
    mua = hsph2euclid(1.7, -1.5, 2.3)
    mub = hsph2euclid(-3.0, 1.0, 3.0)
    muc = hsph2euclid(0.6, -2.6, 4.5)
    mud = hsph2euclid(-2.5, 3.0, 5.0)
    return p(xsph, mua) + p(xsph, mub) + p(xsph, muc) + p(xsph, mud)

def embedded_earth_density(xsph: jnp.ndarray) -> jnp.ndarray:
    c = 180. / jnp.pi
    x, y, z = xsph[..., 0], xsph[..., 1], xsph[..., 2]
    lat = c * jnp.arctan2(z, jnp.sqrt(jnp.square(x) + jnp.square(y)))
    lng = c * jnp.arctan2(y, x)
    return 25000. * is_land(lat, lng)

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
    # M = 22355. / prop_dens
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


if __name__ == '__main__':
    import os
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    rng = random.PRNGKey(0)
    xsph = rejection_sampling(rng, 10000, 3, embedded_sphere_density)
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(xsph[:, 0], xsph[:, 1], xsph[:, 2], '.', alpha=0.1)
    ax.grid(linestyle=':')
    plt.savefig(os.path.join('images', 'rejection-samples.png'))

    xsph = rejection_sampling(rng, 10000, 3, embedded_earth_density)
    dens = embedded_earth_density(xsph)
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(xsph[:, 0], xsph[:, 1], xsph[:, 2], '.', alpha=0.1)
    ax.grid(linestyle=':')
    plt.savefig(os.path.join('images', 'earth-rejection-samples.png'))
