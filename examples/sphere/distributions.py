import jax.numpy as jnp

import prax.manifolds as pm


def embedded_sphere_density(xsph: jnp.ndarray) -> jnp.ndarray:
    """Unnormalized multimodal density on the sphere.

    Args:
        xsph: Observations on the sphere at which to compute the unnormalized
            density.

    Returns:
        out: The unnormalized density at the provided points on the sphere.

    """
    p = lambda x, mu: jnp.exp(10.*x.dot(mu))
    mua = pm.sphere.sph2euclid(0.7, 1.5)
    mub = pm.sphere.sph2euclid(-1., 1.)
    muc = pm.sphere.sph2euclid(0.6, 0.5)
    mud = pm.sphere.sph2euclid(-0.7, 4.)
    return p(xsph, mua) + p(xsph, mub) + p(xsph, muc) + p(xsph, mud)
