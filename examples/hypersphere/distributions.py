import jax.numpy as jnp

import prax.manifolds as pm


def embedded_sphere_density(xsph: jnp.ndarray) -> jnp.ndarray:
    """Unnormalized multimodal density on the four-dimensional sphere.

    Args:
        xsph: Observations on the sphere at which to compute the unnormalized
            density.

    Returns:
        out: The unnormalized density at the provided points on the sphere.

    """
    p = lambda x, mu: jnp.exp(10.*x.dot(mu))
    mua = pm.sphere.hsph2euclid(1.7, -1.5, 2.3)
    mub = pm.sphere.hsph2euclid(-3.0, 1.0, 3.0)
    muc = pm.sphere.hsph2euclid(0.6, -2.6, 4.5)
    mud = pm.sphere.hsph2euclid(-2.5, 3.0, 5.0)
    return p(xsph, mua) + p(xsph, mub) + p(xsph, muc) + p(xsph, mud)
