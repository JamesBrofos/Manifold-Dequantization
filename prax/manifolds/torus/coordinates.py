import jax.numpy as jnp


_ang2euclid = lambda theta: jnp.vstack((jnp.cos(theta), jnp.sin(theta))).T
_euclid2ang = lambda xy: jnp.arctan2(xy[..., 1], xy[..., 0])

def ang2euclid(theta: jnp.ndarray) -> jnp.ndarray:
    """Given two angular coordinates, compute the corresponding position on the
    product manifold of two circles embedded in the plane.
    """
    thetaa, thetab = theta[..., 0], theta[..., 1]
    return jnp.hstack((_ang2euclid(thetaa), _ang2euclid(thetab)))

def euclid2ang(xtor: jnp.ndarray) -> jnp.ndarray:
    """Given two points on the circle, compute the angular coordinates."""
    thetaa = _euclid2ang(xtor[..., :2])
    thetab = _euclid2ang(xtor[..., 2:])
    return jnp.vstack((thetaa, thetab)).T
