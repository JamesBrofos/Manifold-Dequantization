from typing import Callable, Sequence

import jax.numpy as jnp
from jax import lax
from jax import grad, vmap


def sphdist(x: jnp.ndarray, y: jnp.ndarray) -> float:
    """Great circle distance on a sphere. Implementation derived from [1].

    [1] https://www.manopt.org/

    Args:
        x: A point on the sphere.
        y: A point on the sphere

    Returns:
        d: The distance between two points on the sphere.

    """
    chordal_distance = jnp.linalg.norm(x - y)
    d = jnp.real(2 * jnp.arcsin(0.5 * chordal_distance))
    return d

proj = lambda x, d: d - (x.dot(d)) * x

def logmap(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    """Logarithmic map to compute the coordinates of a point `y` in terms of the
    velocity in the tangent space of `x` which would be required for a geodesic
    to proceed from `x` to `y`. Implementation derived from [1].

    [1] https://www.manopt.org/

    Args:
        x: A point on the sphere.
        y: A point on the sphere

    Returns:
        out: The vector in the tangent space of `x` such that the geodesic
            running for one unit of time from `x` in the that direction such
            that the destination of the geodesic is `y`.

    """
    v = proj(x, y - x)
    nv = jnp.linalg.norm(v)
    di = sphdist(x, y)
    return lax.cond(di > 1e-6, lambda _: v * (di / nv), lambda _: v, None)

def sphgrad(fn: Callable, obs: jnp.ndarray, *args: Sequence) -> jnp.ndarray:
    """Compute the gradient of a function constrained to the sphere. This is the
    orthogonal projection of the ambient Euclidean gradient to the tangent
    space of the sphere.

    Args:
        fn: A function defined on the sphere that is smoothly extended to the
            ambient Euclidean space.
        obs: An array of locations at which to compute the sphere gradient.
        *args: Parameters to pass to the function.

    Returns:
        sph_grad: The gradient on the sphere.

    """
    amb_grad = vmap(lambda _: grad(fn)(_, *args))(obs)
    sph_grad = amb_grad - jnp.sum(amb_grad * obs, axis=-1, keepdims=True) * obs
    return sph_grad
