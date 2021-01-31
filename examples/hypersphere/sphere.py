from typing import Callable, Sequence, Tuple

import matplotlib.pyplot as plt

import jax.numpy as jnp
import jax.scipy.special as jspsp
from jax import random
from jax import jacobian, vmap
from jax.experimental.ode import odeint


def project_to_tangent(x: jnp.ndarray, v: jnp.ndarray) -> jnp.ndarray:
    """Project to the tangent space of the sphere.

    Args:
        x: Points on the sphere.
        v: Vectors in the ambient Euclidean space to project to the tangent space
            of the sphere.

    Returns:
        pv: The projection of the ambient vectors to the tangent space of the
            sphere.

    """
    return v - jnp.sum(x * v, axis=-1, keepdims=True) * x

def exponential(x: jnp.ndarray, v: jnp.ndarray) -> jnp.ndarray:
    """Computes the exponential map on the sphere.

    Args:
        x: Points on the sphere.
        v: Vectors in the tangent space of the sphere.

    Returns:
        xp: The position of the sphere after one unit of time of geodesic flow.

    """
    n = jnp.linalg.norm(v, axis=-1, keepdims=True)
    xp = jnp.cos(n)*x + jnp.sin(n) / n * v
    return xp

def logarithmic(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    """Computes the logarithmic map on the sphere.

    Args:
        x: Origination points on the sphere.
        y: Destination points on the sphere.

    Returns:
        lg: The tangent vector in the tangent space of the origination point that
            would produce the destination under the exponential map.

    """
    xy = (x * y).sum(axis=-1, keepdims=True)
    v = jnp.arccos(xy)
    lg = v / jnp.sin(v) * (y - xy * x)
    return lg

exp = exponential
log = logarithmic

project_to_sphere = lambda x: x / jnp.linalg.norm(x, axis=-1, keepdims=True)

def sample_uniform(rng: random.PRNGKey, shape: Sequence[int]) -> jnp.ndarray:
    """Sample from the uniform distribution on the sphere.

    Args:
        rng: Pseudo-random number generator seed.
        shape: Shape of the random sample to generate.

    Returns:
        x: Samples distributed uniformly on the sphere.

    """
    x = project_to_sphere(random.normal(rng, shape))
    return x

def uniform_log_density(x: jnp.ndarray) -> jnp.ndarray:
    """Log-density of the uniform distribution on the sphere. Since the Haar
    distribution is uniform, the density is just the inverse of the surface
    area of the sphere. Therefore, the log-density is the negative logarithm of
    the surface area.

    Args:
        x: The locations on the sphere at which to compute the log-density of
            the Haar distribution on the sphere.

    Returns:
        out: The log-density of the Haar distribution of the sphere.

    """
    n = x.shape[-1]
    halfn = 0.5 * n
    logsa = jnp.log(2.) + halfn*jnp.log(jnp.pi) - jspsp.gammaln(halfn)
    return -logsa*jnp.ones(x.shape[:-1])

def ambient_to_spherical_vector_field(afunc: Callable) -> Callable:
    """Convert a vector field on the ambient Euclidean space into a vector field on
    the sphere projecting the ambient vector field to the tangent space of the
    sphere.

    Args:
        afunc: Vector field in the ambient Euclidean space.

    Returns:
        sfunc: Vector field on the sphere.

    """
    sfunc = lambda x, t, *args: project_to_tangent(x, afunc(x, t, *args))
    return sfunc

def spherical_to_chart_vector_field(loc: jnp.ndarray, sfunc: Callable) -> Tuple[Callable]:
    """Takes a vector field defined in an ambient Euclidean space and produces an
    equivalent vector field defined in a local chart using the exponential map
    transformation centered at `loc`.

    Args:
        loc: The center points of the exponential map defining the local
            coordinate system of the sphere.
        sfunc: A vector field defined in an ambient Euclidean space.

    Returns:
        cfunc: A corresponding field defined in the coordinate chart defined by
            the exponential map.
        divfunc: Augmented local coordinate dynamics that also computes the time
            evolution of the divergence.

    """
    def _cfunc(v, t, loc, *args):
        x = exp(loc, v)
        J = jac_log(loc, x)
        f = (J@sfunc(x, t, *args)[..., jnp.newaxis]).squeeze()
        return f

    def cfunc(v, t, *args):
        return _cfunc(v, t, loc, *args)

    def divergence(v, t, *args):
        g = lambda v, loc: _cfunc(v, t, loc, *args)
        vprom = jnp.atleast_2d(v)
        locprom = jnp.atleast_2d(loc)
        return jnp.trace(vmap(jacobian(g))(vprom, locprom), axis1=-2, axis2=-1).squeeze()

    def divfunc(state, t, *args):
        v, _ = state
        f = cfunc(v, t, *args)
        d = -divergence(v, t, *args)
        return f, d

    return cfunc, divfunc

def log_det_jac_exp(x: jnp.ndarray, v: jnp.ndarray) -> jnp.ndarray:
    """The log-determinant of the Jacobian of the exponential map on the sphere
    with respect to the second argument (the tangent space vector).

    Args:
        x: Points on the sphere.
        v: Vectors in the tangent space of the sphere.

    Returns:
        d: The log-determinant of the exponential map.

    """
    n = x.shape[-1]
    r = jnp.linalg.norm(v, axis=-1)
    d = (n - 2) * (jnp.log(jnp.abs(jnp.sin(r))) - jnp.log(r))
    return d

def jac_log(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    """Computes the Jacobian of the logarithmic map on the sphere.

    Args:
        x: Origination points on the sphere.
        y: Destination points on the sphere.

    Returns:
        d: The Jacobian of the logarithmic map.

    """
    r = (x * y).sum(axis=-1, keepdims=True)
    v = r * jnp.arccos(r) / jnp.power(1 - jnp.square(r), 1.5) - 1 / (1 - jnp.square(r))
    a = v[..., jnp.newaxis] * (y - r * x)[..., jnp.newaxis] * x[..., jnp.newaxis, :]
    acr = jnp.arccos(r)
    b = (acr / jnp.sin(acr))[..., jnp.newaxis]
    c = jnp.eye(x.shape[-1])[jnp.newaxis] - x[..., jnp.newaxis] * x[..., jnp.newaxis, :]
    d = a + b * c
    return d

