import jax.numpy as jnp
from jax import jacobian, vmap

import prax.manifolds as pm


def mobius_circle(x: jnp.ndarray, w: jnp.ndarray) -> jnp.ndarray:
    """Implements the Mobius transformation on a circle embedded in the plane. The
    implementation is vectorized so that if `x` and `w` are both matrices then
    the output is the individual Mobius transformation of `x` by each center
    individually.

    Args:
        x: Observations on the circle.
        w: Center points of the Mobius transformation.

    Returns:
        xp: The observations on the circle transformed under the Mobius
            transformation.

    """
    w = jnp.atleast_2d(w)
    x = jnp.atleast_2d(x)
    w_sqnorm = jnp.square(jnp.linalg.norm(w, axis=-1))
    xmw = jnp.moveaxis(x[:, jnp.newaxis] - w, 0, 1)
    xmw_sqnorm = jnp.square(jnp.linalg.norm(xmw, axis=-1))
    ratio = ((1 - w_sqnorm)[..., jnp.newaxis] / xmw_sqnorm)[..., jnp.newaxis]
    xp = ratio * xmw - w[:, jnp.newaxis]
    return xp

def mobius_angle(theta: jnp.ndarray, w: jnp.ndarray) -> jnp.ndarray:
    """The Mobius transformation on the circle as viewed as an angle in the
    interval [0, 2pi).

    Args:
        theta: The angle parameterizing a point on the circle.
        w: Center points of the Mobius transformation.

    Returns:
        out: The Mobius transformation as a transformation on angles.

    """
    x = pm.sphere.ang2euclid(theta)
    xp = mobius_circle(x, w)
    return jnp.squeeze(pm.sphere.euclid2ang(xp))

def mobius_flow(theta: jnp.ndarray, w: jnp.ndarray) -> jnp.ndarray:
    """Following [1] the Mobius flow must have the property that the endpoints of
    the interval [0, 2pi] are mapped to themselves. This transformation
    enforces this property by appling a backwards rotation to bring the zero
    angle back to itself.

    Args:
        theta: The angle parameterizing a point on the circle.
        w: Center points of the Mobius transformation.

    Returns:
        out: The Mobius flow as a transformation on angles leaving the endpoints
            of the interval invariant.

    """
    theta = jnp.hstack((0., theta))
    omega = mobius_angle(theta, w)
    omega -= omega[..., [0]]
    return jnp.squeeze(jnp.mod(omega[..., 1:], 2.*jnp.pi))

def mobius_log_prob(theta: jnp.ndarray, w: jnp.ndarray) -> jnp.ndarray:
    """Compute the log-density of the Mobius flow assuming the base distribution is
    uniform on the interval [0, 2pi).

    Args:
        theta: The angle parameterizing a point on the circle.
        w: Center points of the Mobius transformation.

    Returns:
        out: The log-density of the transformation of a uniformly-distributed
            angle under a Mobius transformation.

    """
    theta = jnp.atleast_1d(theta)
    log_base = -jnp.log(2.*jnp.pi)
    jac = vmap(jacobian(lambda t: mobius_flow(t, w)))(theta).mean(-1)
    fldj = jnp.log(jac)
    return jnp.squeeze(log_base - fldj)
