import jax.numpy as jnp
from jax import jacobian, vmap

from coordinates import ang2euclid, euclid2ang


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
    x = ang2euclid(theta)
    xp = mobius_circle(x, w)
    return jnp.squeeze(euclid2ang(xp))

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


if __name__ == '__main__':
    import os
    import matplotlib.pyplot as plt
    from jax import random

    rng = random.PRNGKey(1)
    x = random.normal(rng, [10, 2])
    x /= jnp.linalg.norm(x, axis=-1)[..., jnp.newaxis]
    w = random.normal(rng, [15, 2])
    w = 0.99 / (1. + jnp.linalg.norm(w, axis=-1)[..., jnp.newaxis]) * w

    xp = mobius_circle(x, w)
    theta = jnp.linspace(0., 2*jnp.pi, 100, endpoint=False)
    omega = jnp.squeeze(mobius_flow(theta, w))
    ctheta = ang2euclid(theta)
    comega = ang2euclid(omega[0])

    unif = 2.0*jnp.pi*random.uniform(rng, [1000000])
    x = mobius_flow(unif, w).mean(0)
    log_prob = mobius_log_prob(unif, w)
    prob = jnp.exp(log_prob)

    fig, axes = plt.subplots(1, 4, figsize=(13, 4))
    axes[0].plot(ctheta[:, 0], ctheta[:, 1], '.')
    axes[0].grid(linestyle=':')
    axes[0].axis('square')
    axes[1].plot(comega[:, 0], comega[:, 1], '.')
    axes[1].plot(w[0][0], w[0][1], '.', markersize=10)
    axes[1].grid(linestyle=':')
    axes[1].axis('square')
    axes[2].plot(theta, omega.T, '-')
    axes[2].grid(linestyle=':')
    axes[2].axis('square')
    axes[3].hist(x, density=True, bins=100, alpha=0.7)
    axes[3].plot(x, prob, '.')
    axes[3].grid(linestyle=':')
    plt.tight_layout()
    plt.savefig(os.path.join('images', 'mobius-flow.png'))
