import jax.numpy as jnp
from jax import lax, nn, random
from jax import grad, jit, vmap


def rational_quadratic_parameters(x: jnp.ndarray, xk: jnp.ndarray, yk: jnp.ndarray, delta: jnp.ndarray) -> jnp.ndarray:
    """Compute necessary intermediate parameters used for computing the spline
    flow. For details on the rational quadratic transformation, consult [1].

    [1] https://arxiv.org/pdf/1906.04032.pdf

    """
    idxn = jnp.searchsorted(xk, x)
    idx = idxn - 1
    xi = (x - xk[idx]) / (xk[idxn] - xk[idx])
    ym = yk[idxn] - yk[idx]
    sk = ym / (xk[idxn] - xk[idx])
    dk = delta[idx]
    dkp = delta[idxn]
    dp = dkp + dk
    xib = xi * (1.0 - xi)
    xisq = jnp.square(xi)
    return (idx, xi, xib, xisq, dk, dkp, dp, sk, ym)

def rational_quadratic(x: jnp.ndarray, xk: jnp.ndarray, yk: jnp.ndarray, delta: jnp.ndarray) -> jnp.ndarray:
    """Compute the rational quadratic diffeomorphism of the interval [-1, +1]. The
    rational quadratic is defined as a series of segments, each of which is
    equipped with a rational quadratic function. The derivative of the rational
    quadratic function is constructed so as to be strictly positive.

    Args:
        x: Points to transform by the rational quadratic function.
        xk: The x-coordinates of the intervals on which the rational quadratics
            are defined.
        yk: The y-coordinates of the destination intervals of the rational
            quadratic transforms.
        delta: Derivatives at internal points.

    Returns:
        rq: The rational quadratic transformation of the [-1, +1] interval
            evaluated at the inputs `x` given the knots defined by `xk` and `yk`
            and derivatives `delta`.

    """
    idx, xi, xib, xisq, dk, dkp, dp, sk, ym = rational_quadratic_parameters(x, xk, yk, delta)
    rq = yk[idx] + ym * (sk * xisq + dk * xib) / (sk + (dp - 2.0 * sk) * xib)
    return rq

def grad_rational_quadratic(x: jnp.ndarray, xk: jnp.ndarray, yk: jnp.ndarray, delta: jnp.ndarray) -> jnp.ndarray:
    """The gradient of the rational quadratic transformation of the interval [-1, +1].

    Args:
        x: Points to transform by the rational quadratic function.
        xk: The x-coordinates of the intervals on which the rational quadratics
            are defined.
        yk: The y-coordinates of the destination intervals of the rational
            quadratic transforms.
        delta: Derivatives at internal points.

    Returns:
        drq: The derivative of the rational quadratic function.

    """
    idx, xi, xib, xisq, dk, dkp, dp, sk, ym = rational_quadratic_parameters(x, xk, yk, delta)
    drq = jnp.square(sk) * (dkp * xisq + 2.0 * sk * xib + dk * jnp.square(1.0 - xi)) / jnp.square(sk + (dp - 2.0 * sk) * xib)
    return drq

