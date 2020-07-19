import jax.numpy as np
from jax import lax, random


def randson(rng, n):
    """Generate a random element of SO(n) by randomly sampling a normal random
    matrix, computing its QR-decomposition and checking if the determinant of
    the Q-factor equals one. If the determinant is negative one, then the
    process is iterated with a new random seed.

    """
    def cond(val):
        return np.abs(np.linalg.det(val[0]) - 1.) > 1e-10

    def body(val):
        _, it = val
        Q, _ = np.linalg.qr(random.normal(random.fold_in(rng, it), [n, n]))
        return [Q, it + 1]

    Q = np.zeros((n, n))
    return lax.while_loop(cond, body, [Q, 0])[0]

def randlie(rng, n):
    """Generate a random vector in the Lie algebra."""
    v = random.normal(rng, [n, n])
    v = v - v.T
    return v / np.linalg.norm(v)
