import jax.numpy as jnp
from jax import random
from jax import jacobian


def euclid2sphericalrad(x):
    r = jnp.linalg.norm(x)
    s = x / r
    return jnp.hstack((r, s))

rng = random.PRNGKey(0)
for ndim in range(1, 10):
    x = random.normal(rng, [ndim])
    xs = x / jnp.linalg.norm(x)
    scale = 5.
    J = jacobian(euclid2sphericalrad)(scale * xs)
    det = jnp.sqrt(jnp.linalg.det(J.T@J))
    print('dim.: {} - determinant: {:.10f} - pred.: {:.10f}'.format(ndim, det, 1 / scale**(ndim-1)))
