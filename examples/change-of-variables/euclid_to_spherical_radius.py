import jax.numpy as jnp
from jax import random
from jax import jacobian


def euclid2sphericalrad(x):
    r = jnp.linalg.norm(x)
    s = x / r
    return jnp.hstack((r, s))

rng = random.PRNGKey(0)
for ndim in range(1, 5):
    x = random.normal(rng, [ndim])
    xs = x / jnp.linalg.norm(x)
    r = 5.
    xp = r * xs
    J = jacobian(euclid2sphericalrad)(xp)
    det = jnp.sqrt(jnp.linalg.det(J.T@J))
    print('dim.: {} - determinant: {:.10f} - pred.: {:.10f}'.format(ndim, det, 1 / r**(ndim-1)))
