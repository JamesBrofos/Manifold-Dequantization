from functools import partial

import jax
import jax.numpy as np
from jax import lax, random

import rodrigues

def kernel_and_grad(x, y):
    x = np.expand_dims(x, -2)
    y = np.expand_dims(y, -3)
    delta = (x - y)
    rsq = np.sum(delta**2, axis=-1)
    K = np.exp(-0.5*rsq)
    dK = np.expand_dims(K, axis=-1) * (-delta)
    return K, dK


def _kernel(x, y):
    def step(_, ys):
        d = np.square(rodrigues.liedist(x, ys))
        # d = np.sum(np.square(x - ys))
        d = lax.cond(np.linalg.norm(x - ys) > 1e-15,
                     lambda _: d,
                     lambda _: np.zeros_like(d),
                     None)
        return _, d
    Dsq = lax.scan(step, None, y)[1]
    return np.exp(-0.5 * Dsq)

@jax.jit
def jacobian(y):
    Zero = np.zeros((1, 3))
    def stack(A, idx):
        return np.vstack((A[:idx], Zero, A[idx:]))
    return np.array([stack(
        jax.jacobian(_kernel)(y[idx], np.vstack((y[:idx], y[idx+1:]))), idx) for idx in range(y.shape[0])])


kernel = jax.vmap(_kernel, in_axes=(0, None))

if __name__ == '__main__':
    from jax.config import config
    config.update("jax_enable_x64", True)
    rng = random.PRNGKey(0)
    y = random.normal(rng, [50, 3])
    K = kernel(y, y)
    dK = jacobian(y)
    Kp, dKp = kernel_and_grad(y, y)
