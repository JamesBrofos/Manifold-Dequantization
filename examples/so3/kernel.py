from functools import partial

import jax
import jax.numpy as np
from jax import lax, random

import rodrigues

def _kernel(x, y, bandwidth):
    def step(_, ys):
        d = np.square(rodrigues.liedist(x, ys))
        # This is to reproduce the Euclidean kernel case.
        # d = np.square(x - ys).sum()
        return _, d
    Dsq = lax.scan(step, None, y)[1]
    return np.exp(-0.5 * Dsq / np.square(bandwidth))

def lie_kernel_and_grad(rng, theta, bandwidth, scale=1e-5):
    kernel = jax.vmap(_kernel, in_axes=(0, None, None))
    jac = lambda x, y, bw: jax.jacobian(kernel)(x, y, bw).sum(2)
    K = kernel(theta, theta, bandwidth)
    noise = scale * random.normal(rng, theta.shape)
    dK = jac(theta, theta + noise, bandwidth)
    return K, dK

def euclid_kernel_and_grad(rng, theta, bandwidth):
    x = np.expand_dims(theta, -2)
    y = np.expand_dims(theta, -3)
    delta = (x - y) / bandwidth
    rsq = np.sum(delta**2, axis=-1)
    K = np.exp(-0.5*rsq)
    dK = np.expand_dims(K, axis=-1) * (-delta) / bandwidth
    return K, dK


if __name__ == '__main__':
    from jax.config import config
    config.update("jax_enable_x64", True)

    rng = random.PRNGKey(0)
    theta = random.normal(random.fold_in(rng, 0), [10, 3])
    bw = 2.
    K, dK = lie_kernel_and_grad(rng, theta, bw, scale=0.)
    Kp, dKp = euclid_kernel_and_grad(rng, theta, bw)

