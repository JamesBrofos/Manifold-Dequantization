import os

import matplotlib.pyplot as plt
import numpy as np

import jax.numpy as jnp
from jax import random


def target(theta: jnp.ndarray) -> jnp.ndarray:
    phi = 1.94
    return jnp.exp(3.*jnp.cos(theta.sum(-1) - phi))

# Brute force grid search to find a value that satisfies the condition for
# rejection sampling.
linear = jnp.linspace(0., 2.*jnp.pi, 10000, endpoint=False) - jnp.pi
xx, yy = jnp.meshgrid(linear, linear)
grid = jnp.vstack((xx.ravel(), yy.ravel())).T
dens = target(grid)
M = jnp.square(2.*jnp.pi) * (dens.max() + 1.)

# Perform rejection sampling.
rng = random.PRNGKey(0)
rng, prop_rng, acc_rng = random.split(rng, 3)
theta = 2.*jnp.pi*random.uniform(prop_rng, [100000, 2]) - jnp.pi
denom = M / jnp.square(2.*jnp.pi)
numer = target(theta)
alpha = numer / denom
unif = random.uniform(acc_rng, alpha.shape)
samples = theta[unif < alpha]
print('obtained {} samples'.format(len(samples)))
np.savetxt(os.path.join('data', 'samples.txt'), samples)

# Visualize target density and samples.
fig, axes = plt.subplots(1, 2, figsize=(9, 4))
axes[0].contourf(xx, yy, dens.reshape(xx.shape), cmap=plt.cm.jet)
axes[0].set_xlim((-jnp.pi, jnp.pi))
axes[0].set_ylim((jnp.pi, -jnp.pi))
axes[0].set_title('Target Density')
axes[1].plot(samples[:, 0], samples[:, 1], '.', alpha=0.5)
axes[1].set_xlim((-jnp.pi, jnp.pi))
axes[1].set_ylim((jnp.pi, -jnp.pi))
axes[1].grid(linestyle=':')
axes[1].set_title('Samples')
plt.tight_layout()
plt.savefig(os.path.join('images', 'density-rejection-sampling.png'))
