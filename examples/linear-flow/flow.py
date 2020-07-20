import os
from functools import partial

import matplotlib.pyplot as plt

import jax.numpy as np
import jax.scipy.stats as spst
from jax import jit, lax, random, value_and_grad, vmap
from jax.experimental.optimizers import adam, clip_grads

from jax.config import config
config.update("jax_enable_x64", True)

import nvp

def project(x):
    norm = np.sqrt(np.square(x).sum(-1))
    return x / norm[..., np.newaxis]

def base_sample_fn(rng, N):
    return random.normal(rng, [N, 2])

def base_log_prob_fn(x):
    return spst.norm.logpdf(x, 0., 1.).sum(axis=-1)

def ambient_density(x):
    Id = np.eye(2)
    mu = np.array([1.5, 1.5])
    d = 0.5 * spst.multivariate_normal.pdf(x, mu, Id)
    d += 0.5 * spst.multivariate_normal.pdf(x, -mu, Id)
    return d

def target_density(y, radii):
    delta = radii[1] - radii[0]
    halfdelta = delta / 2.
    mid = radii - halfdelta
    def _cd(radius):
        return radius * ambient_density(radius * y)
    return delta * vmap(_cd)(mid).sum(0)

def circle_density(y, ps, cs, radii):
    delta = radii[1] - radii[0]
    halfdelta = delta / 2.
    mid = radii - halfdelta
    def _cd(radius):
        return radius * nvp.prob_nvp_chain(ps, cs, base_log_prob_fn, radius * y)
    return delta * vmap(_cd)(mid).sum(0)

def kl_divergence(rng, ps, cs):
    x = nvp.sample_nvp_chain(rng, ps, cs, base_sample_fn, 15)
    y = project(x)
    m = 10.
    radii = np.linspace(0., m, 20)[1:]
    logp = np.log(target_density(y, radii))
    logq = np.log(circle_density(y, ps, cs, radii))
    return np.mean(logq - logp)

from jax.tree_util import tree_map, tree_flatten

def zero_nans(grad_tree):
  zero = lambda g: np.where(np.isnan(g), np.zeros_like(g), g)
  return tree_map(zero, grad_tree)


@partial(jit, static_argnums=(1, ))
def flow(ps, cs):
    init_fun, update_fun, get_params = adam(1e-5)
    opt_state = init_fun(ps)
    def step(opt_state, it):
        ps = get_params(opt_state)
        rng_ = random.fold_in(rng, it)
        kl, grads = value_and_grad(kl_divergence, (1, ))(rng_, ps, cs)
        grads = grads[0]
        grads = zero_nans(grads)
        grads = clip_grads(grads, 1.)
        opt_state = update_fun(it, grads, opt_state)
        return opt_state, kl
    opt_state, kl = lax.scan(step, opt_state, np.arange(0))
    return get_params(opt_state), kl


rng = random.PRNGKey(0)
ps, cs = nvp.init_nvp_chain(rng, 4)
ps, kl = flow(ps, cs)

x = nvp.sample_nvp_chain(rng, ps, cs, base_sample_fn, 20000)
y = project(x)
theta_samples = np.arctan2(y[:, 0], y[:, 1])
theta = np.linspace(-np.pi, np.pi, 1000)
y = np.vstack((np.cos(theta), np.sin(theta))).T
radii = np.linspace(0., 10., 50)[1:]
dens = circle_density(y, ps, cs, radii)
target = target_density(y, radii)



fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].hist(theta_samples, bins=20, density=True, alpha=0.7, label='Approx. Samples')
axes[0].plot(theta, dens, '-', label='Approx. Density')
axes[0].plot(theta, target, '--', label='Target Density')
axes[0].legend()
axes[0].grid(linestyle=':')
axes[1].plot(np.arange(kl.size) + 1, kl, '-')
axes[1].grid(linestyle=':')
plt.savefig(os.path.join('images', 'real-nvp-initial.png'))
