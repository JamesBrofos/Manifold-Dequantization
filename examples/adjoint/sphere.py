import argparse
import os
from functools import partial

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import jax.numpy as np
import jax.scipy.stats as spst
from jax import jit, lax, pmap, random, value_and_grad, vmap
from jax.experimental.optimizers import adam, clip_grads
from jax.tree_util import tree_map, tree_flatten

import nvp

parser = argparse.ArgumentParser(description='Projecting ambient distribution to circle')
parser.add_argument('--step-size', type=float, default=1e-2, help='Linear flow learning rate')
parser.add_argument('--num-steps', type=int, default=1000, help='Number of learning steps')
parser.add_argument('--num-samples', type=int, default=1000, help='Number of samples to use for computing KL divergence')
parser.add_argument('--num-discrete', type=int, default=10, help='Number of discrete integration steps for computing manifold density')
parser.add_argument('--num-hidden', type=int, default=50, help='Number of hidden units to use in RealNVP if this is the ambient density selected.')
parser.add_argument('--seed', type=int, default=0, help='Pseudo-random number generator seed')
args = parser.parse_args()


def zero_nans(grad_tree):
    """If there are any NaNs in the array, convert them to zeros."""
    zero = lambda g: np.where(np.isnan(g), np.zeros_like(g), g)
    return tree_map(zero, grad_tree)

def project(x):
    """Projection of the ambient space to the unit circle."""
    norm = np.sqrt(np.square(x).sum(-1))
    return x / norm[..., np.newaxis]

def ambient_gaussian(params, x):
    """Density of the ambient multivariate normal density."""
    mu, log_sigmasq = params
    sigmasq = np.exp(log_sigmasq)
    Sigma = np.diag(sigmasq)
    return spst.multivariate_normal.pdf(x, mu, Sigma)

def ambient_gaussian_mixture(params, x):
    """Density of the ambient bivariate Gaussian mixture density."""
    mu, log_sigmasq = params
    Sigma = np.diag(np.exp(log_sigmasq))
    d = 0.25 * spst.multivariate_normal.pdf(x, mu, Sigma)
    d += 0.25 * spst.multivariate_normal.pdf(x, -mu, Sigma)
    d += 0.25 * spst.multivariate_normal.pdf(x, np.array([0., -1.5, 0.]), Sigma)
    d += 0.25 * spst.multivariate_normal.pdf(x, np.array([0., 1.5, 0.]), Sigma)
    return d

def circle_density(y, ambient, params, bound):
    """Compute the manifold-constrained density by integrating over the set of
    points that retract to the specified point on the circle.

    """
    radii = np.linspace(0, bound, args.num_discrete + 1)[1:]
    delta = radii[1] - radii[0]
    halfdelta = delta / 2.
    mid = radii - halfdelta
    def _cd(radius):
        return radius * ambient(params, radius * y)
    return delta * vmap(_cd)(mid).sum(0)

def ambient_gaussian_sample(rng, params, shape):
    """Sample from the ambient multivariate normal density."""
    mu, log_sigmasq = params
    sigmasq = np.exp(log_sigmasq)
    sigma = np.sqrt(sigmasq)
    z = random.normal(rng, shape)
    return sigma * z + mu

def klcircle(params, rng):
    """Kullback-Liebler divergence for circle-constrained random variables."""
    x = nvp.ambient_nvp_chain_sample(rng, params, [args.num_samples, 3])
    y = project(x)
    bound = lax.stop_gradient(np.abs(x).max() + 5.)
    logq = np.log(circle_density(y, nvp.ambient_nvp_chain_density, params, bound))
    logp = np.log(circle_density(y, ambient_gaussian_mixture, params_actual, 10.))
    return np.square(np.mean(logq - logp))

@partial(jit, static_argnums=(1, ))
def ambient_flow(params, loss):
    """Estimate a manifold-constrained density by learning a flow in an ambient
    Euclidean space and using a retraction to induce a density on the manifold.

    """
    init_fun, update_fun, get_params = adam(args.step_size)
    opt_state = init_fun(params)
    def step(opt_state, it):
        params = get_params(opt_state)
        rng_ = random.fold_in(rng, it)
        kl, grads = value_and_grad(loss)(params, rng_)
        grads = zero_nans(grads)
        grads = clip_grads(grads, 1.)
        opt_state = update_fun(it, grads, opt_state)
        return opt_state, kl
    opt_state, klsq = lax.scan(step, opt_state, np.arange(args.num_steps))
    kl = np.sqrt(klsq)
    params = get_params(opt_state)
    return params, kl


rng = random.PRNGKey(args.seed)
mu_actual = np.array([1.5, 1.5, 1.5])
sigmasq_actual = np.array([1., 1., 1.])
params_actual = (mu_actual, np.log(sigmasq_actual))

params = nvp.init_nvp_chain(rng, 4, 3, args.num_hidden, np.float64)
params, kl = ambient_flow(params, klcircle)
x = nvp.ambient_nvp_chain_sample(rng, params, [1000, 3])
y = project(x)
dens = circle_density(y, nvp.ambient_nvp_chain_density, params, 5.)


plt.figure()
plt.plot(np.arange(kl.size) + 1, kl, '-')
plt.grid(linestyle=':')
plt.title('KL-Divergence by Iteration')
plt.savefig(os.path.join('images', 'sphere-kl-divergence.png'))

fig = plt.figure(figsize=(4, 4))
ax = fig.add_subplot(111, projection='3d')
ax.view_init(elev=40., azim=30.)
ax.scatter(y[:, 0], y[:, 1], y[:, 2], c=dens)
ax.grid(linestyle=':')
ax.set_xlim((-1.1, 1.1))
ax.set_ylim((-1.1, 1.1))
ax.set_zlim((-1.1, 1.1))
for ii in range(0, 360):
    ax.view_init(elev=20., azim=ii)
    plt.savefig(os.path.join('images', "sphere-samples-{}.png".format(ii)))

