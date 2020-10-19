import argparse
import os
from functools import partial
from typing import Callable, Sequence, Tuple

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import jax.numpy as jnp
import jax.scipy.special as jspsp
import jax.scipy.stats as jspst
from jax import lax, nn, ops, random, tree_util
from jax import grad, jacobian, jit, value_and_grad, vmap
from jax.experimental import optimizers, stax

import prax.distributions as pd
import prax.manifolds as pm
from prax.bijectors import realnvp, permute

import ambient
import dequantization
from distributions import log_unimodal, log_multimodal
from polar import polar, transp, vecpolar


parser = argparse.ArgumentParser(description='Density estimation for the orthogonal group')
parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
parser.add_argument('--num-batch', type=int, default=100, help='Number of samples per batch')
parser.add_argument('--num-realnvp', type=int, default=5, help='Number of RealNVP bijectors to employ')
parser.add_argument('--num-steps', type=int, default=3000, help='Number of gradient descent iterations')
parser.add_argument('--num-ambient', type=int, default=512, help='Number of hidden units in the ambient network')
parser.add_argument('--num-dequantization', type=int, default=128, help='Number of hidden units in dequantization network')
parser.add_argument('--noise-scale', type=float, default=1., help='Additive noise scale')
parser.add_argument('--seed', type=int, default=0, help='Pseudo-random number generator seed')
args = parser.parse_args()

def negative_elbo(rng: jnp.ndarray, bij_params: Sequence[jnp.ndarray], bij_fns: Sequence[Callable], deq_params: Sequence[jnp.ndarray], deq_fn: Callable, xon: jnp.ndarray) -> jnp.ndarray:
    """Compute the negative evidence lower bound of the dequantizing distribution.

    Args:
        rng: Pseudo-random number generator seed.
        bij_params: List of arrays parameterizing the RealNVP bijectors.
        bij_fns: List of functions that compute the shift and scale of the RealNVP
            affine transformation.
        deq_params: Parameters of the mean and scale functions used in
            the log-normal dequantizer.
        deq_fn: Function that computes the mean and scale of the dequantization
            distribution.
        xon: Observations on O(n).

    Returns:
        nelbo: The negative evidence lower bound.

    """
    xdeq, deq_log_dens = dequantization.dequantize(rng, deq_params, deq_fn, xon, 1)
    amb_log_dens = ambient.log_prob(bij_params, bij_fns, xdeq)
    elbo = jnp.mean(amb_log_dens - deq_log_dens, axis=0)
    nelbo = -elbo
    return nelbo

def clip_and_zero_nans(g: jnp.ndarray) -> jnp.ndarray:
    """Clip the input to within a certain range and remove the NaNs in a matrix by
    replacing them with zeros. This function is useful for ensuring stability
    in gradient descent.

    Args:
        g: Matrix whose elements should be clipped to within a certain range
            and whose NaN elements should be replaced by zeros.

    Returns:
        out: The input matrix but with clipped values and NaN elements replaced
            by zeros.

    """
    g = jnp.where(jnp.isnan(g), jnp.zeros_like(g), g)
    g = jnp.clip(g, -1., 1.)
    return g

def loss(rng: jnp.ndarray, bij_params: Sequence[jnp.ndarray], bij_fns: Sequence[Callable], deq_params: Sequence[jnp.ndarray], deq_fn: Callable, xon: jnp.ndarray) -> float:
    """Loss function composed of the evidence lower bound and score matching
    loss.

    Args:
        rng: Pseudo-random number generator seed.
        bij_params: List of arrays parameterizing the RealNVP bijectors.
        bij_fns: List of functions that compute the shift and scale of the RealNVP
            affine transformation.
        deq_params: Parameters of the mean and scale functions used in
            the log-normal dequantizer.
        deq_fn: Function that computes the mean and scale of the dequantization
            distribution.
        xon: Observations on O(n).

    Returns:
        nelbo: The negative evidence lower bound.

    """
    rng, rng_rej, rng_elbo, rng_deq = random.split(rng, 4)
    nelbo = negative_elbo(rng_elbo, bij_params, bij_fns, deq_params, deq_fn, xon)
    nelbo = nelbo.mean()
    return nelbo

@partial(jit, static_argnums=(2, 4, 5, 6))
def train(rng: random.PRNGKey, bij_params: Sequence[jnp.ndarray], bij_fns: Sequence[Callable], deq_params: Sequence[jnp.ndarray], deq_fn: Callable, num_steps: int, lr: float) -> Tuple:
    opt_init, opt_update, get_params = optimizers.adam(lr)
    def step(opt_state, it):
        rng_step = random.fold_in(rng, it)
        bij_params, deq_params = get_params(opt_state)
        loss_val, loss_grad = value_and_grad(loss, (1, 3))(rng_step, bij_params, bij_fns, deq_params, deq_fn, xobs)
        loss_grad = tree_util.tree_map(clip_and_zero_nans, loss_grad)
        opt_state = opt_update(it, loss_grad, opt_state)
        return opt_state, loss_val
    opt_state = opt_init((bij_params, deq_params))
    opt_state, trace = lax.scan(step, opt_state, jnp.arange(num_steps))
    bij_params, deq_params = get_params(opt_state)
    return (bij_params, deq_params), trace

def log_density_factory(data, target, scale):
    """Factory function for building a log-likelihood for the Procrustes problem
    given initial data and target points, and a (assumed known) noise scale.

    Args:
        data: Data on which to apply an orthogonal matrix so as to best match the
            target observations.
        target: Array of points to approximate.
        scale: Noise variance of the estimation problem.

    Returns:
        log_density: A function to compute the log-density given an orthogonal
            matrix input.

    """
    def log_density(xon):
        p = data@transp(xon)
        ll = -0.5 * jnp.square(target - p).sum(axis=(-1, -2)) / jnp.square(scale)
        return ll
    return log_density

rng = random.PRNGKey(args.seed)
rng, rng_data, rng_ortho, rng_noise = random.split(rng, 4)
rng, rng_haar = random.split(rng, 2)
rng, rng_deq, rng_bij = random.split(rng, 3)
rng, rng_train = random.split(rng, 2)
rng, rng_amb, rng_mse, rng_kl = random.split(rng, 4)

num_dims = 3
data = random.normal(rng_data, [10, num_dims])
O = pd.orthogonal.sample(rng_ortho, 1, num_dims)[0]
noise = args.noise_scale * random.normal(rng_noise, data.shape)
target = data@O.T + noise
log_density = log_density_factory(data, target, args.noise_scale)
U, _, VT = jnp.linalg.svd(data.T@target)
Oml = (U@VT).T

xhaar = pd.orthogonal.sample(rng_haar, 10000000, num_dims)
lprop = pd.orthogonal.logpdf(xhaar)
ld = log_density(xhaar)
lm = -lprop[0] + log_density(Oml)
la = ld - lprop - lm
logu = jnp.log(random.uniform(rng_haar, [len(xhaar)]))
xobs = xhaar[logu < la]
print('number of rejection samples: {}'.format(len(xobs)))
assert jnp.all(la < 0.)

# Generate parameters of the dequantization network.
deq_params, deq_fn = dequantization.network(rng_deq, num_dims, args.num_dequantization)

# Generate the parameters of RealNVP bijectors.
bij_params, bij_fns = [], []
num_dims_sq = num_dims**2
half_num_dims_sq = num_dims_sq // 2
num_masked = num_dims_sq - half_num_dims_sq
for i in range(args.num_realnvp):
    p, f = ambient.network_factory(random.fold_in(rng_bij, i), num_masked, half_num_dims_sq, args.num_ambient)
    bij_params.append(p)
    bij_fns.append(f)

# Train dequantization networks.
(bij_params, deq_params), trace = train(rng_train, bij_params, bij_fns, deq_params, deq_fn, args.num_steps, args.lr)

# Compute an estimate of the KL divergence.
num_is = 150
_, xon = ambient.sample(rng_mse, 10000, bij_params, bij_fns, num_dims)
xdeq, deq_log_dens = dequantization.dequantize(rng_kl, deq_params, deq_fn, xon, num_is)
amb_log_dens = vmap(ambient.log_prob, in_axes=(None, None, 0))(bij_params, bij_fns, xdeq)
log_approx = jspsp.logsumexp(amb_log_dens - deq_log_dens, axis=0) - jnp.log(num_is)
log_approx = jnp.clip(log_approx, -10., 10.)
log_target = log_density(xon)
approx, target = jnp.exp(log_approx), jnp.exp(log_target)
w = jnp.exp(log_target - log_approx)
Z = jnp.nanmean(w)
logZ = jspsp.logsumexp(log_target - log_approx) - jnp.log(len(xon))
kl = jnp.nanmean(log_approx - log_target) + logZ
ess = jnp.square(jnp.nansum(w)) / jnp.nansum(jnp.square(w))
ress = 100 * ess / len(w)
print('estimate KL(q||p): {:.5f} - relative effective sample size: {:.2f}%'.format(kl, ress))

# Construct visualization of the Procrustes problem.
pro = data@transp(xon)
target = data@O.T

fig = plt.figure(figsize=(10, 4))
ax = fig.add_subplot(121, projection='3d')
ax.set_title('Procrustes Posterior')
for i in range(1000):
    ax.plot(pro[i, :, 0], pro[i, :, 1], pro[i, :, 2], '.', alpha=0.05, color='tab:blue', label='Samples' if i == 0 else '_')

ax.plot(target[:, 0], target[:, 1], target[:, 2], '.', label='Target', color='tab:orange')
ax.legend()
ax.grid(linestyle=':')
lim = 4.
ax.set_xlim((-lim, lim))
ax.set_ylim((-lim, lim))
ax.set_zlim((-lim, lim))
ax = fig.add_subplot(122)
ax.plot(trace)
ax.grid(linestyle=':')
ax.set_title('ELBO Loss')
ax.set_xlabel('Number of Iterations')
plt.suptitle('KL$(q\Vert p)$ = {:.5f} - Rel. ESS: {:.2f}%'.format(kl, ress))
plt.subplots_adjust(top=0.85)
plt.savefig(os.path.join('images', 'procrustes.png'))
