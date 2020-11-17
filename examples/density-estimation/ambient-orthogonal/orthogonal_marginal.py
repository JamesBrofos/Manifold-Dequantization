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
parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
parser.add_argument('--num-dims', type=int, default=3, help='Dimensionality of orthogonal group')
parser.add_argument('--num-batch', type=int, default=100, help='Number of samples per batch')
parser.add_argument('--num-realnvp', type=int, default=5, help='Number of RealNVP bijectors to employ')
parser.add_argument('--num-steps', type=int, default=100, help='Number of gradient descent iterations')
parser.add_argument('--density', type=str, default='unimodal', help='Which density on O(n) to sample')
parser.add_argument('--num-ambient', type=int, default=128, help='Number of hidden units in the ambient network')
parser.add_argument('--num-dequantization', type=int, default=64, help='Number of hidden units in dequantization network')
parser.add_argument('--seed', type=int, default=0, help='Pseudo-random number generator seed')
args = parser.parse_args()


log_dens = {
    'unimodal': log_unimodal,
    'multimodal': log_multimodal
}[args.density]

def l2_squared(pytree):
    """Squared L2-norm penalization term."""
    leaves, _ = tree_util.tree_flatten(pytree)
    return sum(jnp.vdot(x, x) for x in leaves)

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

def loss(rng: jnp.ndarray, bij_params: Sequence[jnp.ndarray], bij_fns: Sequence[Callable], deq_params: Sequence[jnp.ndarray], deq_fn: Callable, xso: jnp.ndarray) -> float:
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
        xso: Observations on SO(n).

    Returns:
        nelbo: The negative evidence lower bound.

    """
    rng, rng_rej, rng_elbo, rng_deq = random.split(rng, 4)
    rng_elboa, rng_elbob = random.split(rng, 2)
    nelbo = 0.
    nelbo += 0.5 * negative_elbo(rng_elboa, bij_params, bij_fns, deq_params, deq_fn, xso).mean()
    nelbo += 0.5 * negative_elbo(rng_elbob, bij_params, bij_fns, deq_params, deq_fn, -xso).mean()
    return nelbo

@partial(jit, static_argnums=(2, 4, 5, 6))
def train(rng: random.PRNGKey,
          bij_params: Sequence[jnp.ndarray],
          bij_fns: Sequence[Callable],
          deq_params: Sequence[jnp.ndarray],
          deq_fn: Callable,
          num_steps: int,
          lr: float) -> Tuple:
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

# Set pseudo-random number generator keys.
rng = random.PRNGKey(args.seed)
rng, rng_deq, rng_bij = random.split(rng, 3)
rng, rng_haar, rng_amb = random.split(rng, 3)
rng, rng_mse, rng_kl = random.split(rng, 3)
rng, rng_train = random.split(rng, 2)
rng, rng_sample, rng_lp = random.split(rng, 3)

# Generate parameters of the dequantization network.
deq_params, deq_fn = dequantization.network(rng_deq, args.num_dims, args.num_dequantization)

# Generate the parameters of RealNVP bijectors.
bij_params, bij_fns = [], []
num_dims_sq = args.num_dims**2
half_num_dims_sq = num_dims_sq // 2
num_masked = num_dims_sq - half_num_dims_sq
for i in range(args.num_realnvp):
    p, f = ambient.network_factory(random.fold_in(rng_bij, i), num_masked, half_num_dims_sq, args.num_ambient)
    bij_params.append(p)
    bij_fns.append(f)

# Sample from a target distribution using rejection sampling
xhaar = pd.orthogonal.sample(rng_haar, 5000000, args.num_dims)
lprop = pd.orthogonal.logpdf(xhaar)
ld = log_dens(xhaar)
lm = -lprop[0] - ld.max() + 0.5
la = ld - lprop - lm
logu = jnp.log(random.uniform(rng_haar, [len(xhaar)]))
xobs = xhaar[logu < la]
print('number of rejection samples: {}'.format(len(xobs)))
assert jnp.all(la < 0.)

# Convert the samples on the orthogonal group into samples on SO(n).
xobs = xobs * jnp.linalg.det(xobs)[..., jnp.newaxis, jnp.newaxis]

# Train dequantization networks.
(bij_params, deq_params), trace = train(rng_train, bij_params, bij_fns, deq_params, deq_fn, args.num_steps, args.lr)

# Sample from the ambient space and compare moments.
_, xon = ambient.sample(rng_sample, len(xobs), bij_params, bij_fns, args.num_dims)
xso = xon * jnp.linalg.det(xon)[..., jnp.newaxis, jnp.newaxis]
mean_mse = jnp.square(jnp.linalg.norm(xso.mean(0) - xobs.mean(0)))
cov_mse = jnp.square(jnp.linalg.norm(jnp.cov(xso.reshape((-1, num_dims_sq)).T) - jnp.cov(xobs.reshape((-1, num_dims_sq)).T)))
print('mean mse: {:.5f} - covariance mse: {:.5f}'.format(mean_mse, cov_mse))

# Density estimation on SO(n).
num_is = 100
_, xon = ambient.sample(rng_sample, 10000, bij_params, bij_fns, args.num_dims)
xso = xon * jnp.linalg.det(xon)[..., jnp.newaxis, jnp.newaxis]
approx = 0.

for s in [-1, +1]:
    xdeq, deq_log_dens = dequantization.dequantize(random.fold_in(rng_lp, s), deq_params, deq_fn, s*xso, num_is)
    amb_log_dens = vmap(ambient.log_prob, in_axes=(None, None, 0))(bij_params, bij_fns, xdeq)
    log_approx = jspsp.logsumexp(amb_log_dens - deq_log_dens, axis=0) - jnp.log(num_is)
    log_approx = jnp.clip(log_approx, -10., 10.)
    approx += jnp.exp(log_approx)

log_approx = jnp.log(approx)
log_target = log_dens(xso) + jnp.log(2.)
target = jnp.exp(log_target)
w = target / approx
Z = jnp.nanmean(w)
kl = jnp.nanmean(log_approx - log_target) + jnp.log(Z)
ess = jnp.square(jnp.nansum(w)) / jnp.nansum(jnp.square(w))
ress = 100 * ess / len(w)
print('estimate KL(q||p): {:.5f} - relative effective sample size: {:.2f}%'.format(kl, ress))

# Visualize the action of the rejection and dequantization samples on a vector.
vec = jnp.ones((args.num_dims, ))
xsovec = xso@vec
xobsvec = xobs@vec
num_obs = 2000

fig = plt.figure(figsize=(10, 4))
ax = fig.add_subplot(121)
ax.plot(trace, '-')
ax.set_ylabel('ELBO Loss')
ax.grid(linestyle=':')
ax = fig.add_subplot(122, projection='3d')
ax.plot(xobsvec[:num_obs, 0], xobsvec[:num_obs, 1], xobsvec[:num_obs, 2], '.', alpha=0.1, label='Rejection Samples')
ax.plot(xsovec[:num_obs, 0], xsovec[:num_obs, 1], xsovec[:num_obs, 2], '.', alpha=0.1, label='Dequantization Samples')
ax.grid(linestyle=':')
leg = ax.legend()
for lh in leg.legendHandles:
    lh._legmarker.set_alpha(1)

plt.tight_layout()
plt.suptitle('Mean MSE: {:.5f} - Covariance MSE: {:.5f} - KL$(q\Vert p)$ = {:.5f} - Rel. ESS: {:.2f}%'.format(mean_mse, cov_mse, kl, ress))
plt.subplots_adjust(top=0.85)
plt.savefig(os.path.join('images', 'orthogonal-marginal-{}.png'.format(args.density)))