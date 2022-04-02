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
import prax.utils as put
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
parser.add_argument('--elbo-loss', type=int, default=1, help='Flag to indicate using the ELBO loss')
parser.add_argument('--num-importance', type=int, default=20, help='Number of importance samples to draw during training; if the ELBO loss is used, this argument is ignored')
parser.add_argument('--num-ambient', type=int, default=128, help='Number of hidden units in the ambient network')
parser.add_argument('--num-dequantization', type=int, default=64, help='Number of hidden units in dequantization network')
parser.add_argument('--seed', type=int, default=0, help='Pseudo-random number generator seed')
args = parser.parse_args()


log_dens = {
    'unimodal': log_unimodal,
    'multimodal': log_multimodal
}[args.density]

def importance_log_density(rng: jnp.ndarray, bij_params: Sequence[jnp.ndarray], bij_fns: Sequence[Callable], deq_params: Sequence[jnp.ndarray], deq_fn: Callable, num_is: int, xso: jnp.ndarray) -> jnp.ndarray:
    """Use importance samples to estimate the log-density on the special orthogonal
    group.

    Args:
        rng: Pseudo-random number generator seed.
        bij_params: List of arrays parameterizing the RealNVP bijectors.
        bij_fns: List of functions that compute the shift and scale of the RealNVP
            affine transformation.
        deq_params: Parameters of the mean and scale functions used in the
            log-normal dequantizer.
        deq_fn: Function that computes the mean and scale of the dequantization
            distribution.
        num_is: Number of importance samples.
        xso: Observations on the special orthogonal group.

    Returns:
        out: The estimated log-density on the manifold, computed via importance
           sampling.

    """
    approx = 0.
    for s in [-1, +1]:
        xdeq, deq_log_dens = dequantization.dequantize(random.fold_in(rng, s), deq_params, deq_fn, s*xso, num_is)
        amb_log_dens = vmap(ambient.log_prob, in_axes=(None, None, 0))(bij_params, bij_fns, xdeq)
        log_approx = jspsp.logsumexp(amb_log_dens - deq_log_dens, axis=0) - jnp.log(num_is)
        log_approx = jnp.clip(log_approx, -10., 10.)
        approx += jnp.exp(log_approx)
    return jnp.log(approx)

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
    rng, rng_loss, rng_idx = random.split(rng, 3)
    idx = random.permutation(rng_idx, len(xso))[:100]
    xobs = xso[idx]
    if args.elbo_loss:
        rng_elboa, rng_elbob = random.split(rng, 2)
        nelbo = 0.
        nelbo += 0.5 * negative_elbo(rng_elboa, bij_params, bij_fns, deq_params, deq_fn, xobs).mean()
        nelbo += 0.5 * negative_elbo(rng_elbob, bij_params, bij_fns, deq_params, deq_fn, -xobs).mean()
        return nelbo
    else:
        log_is = importance_log_density(rng_loss, bij_params, bij_fns, deq_params, deq_fn, args.num_importance, xobs)
        log_target = log_dens(xobs)
        return jnp.mean(log_target - log_is)


@partial(jit, static_argnums=(3, 4))
def train(rng: random.PRNGKey,
          bij_params: Sequence[jnp.ndarray],
          # bij_fns: Sequence[Callable],
          deq_params: Sequence[jnp.ndarray],
          # deq_fn: Callable,
          num_steps: int,
          lr: float) -> Tuple:
    opt_init, opt_update, get_params = optimizers.adam(lr)
    def step(opt_state, it):
        rng_step = random.fold_in(rng, it)
        bij_params, deq_params = get_params(opt_state)
        loss_val, loss_grad = value_and_grad(loss, (1, 3))(rng_step, bij_params, bij_fns, deq_params, deq_fn, xobs)
        loss_grad = tree_util.tree_map(partial(put.clip_and_zero_nans, clip_value=1.), loss_grad)
        opt_state = opt_update(it, loss_grad, opt_state)
        return opt_state, loss_val
    opt_state = opt_init((bij_params, deq_params))
    opt_state, trace = lax.scan(step, opt_state, jnp.arange(num_steps))
    bij_params, deq_params = get_params(opt_state)
    return (bij_params, deq_params), trace

# Set pseudo-random number generator keys.
rng = random.PRNGKey(args.seed)
rng, rng_deq, rng_bij = random.split(rng, 3)
rng, rng_haar, rng_acc, rng_amb = random.split(rng, 4)
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

if not args.elbo_loss:
    print('rescaling initialization')
    bij_params = tree_util.tree_map(lambda x: x / 10., bij_params)
    deq_params = tree_util.tree_map(lambda x: x / 10., deq_params)


# Sample from a target distribution using rejection sampling
xhaar = pd.orthogonal.haar.rvs(rng_haar, 5000000, args.num_dims)
xhaar = xhaar * jnp.linalg.det(xhaar)[..., jnp.newaxis, jnp.newaxis]
lprop = pd.orthogonal.haar.logpdf(xhaar) + jnp.log(2.0)
ld = log_dens(xhaar)
lm = ld.max() + 0.5
la = ld - lm
logu = jnp.log(random.uniform(rng_acc, [len(xhaar)]))
xobs = xhaar[logu < la]
print('number of rejection samples: {}'.format(len(xobs)))
assert jnp.all(la < 0.)

# Train dequantization networks.
(bij_params, deq_params), trace = train(rng_train, bij_params, deq_params, args.num_steps, args.lr)

# Sample from the ambient space and compare moments.
rng_sample_a, rng_sample_b = random.split(rng_sample, 2)
_, xon = ambient.sample(rng_sample_a, len(xobs), bij_params, bij_fns, args.num_dims)
xso = xon * jnp.linalg.det(xon)[..., jnp.newaxis, jnp.newaxis]
mean_mse = jnp.square(jnp.linalg.norm(xso.mean(0) - xobs.mean(0)))
cov_mse = jnp.square(jnp.linalg.norm(jnp.cov(xso.reshape((-1, num_dims_sq)).T) - jnp.cov(xobs.reshape((-1, num_dims_sq)).T)))

# Density estimation on SO(n).
num_is = 100
_, xon = ambient.sample(rng_sample_b, 10000, bij_params, bij_fns, args.num_dims)
xso = xon * jnp.linalg.det(xon)[..., jnp.newaxis, jnp.newaxis]

# Compute comparison statistics.
log_approx = importance_log_density(rng_lp, bij_params, bij_fns, deq_params, deq_fn, num_is, xso)
approx = jnp.exp(log_approx)
log_target = log_dens(xso) + jnp.log(2.)
target = jnp.exp(log_target)
w = target / approx
Z = jnp.nanmean(w)
klqp = jnp.nanmean(log_approx - log_target) + jnp.log(Z)
ess = jnp.square(jnp.nansum(w)) / jnp.nansum(jnp.square(w))
ress = 100 * ess / len(w)
del log_approx, approx, log_target, target, w, Z
log_approx = importance_log_density(rng_lp, bij_params, bij_fns, deq_params, deq_fn, num_is, xobs[:10000])
approx = jnp.exp(log_approx)
log_target = log_dens(xobs[:10000]) + jnp.log(2.)
target = jnp.exp(log_target)
w = approx / target
Z = jnp.nanmean(w)
klpq = jnp.nanmean(log_target - log_approx) + jnp.log(Z)
method = 'orthogonal ({})'.format('ELBO' if args.elbo_loss else 'KL')
print('{} - Mean MSE: {:.5f} - Covariance MSE: {:.5f} - KL$(q\Vert p)$ = {:.5f} - KL$(p\Vert q)$ = {:.5f} - Rel. ESS: {:.2f}%'.format(method, mean_mse, cov_mse, klqp, klpq, ress))

# # Visualize the action of the rejection and dequantization samples on a vector.
# vec = jnp.ones((args.num_dims, ))
# xsovec = xso@vec
# xobsvec = xobs@vec
# num_obs = 2000

# fig = plt.figure(figsize=(10, 4))
# ax = fig.add_subplot(121)
# ax.plot(trace, '-')
# ax.set_ylabel('ELBO Loss')
# ax.grid(linestyle=':')
# ax = fig.add_subplot(122, projection='3d')
# ax.plot(xobsvec[:num_obs, 0], xobsvec[:num_obs, 1], xobsvec[:num_obs, 2], '.', alpha=0.1, label='Rejection Samples')
# ax.plot(xsovec[:num_obs, 0], xsovec[:num_obs, 1], xsovec[:num_obs, 2], '.', alpha=0.1, label='Dequantization Samples')
# ax.grid(linestyle=':')
# leg = ax.legend()
# # for lh in leg.legendHandles:
# #     lh._legmarker.set_alpha(1)

# plt.tight_layout()
# plt.suptitle('Mean MSE: {:.5f} - Covariance MSE: {:.5f} - KL$(q\Vert p)$ = {:.5f} - Rel. ESS: {:.2f}%'.format(mean_mse, cov_mse, klqp, ress))
# plt.subplots_adjust(top=0.85)
# plt.savefig(os.path.join('images', 'orthogonal-marginal-{}.png'.format(args.density)))
