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
from polar import transp


parser = argparse.ArgumentParser(description='Density estimation for the orthogonal group')
parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
parser.add_argument('--num-batch', type=int, default=100, help='Number of samples per batch')
parser.add_argument('--num-realnvp', type=int, default=5, help='Number of RealNVP bijectors to employ')
parser.add_argument('--num-steps', type=int, default=20000, help='Number of gradient descent iterations')
parser.add_argument('--elbo-loss', type=int, default=1, help='Flag to indicate using the ELBO loss')
parser.add_argument('--num-importance', type=int, default=1, help='Number of importance samples to draw during training; if the ELBO loss is used, this argument is ignored')
parser.add_argument('--num-ambient', type=int, default=512, help='Number of hidden units in the ambient network')
parser.add_argument('--num-dequantization', type=int, default=128, help='Number of hidden units in dequantization network')
parser.add_argument('--noise-scale', type=float, default=1., help='Additive noise scale')
parser.add_argument('--seed', type=int, default=0, help='Pseudo-random number generator seed')
args = parser.parse_args()

def importance_log_density(rng: jnp.ndarray, bij_params: Sequence[jnp.ndarray], bij_fns: Sequence[Callable], deq_params: Sequence[jnp.ndarray], deq_fn: Callable, num_is: int, xon: jnp.ndarray) -> jnp.ndarray:
    """Use importance samples to estimate the log-density on the orthogonal group.

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
        xon: Observations on the orthogonal group.

    Returns:
        log_approx: The estimated log-density on the manifold, computed via
           importance sampling.

    """
    xdeq, deq_log_dens = dequantization.dequantize(rng, deq_params, deq_fn, xon, num_is)
    amb_log_dens = vmap(ambient.log_prob, in_axes=(None, None, 0))(bij_params, bij_fns, xdeq)
    log_approx = jspsp.logsumexp(amb_log_dens - deq_log_dens, axis=0) - jnp.log(num_is)
    return log_approx

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
    rng, rng_loss, rng_idx = random.split(rng, 3)
    idx = random.permutation(rng_idx, len(xon))[:100]
    xobs = xon[idx]
    if args.elbo_loss:
        nelbo = negative_elbo(rng_loss, bij_params, bij_fns, deq_params, deq_fn, xobs)
        nelbo = nelbo.mean()
        return nelbo
    else:
        log_is = importance_log_density(rng_loss, bij_params, bij_fns, deq_params, deq_fn, args.num_importance, xobs)
        log_target = log_density(xobs)
        return jnp.mean(log_target - log_is)


@partial(jit, static_argnums=(3, 4))
def train(rng: random.PRNGKey, bij_params: Sequence[jnp.ndarray], deq_params: Sequence[jnp.ndarray], num_steps: int, lr: float) -> Tuple:
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
rng, rng_haar, rng_acc = random.split(rng, 3)
rng, rng_deq, rng_bij = random.split(rng, 3)
rng, rng_train = random.split(rng, 2)
rng, rng_amb, rng_mse, rng_kl = random.split(rng, 4)

num_dims = 3
data = random.normal(rng_data, [10, num_dims])
O = pd.orthogonal.haar.rvs(rng_ortho, 1, num_dims)[0]
noise = args.noise_scale * random.normal(rng_noise, data.shape)
target = data@O.T + noise
log_density = log_density_factory(data, target, args.noise_scale)
U, _, VT = jnp.linalg.svd(data.T@target)
Oml = (U@VT).T

xhaar = pd.orthogonal.haar.rvs(rng_haar, 10000000, num_dims)
lprop = pd.orthogonal.haar.logpdf(xhaar)
ld = log_density(xhaar)
lm = -lprop[0] + log_density(Oml)
la = ld - lprop - lm
logu = jnp.log(random.uniform(rng_acc, [len(xhaar)]))
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

if not args.elbo_loss:
    print('rescaling initialization')
    bij_params = tree_util.tree_map(lambda x: x / 10., bij_params)
    deq_params = tree_util.tree_map(lambda x: x / 10., deq_params)

# Train dequantization networks.
(bij_params, deq_params), trace = train(rng_train, bij_params, deq_params, args.num_steps, args.lr)

# Compute an estimate of the KL divergence.
num_is = 150
_, xon = ambient.sample(rng_mse, 10000, bij_params, bij_fns, num_dims)
log_approx = importance_log_density(rng_kl, bij_params, bij_fns, deq_params, deq_fn, num_is, xon)
log_approx = jnp.clip(log_approx, -10., 10.)
log_target = log_density(xon)
approx, target = jnp.exp(log_approx), jnp.exp(log_target)
w = jnp.exp(log_target - log_approx)
Z = jnp.nanmean(w)
logZ = jspsp.logsumexp(log_target - log_approx) - jnp.log(len(xon))
klqp = jnp.nanmean(log_approx - log_target) + logZ
ess = jnp.square(jnp.nansum(w)) / jnp.nansum(jnp.square(w))
ress = 100 * ess / len(w)
del w, Z, logZ, log_approx, approx, log_target, target
xobs = xobs[:1000]
log_approx = importance_log_density(rng_kl, bij_params, bij_fns, deq_params, deq_fn, num_is, xobs)
log_approx = jnp.clip(log_approx, -10., 10.)
log_target = log_density(xobs)
approx, target = jnp.exp(log_approx), jnp.exp(log_target)
logZ = jspsp.logsumexp(log_approx - log_target) - jnp.log(len(xobs))
klpq = jnp.nanmean(log_target - log_approx) + logZ
del logZ, log_approx, approx, log_target, target
mean_mse = jnp.square(jnp.linalg.norm(xon.mean(0) - xobs.mean(0)))
cov_mse = jnp.square(jnp.linalg.norm(jnp.cov(xon.reshape((-1, num_dims_sq)).T) - jnp.cov(xobs.reshape((-1, num_dims_sq)).T)))
method = 'procrustes ({})'.format('ELBO' if args.elbo_loss else 'KL')
print('{} - Mean MSE: {:.5f} - Covariance MSE: {:.5f} - KL$(q\Vert p)$ = {:.5f} - KL$(p\Vert q)$ = {:.5f} - Rel. ESS: {:.2f}%'.format(method, mean_mse, cov_mse, klqp, klpq, ress))


# # Construct visualization of the Procrustes problem.
# pro = data@transp(xon)
# target = data@O.T

# fig = plt.figure(figsize=(10, 4))
# ax = fig.add_subplot(121, projection='3d')
# ax.set_title('Procrustes Posterior')
# for i in range(1000):
#     ax.plot(pro[i, :, 0], pro[i, :, 1], pro[i, :, 2], '.', alpha=0.05, color='tab:blue', label='Samples' if i == 0 else '_')

# ax.plot(target[:, 0], target[:, 1], target[:, 2], '.', label='Target', color='tab:orange')
# leg = ax.legend()
# for lh in leg.legendHandles:
#     lh._legmarker.set_alpha(1)

# ax.grid(linestyle=':')
# lim = 4.
# ax.set_xlim((-lim, lim))
# ax.set_ylim((-lim, lim))
# ax.set_zlim((-lim, lim))
# ax = fig.add_subplot(122)
# ax.plot(trace)
# ax.grid(linestyle=':')
# ax.set_title('ELBO Loss')
# ax.set_xlabel('Number of Iterations')
# plt.suptitle('KL$(q\Vert p)$ = {:.5f} - Rel. ESS: {:.2f}%'.format(klqp, ress))
# plt.subplots_adjust(top=0.85)
# plt.savefig(os.path.join('images', 'procrustes.png'))
