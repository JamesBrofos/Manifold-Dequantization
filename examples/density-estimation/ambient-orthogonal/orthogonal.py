import argparse
from functools import partial
from typing import Callable, Sequence, Tuple

import jax.numpy as jnp
import jax.scipy.special as jspsp
import jax.scipy.stats as jspst
from jax import lax, nn, ops, random, tree_util
from jax import grad, jacobian, jit, value_and_grad, vmap
from jax.experimental import optimizers, stax

from jax.config import config
config.update("jax_enable_x64", True)

import prax.distributions as pd
import prax.manifolds as pm
from prax.bijectors import realnvp, permute

import ambient
import dequantization
import haaron
from polar import polar, transp, vecpolar


parser = argparse.ArgumentParser(description='Density estimation for the orthogonal group')
parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
parser.add_argument('--num-dims', type=int, default=3, help='Dimensionality of orthogonal group')
parser.add_argument('--num-batch', type=int, default=100, help='Number of samples per batch')
parser.add_argument('--num-realnvp', type=int, default=5, help='Number of RealNVP bijectors to employ')
parser.add_argument('--num-steps', type=int, default=100, help='Number of gradient descent iterations')
parser.add_argument('--seed', type=int, default=0, help='Pseudo-random number generator seed')
args = parser.parse_args()


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
    # return -amb_log_dens

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

@partial(jit, static_argnums=(2, 4))
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


# Set pseudo-random number generator keys.
rng = random.PRNGKey(args.seed)
rng, rng_deq, rng_bij = random.split(rng, 3)
rng, rng_haar, rng_amb = random.split(rng, 3)
rng, rng_mse = random.split(rng, 2)

# Generate parameters of the dequantization network.
deq_params, deq_fn = dequantization.network(rng_deq, args.num_dims, 64)

# Generate the parameters of RealNVP bijectors.
bij_params, bij_fns = [], []
num_dims_sq = args.num_dims**2
half_num_dims_sq = num_dims_sq // 2
num_masked = num_dims_sq - half_num_dims_sq
for i in range(args.num_realnvp):
    p, f = ambient.network_factory(random.fold_in(rng_bij, i), num_masked, half_num_dims_sq, 128)
    bij_params.append(p)
    bij_fns.append(f)

# Sample from a target distribution using rejection sampling
xhaar = haaron.sample(rng_haar, 200000, args.num_dims)
target = lambda x: -0.5 * jnp.square(x - jnp.eye(args.num_dims)).sum(axis=(-1, -2)) / 3.
lprop = haaron.logpdf(xhaar)
lm = -lprop[0]
la = target(xhaar) - lprop - lm
logu = jnp.log(random.uniform(rng_haar, [len(xhaar)]))
xobs = xhaar[logu < la]
print('number of rejection samples: {}'.format(len(xobs)))

opt_init, opt_update, get_params = optimizers.adam(args.lr)
opt_state = opt_init((bij_params, deq_params))

for it in range(args.num_steps):
    rng_step = random.fold_in(rng, it)
    bij_params, deq_params = get_params(opt_state)
    loss_val, loss_grad = value_and_grad(loss, (1, 3))(rng_step, bij_params, bij_fns, deq_params, deq_fn, xobs)
    loss_grad = tree_util.tree_map(clip_and_zero_nans, loss_grad)
    opt_state = opt_update(it, loss_grad, opt_state)
    print('iteration: {} - nelbo: {:.4f}'.format(it + 1, loss_val))

    if (it + 1) % 100 == 0 or (it + 1) == 1:
        _, xon = ambient.sample(rng_mse, 1000000, bij_params, bij_fns, args.num_dims)
        mean_mse = jnp.square(jnp.linalg.norm(xon.mean(0) - xobs.mean(0)))
        cov_mse = jnp.square(jnp.linalg.norm(jnp.cov(xon.reshape((-1, num_dims_sq)).T) - jnp.cov(xobs.reshape((-1, num_dims_sq)).T)))
        print('mean mse: {:.5f} - covariance mse: {:.5f}'.format(mean_mse, cov_mse))

