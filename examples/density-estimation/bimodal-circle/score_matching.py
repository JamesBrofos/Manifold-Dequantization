import argparse
import os
from functools import partial
from typing import Callable, Sequence, Tuple

import matplotlib.pyplot as plt

import jax.numpy as jnp
from jax import lax, ops, random, tree_util
from jax import jacobian, jit, value_and_grad, vmap
from jax.experimental import optimizers

import prax.distributions as pd
import prax.manifolds as pm

from ambient import sample_ambient
from density import log_importance_sample_density, mixture_density, log_mixture_density
from dequantization import negative_elbo
from network import network_factory

parser = argparse.ArgumentParser(description='Density estimation for power spherical mixture distribution')
parser.add_argument('--num-steps', type=int, default=1000, help='Number of gradient descent iterations for score matching training')
parser.add_argument('--lr', type=float, default=1e-3, help='Gradient descent learning rate')
parser.add_argument('--num-batch', type=int, default=100, help='Number of samples per batch')
parser.add_argument('--num-importance', type=int, default=50, help='Number of importance samples')
parser.add_argument('--sm-weight', type=float, default=1.0, help='Score matching weight to apply in loss')
parser.add_argument('--elbo-weight', type=float, default=1.0, help='ELBO weight to apply in loss')
parser.add_argument('--reg-weight', type=float, default=1.0, help='Regularization strength on weights of networks')
parser.add_argument('--seed', type=int, default=0, help='Pseudo-random number generator seed')
args = parser.parse_args()


def power_spherical_mixture(rng: jnp.ndarray, num_samples: int) -> jnp.ndarray:
    """Sample from the power spherical mixture distribution.

    Args:
        rng: Pseudo-random number generator seed.
        num_samples: Number of samples from the power spherical mixture.

    Returns:
        obs: Samples from the power spherical mixture distribution.

    """
    rng, rng_obsa, rng_obsb, rng_idx = random.split(rng, 4)
    obsa = pd.sphere.powsph(rng_obsa, kappa, muspha, [num_samples])
    obsb = pd.sphere.powsph(rng_obsb, kappa, musphb, [num_samples])
    idx = random.uniform(rng_idx, [num_samples]) < 0.5
    obs = jnp.where(idx[..., jnp.newaxis], obsa, obsb)
    return obs

def score_matching_loss(deq_params: Sequence[jnp.ndarray], bij_params:
                        Sequence[jnp.ndarray], num_samples: int, num_is: int,
                        rng: jnp.ndarray) -> float:
    """The score matching objective computes compares the gradients of a target
    log-density and an approximation log-density. The score matching loss is
    combined with a weighted ELBO loss.

    Args:
        deq_params: Parameters of the mean and scale functions used in
            the log-normal dequantizer.
        bij_params: List of arrays parameterizing the RealNVP bijectors.
        num_samples: Number of dequantization samples to compute.
        num_is: Number of importance samples.
        rng: Pseudo-random number generator seed.

    Returns:
        out: The sum of the score matching loss and the weighted ELBO loss.

    """
    rng, rng_amb, rng_unif, rng_is = random.split(rng, 4)
    _, xs = sample_ambient(rng_amb, num_samples, bij_params, bij_fns)
    xunif = pd.sphere.haarsph(random.PRNGKey(0), [num_samples, 2])
    # xunif = power_spherical_mixture(rng_amb, num_samples)
    # xs = xunif
    lmd = pm.sphere.sphgrad(log_mixture_density, xunif, kappa, muspha, musphb)
    lis = pm.sphere.sphgrad(log_importance_sample_density, xunif, num_is,
                            deq_params, deq_fn, bij_params, bij_fns, rng_is)
    nelbo = negative_elbo(deq_params, bij_params, deq_fn, bij_fns, rng, xs, 1)
    sm = jnp.mean(jnp.square(lmd - lis))
    bij_reg = jnp.sum(tree_util.tree_flatten(tree_util.tree_map(lambda b: jnp.square(b).sum(), bij_params))[0])
    deq_reg = jnp.sum(tree_util.tree_flatten(tree_util.tree_map(lambda b: jnp.square(b).sum(), deq_params))[0])
    amb_reg = jnp.sum(jnp.square(xs))
    return (
        args.sm_weight * sm +
        args.elbo_weight * nelbo +
        args.reg_weight * bij_reg + args.reg_weight * deq_reg)

def zero_nans(g):
    """Remove the NaNs in a matrix by replaceing them with zeros.

    Args:
        g: Matrix whose NaN elements should be replaced by zeros.

    Returns:
        out: The input matrix but with NaN elements replaced by zeros.

    """
    return jnp.where(jnp.isnan(g), jnp.zeros_like(g), g)

@partial(jit, static_argnums=(2, 5, 6))
def train(deq_params: Sequence[jnp.ndarray], bij_params: Sequence[jnp.ndarray],
          num_steps: int, lr: float, rng: jnp.ndarray, num_samples: int,
          num_is: int) -> Tuple:
    """Train ambient flow and dequantization distribution with score matching
    objective.

    Args:
        deq_params: Parameters of the mean and scale functions used in
            the log-normal dequantizer.
        bij_params: List of arrays parameterizing the RealNVP bijectors.
        num_steps: Number of gradient descent iterations on the score matching
            loss.
        lr: The gradient descent learning rate.
        rng: Pseudo-random number generator seed.
        num_samples: Number of dequantization samples to compute.
        num_is: Number of importance samples.

    Returns:
        out: A tuple containing the optimal parameters and a trace of the
            objective function.

    """
    opt_init, opt_update, get_params = optimizers.adam(lr)
    def step(opt_state, it):
        step_rng = random.fold_in(rng, it)
        deq_params, bij_params = get_params(opt_state)
        loss_val, loss_grad = value_and_grad(score_matching_loss, (0, 1))(deq_params, bij_params, num_samples, num_is, step_rng)
        loss_grad = tree_util.tree_map(zero_nans, loss_grad)
        opt_state = opt_update(it, loss_grad, opt_state)
        return opt_state, loss_val
    params = (deq_params, bij_params)
    opt_state, trace = lax.scan(step, opt_init(params), jnp.arange(num_steps))
    params = get_params(opt_state)
    return params, trace


# Set random number generation seeds.
rng = random.PRNGKey(args.seed)
rng, rng_bij, rng_obs, rng_deq = random.split(rng, 4)
rng, rng_score, rng_x, rng_is = random.split(rng, 4)
rng, rng_obs = random.split(rng, 2)

# Parameters of the power spherical mixture distribution.
kappa = 200.
muspha = jnp.array([0., 1.])
musphb = -muspha
musphb /= jnp.linalg.norm(musphb)

# Parameterize the mean and scale of a log-normal multiplicative dequantizer.
deq_params, deq_fn = network_factory(rng_deq, 2, 1)

# Generate the parameters of RealNVP bijectors.
bij_params, bij_fns = [], []
for i in range(5):
    p, f = network_factory(random.fold_in(rng_bij, i), 1, 1)
    bij_params.append(p)
    bij_fns.append(f)

# Optimize the score matching loss function.
params, trace = train(deq_params, bij_params, args.num_steps, args.lr,
                      rng_score, args.num_batch, args.num_importance)
deq_params, bij_params = params
del params
print(trace)

# Generate samples in the ambient space and on the sphere.
num_samples = 10000
num_importance = 100
xa, xs = sample_ambient(rng_x, num_samples, bij_params, bij_fns)

# Compute the density on the sphere from the analytical target density and
# using the importance sample approximation.
log_p_target = log_mixture_density(xs, kappa, muspha, musphb)
log_p_approx = log_importance_sample_density(xs, num_importance, deq_params,
                                             deq_fn, bij_params, bij_fns,
                                             rng_is)
p_target = jnp.exp(log_p_target)
p_approx = jnp.exp(log_p_approx)

# KL divergence.
isnan = jnp.logical_or(jnp.isnan(log_p_target), jnp.isnan(log_p_approx))
kl = jnp.mean(log_p_approx[~isnan] - log_p_target[~isnan])
print('KL(q || p): {:.5f}'.format(kl))

# Check density.
base = jnp.array([1., 0.])
B = random.normal(rng, [2, 1])
Bp = jnp.concatenate((base[..., jnp.newaxis], B), axis=-1)
O = jnp.linalg.qr(Bp)[0][:, 1:]
f = lambda x: pm.sphere.logmap(base, x)@O
ec = vmap(f)(xs)
J = vmap(jacobian(f))(xs)
B = random.normal(rng, [len(xs), 2, 1])
Bp = jnp.concatenate((xs[..., jnp.newaxis], B), axis=-1)
E = jnp.linalg.qr(Bp)[0][..., 1:]
JE = J@E
det = jnp.sqrt(jnp.linalg.det(jnp.swapaxes(JE, -1, -2)@(JE)))
eprob = p_approx / det

plt.figure(figsize=(5, 5))
plt.hist(ec.ravel(), bins=50, density=True)
plt.plot(ec.ravel(), eprob, '.')
plt.xlim((-jnp.pi, jnp.pi))
plt.grid(linestyle=':')
plt.xlabel('Exponential Coordinates')
plt.ylabel('Probability Density')
plt.savefig(os.path.join('images', 'exponential-coordinates-probability-density.png'))


obs = power_spherical_mixture(rng_obs, len(xs))
fig, axes = plt.subplots(1, 3, figsize=(10, 4))
axes[0].plot(trace)
axes[0].grid(linestyle=':')
axes[0].set_xlabel('Gradient Descent Iteration')
axes[0].set_ylabel('Combined Loss')
axes[1].set_title('Analytic Samples')
axes[1].plot(obs[:, 0], obs[:, 1], '.', color='tab:blue', alpha=0.1, label='Mixture Samples')
axes[1].axis('square')
axes[1].grid(linestyle=':')
axes[1].set_xlim((-2., 2.))
axes[1].set_ylim((-2., 2.))
axes[2].set_title('Approximate Samples')
axes[2].plot(obs[:, 0], obs[:, 1], '.', color='tab:blue', alpha=0.1, label='Mixture Samples')
axes[2].plot(xs[:, 0], xs[:, 1], '.', color='tab:orange', alpha=0.1, label='Mixture Samples')
axes[2].plot(xa[:, 0], xa[:, 1], '.', color='tab:green', alpha=0.1, label='Ambient Samples')
axes[2].grid(linestyle=':')
axes[2].legend()
axes[2].axis('square')
axes[2].set_xlim((-2., 2.))
axes[2].set_ylim((-2., 2.))
plt.tight_layout()
plt.savefig(os.path.join('images', 'bimodal-samples.png'))




from ambient import ambient_flow_log_prob
from dequantization import dequantize

mu = jnp.array([[0., 2.], [0., -2.]])
sigma = jnp.array([jnp.eye(2), jnp.eye(2)])
weights = jnp.array([0.5, 0.5])


num_samples = 100
rng, rng_amb, rng_unif, rng_is = random.split(rng, 4)
_, y = sample_ambient(rng_amb, num_samples, bij_params, bij_fns)
x, ln, qcond = dequantize(rng, deq_params, deq_fn, y, 1)
pamb = ambient_flow_log_prob(bij_params, bij_fns, x)
