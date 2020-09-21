import argparse
import os
from functools import partial
from typing import Callable, Sequence, Tuple

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import jax.numpy as jnp
from jax import lax, random
from jax import jit, value_and_grad
from jax.experimental import optimizers

import prax.distributions as pd
import prax.manifolds as pm

from ambient import sample_ambient
from density import log_importance_sample_density, sphere_density, log_sphere_density
from dequantization import negative_elbo
from network import network_factory

parser = argparse.ArgumentParser(description='Density estimation for power spherical mixture distribution')
parser.add_argument('--num-steps', type=int, default=1000, help='Number of gradient descent iterations for score matching training')
parser.add_argument('--lr', type=float, default=1e-3, help='Gradient descent learning rate')
parser.add_argument('--num-batch', type=int, default=100, help='Number of samples per batch')
parser.add_argument('--num-importance', type=int, default=50, help='Number of importance samples')
parser.add_argument('--elbo-weight', type=float, default=1.0, help='ELBO weight to apply in score matching loss')
parser.add_argument('--seed', type=int, default=0, help='Pseudo-random number generator seed')
args = parser.parse_args()


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
    rng, rng_amb, rng_is = random.split(rng, 3)
    _, xs = sample_ambient(rng_amb, num_samples, bij_params, bij_fns)
    lmd = pm.sphere.sphgrad(log_sphere_density, xs, kappa, musph)
    lis = pm.sphere.sphgrad(log_importance_sample_density, xs, num_is,
                            deq_params, deq_fn, bij_params,
                            bij_fns, rng_is)
    nelbo = negative_elbo(deq_params, bij_params, deq_fn, bij_fns, rng, xs, 1)
    sm = jnp.mean(jnp.square(lmd - lis))
    return sm + args.elbo_weight * nelbo

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

# Parameters of the power spherical distribution.
kappa = 50.
musph = jnp.array([0., 0., 1.])

# Parameterize the mean and scale of a log-normal multiplicative dequantizer.
deq_params, deq_fn = network_factory(rng_deq, 3, 1)

# Generate the parameters of RealNVP bijectors.
bij_params, bij_fns = [], []
for i in range(5):
    p, f = network_factory(random.fold_in(rng_bij, i), 1, 2)
    bij_params.append(p)
    bij_fns.append(f)

# Optimize the score matching loss function.
params, trace = train(deq_params, bij_params, args.num_steps, args.lr,
                      rng_score, args.num_batch, args.num_importance)
deq_params, bij_params = params
del params

# Generate samples in the ambient space and on the sphere.
num_samples = 20000
xa, xs = sample_ambient(rng_x, num_samples, bij_params, bij_fns)
# Generate samples from the target distribution for later comparison.
obs = pd.sphere.powsph(rng_obs, kappa, musph, [len(xs)])

# Compute the density on the sphere from the analytical target density and
# using the importance sample approximation.
log_p_target = log_sphere_density(xs, kappa, musph)
log_p_approx = log_importance_sample_density(
    xs, args.num_importance, deq_params, deq_fn, bij_params,
    bij_fns, rng_is)
p_target = jnp.exp(log_p_target)
p_approx = jnp.exp(log_p_approx)

# KL divergence.
kl = jnp.mean(log_p_approx - log_p_target)
print('kl-divergence: {:.5f}'.format(kl))

# Visualize approximate and target densities.
fig = plt.figure(figsize=(13, 4))
ax = fig.add_subplot(131)
ax.plot(trace)
ax.set_ylim((-1., 10))
ax.grid(linestyle=':')
ax.set_xlabel('Gradient Descent')
ax.set_ylabel('Combined Loss (Score Matching + ELBO)')
ax = fig.add_subplot(132, projection='3d')
ax.set_title('Analytic Density')
m = ax.scatter(xs[:, 0], xs[:, 1], xs[:, 2], c=p_target, vmin=0., vmax=4.2)
ax.grid(linestyle=':')
lim = 1.1
ax.set_xlim((-lim, lim))
ax.set_ylim((-lim, lim))
ax.set_zlim((-lim, lim))
ax.view_init(30, 140)
ax = fig.add_subplot(133, projection='3d')
ax.set_title('Approximate Density')
m = ax.scatter(xs[:, 0], xs[:, 1], xs[:, 2], c=p_approx, vmin=0., vmax=4.2)
ax.grid(linestyle=':')
ax.set_xlim((-lim, lim))
ax.set_ylim((-lim, lim))
ax.set_zlim((-lim, lim))
ax.view_init(30, 140)
plt.tight_layout()
plt.savefig(os.path.join('images', 'power-spherical-mixture-score-matching-elbo-weight-{}.png'.format(args.elbo_weight)))

# Compare samples from an actual power spherical distribution and the
# approximation.
fig = plt.figure(figsize=(13, 4))
ax = fig.add_subplot(131, projection='3d')
ax.set_title('Power Spherical Samples')
m = ax.scatter(obs[:, 0], obs[:, 1], obs[:, 2], color='tab:blue')
ax.grid(linestyle=':')
lim = 1.1
ax.set_xlim((-lim, lim))
ax.set_ylim((-lim, lim))
ax.set_zlim((-lim, lim))
ax.view_init(30, 140)
ax = fig.add_subplot(132, projection='3d')
ax.set_title('Approximate Samples')
m = ax.scatter(xs[:, 0], xs[:, 1], xs[:, 2], color='tab:orange')
ax.grid(linestyle=':')
ax.set_xlim((-lim, lim))
ax.set_ylim((-lim, lim))
ax.set_zlim((-lim, lim))
ax.view_init(30, 140)
ax = fig.add_subplot(133, projection='3d')
ax.set_title('Overlay Samples')
m = ax.scatter(obs[:, 0], obs[:, 1], obs[:, 2], color='tab:blue', alpha=0.2)
m = ax.scatter(xs[:, 0], xs[:, 1], xs[:, 2], color='tab:orange', alpha=0.2)
ax.grid(linestyle=':')
ax.set_xlim((-lim, lim))
ax.set_ylim((-lim, lim))
ax.set_zlim((-lim, lim))
ax.view_init(30, 140)
plt.tight_layout()
plt.savefig(os.path.join('images', 'power-spherical-samples.png'))
