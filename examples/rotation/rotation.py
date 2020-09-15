import argparse
import os
from functools import partial
from typing import Callable, Sequence, Tuple

import matplotlib.pyplot as plt

import jax.numpy as jnp
import jax.scipy.special as jspsp
import jax.scipy.stats as jspst
from jax import lax, random, tree_util
from jax import jit, value_and_grad, vmap
from jax.experimental import optimizers

import prax.manifolds as pm

from ambient import ambient_flow_log_prob, sample_ambient
from dequantization import dequantize, negative_elbo
from network import network_factory
from rodrigues import rodrigues


parser = argparse.ArgumentParser(description='Density estimation for rotation distribution')
parser.add_argument('--num-steps', type=int, default=1000, help='Number of gradient descent iterations for score matching training')
parser.add_argument('--lr', type=float, default=1e-3, help='Gradient descent learning rate')
parser.add_argument('--num-batch', type=int, default=100, help='Number of samples per batch')
parser.add_argument('--num-importance', type=int, default=50, help='Number of importance samples')
parser.add_argument('--elbo-weight', type=float, default=1.0, help='ELBO weight to apply in score matching loss')
parser.add_argument('--seed', type=int, default=0, help='Pseudo-random number generator seed')
args = parser.parse_args()

def log_importance_sample_density(axis_and_angle: jnp.ndarray, num_is: int,
                                  deq_params: Sequence[jnp.ndarray], deq_fn:
                                  Callable, bij_params: Sequence[jnp.ndarray],
                                  bij_fns: Callable, rng: jnp.ndarray) -> jnp.ndarray:
    """Use importance samples to estimate the log-density on the sphere.

    Args:
        axis_and_angle: Concatenation of axis and angle vectors.
        num_is: Number of importance samples.
        deq_params: Parameters of the mean and scale functions used in the
            log-normal dequantizer.
        deq_fn: Function that computes the mean and scale given its
            parameterization and input.
        bij_params: List of arrays parameterizing the RealNVP bijectors.
        bij_fns: List of functions that compute the shift and scale of the RealNVP
            affine transformation.
        rng: Pseudo-random number generator seed.

    Returns:
        log_isdens: The estimated log-density on the manifold, computed via
            importance sampling.

    """
    axis, angle = axis_and_angle[..., :3], axis_and_angle[..., 3:]
    (x_axis, x_angle), qcond = dequantize(rng, deq_params, deq_fn, axis, angle, num_is)
    x = jnp.concatenate((x_axis, x_angle), axis=-1)
    pamb = ambient_flow_log_prob(bij_params, bij_fns, x)
    log_isdens = jspsp.logsumexp(pamb - qcond, axis=0) - jnp.log(num_is)
    return log_isdens

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
    _, x_axis, x_angle = sample_ambient(rng_amb, num_samples, bij_params, bij_fns)
    axis_and_angle = jnp.concatenate((x_axis, x_angle), axis=-1)
    lmd = pm.sphere.sphgrad(log_density, axis_and_angle, obs, y, scale)
    lis = pm.sphere.sphgrad(log_importance_sample_density, axis_and_angle,
                            num_is, deq_params, deq_fn, bij_params, bij_fns,
                            rng_is)
    nelbo = negative_elbo(deq_params, bij_params, deq_fn, bij_fns, rng, x_axis, x_angle, 1)
    sm = jnp.mean(jnp.square(lmd - lis))
    return sm + args.elbo_weight * nelbo

def log_density(axis_and_angle, obs, y, scale):
    """Log-density describing the likelihood of normal observations given a
    rotation matrix.

    Args:
        axis_and_angle: Concatenation of axis and angle parameterization of the
            rotation matrix.
        obs: Observations to rotate toward the targets.
        y: Target observations.
        scale: Scale parameter measuring uncertainty about the means of the
            rotation.

    Returns:
        out: The log-likelihood of the rotation given the target observations.

    """
    axis, angle = axis_and_angle[..., :3], axis_and_angle[..., 3:]
    R = rodrigues(axis, angle)
    yh = obs@R.T
    return -0.5 * jnp.square(yh - y).sum((-1, -2))

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


# Set random seeds.
rng = random.PRNGKey(args.seed)
rng, rng_obs, rng_noise = random.split(rng, 3)
rng, rng_bij, rng_deq = random.split(rng, 3)
rng, rng_score, rng_is = random.split(rng, 3)
rng, rng_amb = random.split(rng, 2)

# Generate data for Procrustes problem.
theta = jnp.pi / 4
scale = 0.1
R = jnp.array([[1.,              0.,             0.],
               [0.,  jnp.cos(theta), jnp.sin(theta)],
               [0., -jnp.sin(theta), jnp.cos(theta)]])
obs = random.normal(rng_obs, [100, 3]) + jnp.ones(3)
noise = scale * random.normal(rng_noise, obs.shape)
y = obs@R.T + noise

# Parameterize the mean and scale of a log-normal multiplicative dequantizer.
deq_params, deq_fn = network_factory(rng_deq, 5, 2)

# Generate the parameters of RealNVP bijectors.
bij_params, bij_fns = [], []
for i in range(5):
    p, f = network_factory(random.fold_in(rng_bij, i), 2, 3)
    bij_params.append(p)
    bij_fns.append(f)

num_samples = 20
num_is = 50
# Optimize the score matching loss function.
params, trace = train(deq_params, bij_params, args.num_steps, args.lr,
                      rng_score, args.num_batch, args.num_importance)
deq_params, bij_params = params
del params
print(trace)

fig, axes = plt.subplots(1, 1)
axes = [axes]
axes[0].semilogy(trace)
axes[0].set_xlabel('Gradient Descent Iterations')
axes[0].set_ylabel('Combined Loss Function')
axes[0].grid(linestyle=':')
plt.tight_layout()
plt.savefig(os.path.join('images', 'loss.png'))

_, x_axis, x_angle = sample_ambient(rng_amb, num_samples, bij_params, bij_fns)
Rs = vmap(rodrigues, in_axes=(0, 0))(x_axis, x_angle)
