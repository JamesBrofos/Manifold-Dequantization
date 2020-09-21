import argparse
import os
from functools import partial
from typing import Callable, Sequence, Tuple

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import jax.numpy as jnp
import jax.scipy.stats as jspst
import jax.scipy.special as jspsp
from jax import lax, random, tree_util
from jax import grad, jacobian, jit, value_and_grad, vmap
from jax.experimental import optimizers

import prax.distributions as pd

from ambient import ambient_flow_log_prob, sample_ambient
from network import network_factory


parser = argparse.ArgumentParser(description='Density estimation for power spherical mixture distribution')
parser.add_argument('--num-steps', type=int, default=1000, help='Number of gradient descent iterations for score matching training')
parser.add_argument('--lr', type=float, default=1e-3, help='Gradient descent learning rate')
parser.add_argument('--num-batch', type=int, default=100, help='Number of samples per batch')
parser.add_argument('--sm-weight', type=float, default=1.0, help='Score matching weight to apply in loss')
parser.add_argument('--elbo-weight', type=float, default=1.0, help='ELBO weight to apply in loss')
parser.add_argument('--bias', type=float, default=0., help='Add bias to the log-density to simulate unknown normalization constant')
parser.add_argument('--seed', type=int, default=0, help='Pseudo-random number generator seed')
args = parser.parse_args()


def sample_mixture(rng: jnp.ndarray, num_samples: int, kappa: float, muspha:
                   jnp.ndarray, musphb: jnp.ndarray) -> jnp.ndarray:
    """Sample from the power spherical mixture distribution.

    Args:
        rng: Pseudo-random number generator seed.
        num_samples: Number of samples from the power spherical mixture.
        kappa: Concentration parameter of both modes
        muspha: Mean parameter of first mode.
        musphb: Mean parameter of second mode.

    Returns:
        obs: Samples from the power spherical mixture distribution.

    """
    rng, rng_obsa, rng_obsb, rng_idx = random.split(rng, 4)
    obsa = pd.sphere.powsph(rng_obsa, kappa, muspha, [num_samples])
    obsb = pd.sphere.powsph(rng_obsb, kappa, musphb, [num_samples])
    idx = random.uniform(rng_idx, [num_samples]) < 0.5
    obs = jnp.where(idx[..., jnp.newaxis], obsa, obsb)
    return obs

def log_mixture_density(obs: jnp.ndarray, kappa: float, muspha: jnp.ndarray,
                        musphb: jnp.ndarray) -> jnp.ndarray:
    """Compute the log-density of the power spherical mixture distribution.

    Args:
        obs: Points on the sphere at which to compute the mixture density.
        kappa: Concentration parameter of both modes
        muspha: Mean parameter of first mode.
        musphb: Mean parameter of second mode.

    Returns:
        out: The log-density of the power spherical mixture distribution.

    """
    densa = pd.sphere.powsphlogdensity(obs, kappa, muspha)
    densb = pd.sphere.powsphlogdensity(obs, kappa, musphb)
    return jspsp.logsumexp(jnp.array([densa, densb]), axis=0) - jnp.log(2.) + args.bias

def ambient_dequantization_density(x: jnp.ndarray, kappa: float, muspha:
                                   jnp.ndarray, musphb: jnp.ndarray) -> jnp.ndarray:
    """Compute the ambient density of the dequantization. The density on the sphere
    defines the spherical component of hyperspherical coordinates and the
    dequantization distribution (log-normal) defines the radius. When computing
    the density in Euclidean space, this introduces a Jacobian correction.

    """
    num_dims = x.shape[-1]
    rad = jnp.linalg.norm(x, axis=-1)
    sph = x / rad[..., jnp.newaxis]
    log_dens_sph = log_mixture_density(sph, kappa, muspha, musphb)
    log_dens_rad = pd.lognormal.logpdf(rad, 0.5, 0.1)
    ldj = -(num_dims - 1) * jnp.log(rad)
    return log_dens_sph + log_dens_rad + ldj

def dequantize(rng: jnp.ndarray, xsph: jnp.ndarray, num_samples: int) -> Tuple:
    """Dequantize samples on the sphere into an ambient Euclidean space.

    Args:
        rng: Pseudo-random number generator seed.
        xsph: Samples on the sphere.
        num_samples: Number of dequantization samples.

    Returns:
        out: A tuple containing the dequantized samples and the log-probability
            of the samples in the ambient space.

    """
    num_dims = xsph.shape[-1]
    rad = pd.lognormal.sample(rng, 0.5, 0.1, [num_samples] + list(xsph.shape[:-1]))
    ldj = -(num_dims - 1) * jnp.log(rad)
    log_dens_rad = pd.lognormal.logpdf(rad, 0.5, 0.1) + ldj
    xdeq = rad[..., jnp.newaxis] * xsph
    return xdeq, log_dens_rad

def negative_elbo(bij_params: Sequence[jnp.ndarray], bij_fns:
                  Sequence[Callable], rng: jnp.ndarray, xsph: jnp.ndarray) -> jnp.ndarray:
    """Compute the negative evidence lower bound of the dequantizing distribution.
    This is the loss function for learning parameters of the dequantizing
    distribution.

    Args:
        bij_params: List of arrays parameterizing the RealNVP bijectors.
        bij_fns: List of functions that compute the shift and scale of the RealNVP
            affine transformation.
        rng: Pseudo-random number generator seed.
        xsph: Observations on the sphere to dequantize.
        num_samples: Number of dequantization samples to compute.

    Returns:
        nelbo: The negative evidence lower bound for each example.

    """
    xdeq, logqcond = dequantize(rng, xsph, 1)
    logpamb = ambient_flow_log_prob(bij_params, bij_fns, xdeq)
    elbo = jnp.mean(logpamb - logqcond, axis=0)
    nelbo = -elbo
    return nelbo

def score_matching_loss(bij_params: Sequence[jnp.ndarray], num_samples: int,
                        rng: jnp.ndarray) -> float:
    """The score matching objective computes compares the gradients of a target
    log-density and an approximation log-density. The evidence lower bound
    objective computes the a lower bound on the probability of an observation
    on the manifold.

    Args:
        bij_params: List of arrays parameterizing the RealNVP bijectors.
        num_samples: Number of samples from the target distribution to sample.
        rng: Pseudo-random number generator seed.

    Returns:
        out: The weighted loss function.

    """
    rng, rng_deq, rng_sph = random.split(rng, 3)
    rng, rng_nelbo, rng_nll = random.split(rng, 3)
    # rng_sph = random.PRNGKey(0)
    xsph = sample_mixture(rng_sph, num_samples, kappa, muspha, musphb)
    xdeq, _ = dequantize(rng_deq, xsph, 1)
    xdeq = xdeq[0]
    deq_grad = vmap(grad(ambient_dequantization_density), in_axes=(0, None, None, None))(xdeq, kappa, muspha, musphb)
    amb_grad = vmap(grad(ambient_flow_log_prob, 2), in_axes=(None, None, 0))(bij_params, bij_fns, xdeq)
    sm = jnp.mean(jnp.square(deq_grad - amb_grad))
    nelbo = negative_elbo(bij_params, bij_fns, rng_nelbo, xsph).mean()
    return args.sm_weight * sm + args.elbo_weight * nelbo

def zero_nans(g):
    """Remove the NaNs in a matrix by replaceing them with zeros.

    Args:
        g: Matrix whose NaN elements should be replaced by zeros.

    Returns:
        out: The input matrix but with NaN elements replaced by zeros.

    """
    return jnp.where(jnp.isnan(g), jnp.zeros_like(g), g)

@partial(jit, static_argnums=(1, 4))
def train(bij_params: Sequence[jnp.ndarray], num_steps: int, lr: float, rng:
          jnp.ndarray, num_samples: int) -> Tuple:
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
        bij_params = get_params(opt_state)
        loss_val, loss_grad = value_and_grad(score_matching_loss, 0)(bij_params, num_samples, step_rng)
        loss_grad = tree_util.tree_map(zero_nans, loss_grad)
        opt_state = opt_update(it, loss_grad, opt_state)
        return opt_state, loss_val
    opt_state, trace = lax.scan(step, opt_init(bij_params), jnp.arange(num_steps))
    params = get_params(opt_state)
    return params, trace


# Set random number generation seeds.
rng = random.PRNGKey(args.seed)
rng, rng_xsph, rng_deq, rng_amb = random.split(rng, 4)
rng, rng_bij = random.split(rng, 2)
rng, rng_score = random.split(rng, 2)

# Parameters of the power spherical mixture distribution.
kappa = 50.
muspha = jnp.array([0., 0., 1.])
musphb = jnp.array([0., -1., 0.])

if args.bias == 0. and len(muspha) == 2:
    # Sample from the power spherical mixture.
    xsph = sample_mixture(rng_xsph, 1000000, kappa, muspha, musphb)
    xdeq, _ = dequantize(rng_deq, xsph, 1)
    xdeq = xdeq[0]
    logprob = ambient_dequantization_density(xdeq, kappa, muspha, musphb)
    logprob = logprob[~jnp.isnan(logprob)]

    # Verify that the ambient dequantization density is in agreement with
    # observed counts in small regions of space.
    prob = jnp.exp(logprob)
    delta = 0.05
    for i in range(50):
        p = xdeq[i]
        pr = prob[i] * delta**2
        idx0 = jnp.abs(xdeq[:, 0] - p[0]) < delta / 2.
        idx1 = jnp.abs(xdeq[:, 1] - p[1]) < delta / 2.
        pr_est = jnp.mean(jnp.logical_and(idx0, idx1))
        print('prob.: {:.10f} - estim. prob.: {:.10f}'.format(pr, pr_est))

# Generate the parameters of RealNVP bijectors.
bij_params, bij_fns = [], []
for i in range(5):
    p, f = network_factory(random.fold_in(rng_bij, i), 1, len(muspha) - 1)
    bij_params.append(p)
    bij_fns.append(f)

# Optimize the score matching loss function.
bij_params, trace = train(bij_params, args.num_steps, args.lr, rng_score,
                      args.num_batch)

# Compute KL(p||q).
num_is = 100
num_samples = 10000
xsph = sample_mixture(rng_xsph, num_samples, kappa, muspha, musphb)
xdeq, logisw = dequantize(rng_deq, xsph, num_is)
logpamb = ambient_flow_log_prob(bij_params, bij_fns, xdeq)
logis = jspsp.logsumexp(logpamb - logisw, axis=0) - jnp.log(num_is)
logtarget = log_mixture_density(xsph, kappa, muspha, musphb)
klpq = jnp.mean(logtarget - logis)

# Compute KL(q||p).
_, xsph = sample_ambient(rng_amb, num_samples, bij_params, bij_fns, len(muspha))
xdeq, logisw = dequantize(rng_deq, xsph, num_is)
logpamb = ambient_flow_log_prob(bij_params, bij_fns, xdeq)
logis = jspsp.logsumexp(logpamb - logisw, axis=0) - jnp.log(num_is)
logtarget = log_mixture_density(xsph, kappa, muspha, musphb)
klqp = jnp.mean(logis - logtarget)

del num_is, xsph, xdeq, logisw, logpamb, logis, logtarget

# Visualize.
xsph = sample_mixture(rng_xsph, num_samples, kappa, muspha, musphb)
xamb, xambsph = sample_ambient(rng_amb, num_samples, bij_params, bij_fns, len(muspha))
xdeq, _ = dequantize(rng_deq, xsph, 1)
xdeq = xdeq[0]

if len(muspha) == 2:
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].plot(trace)
    axes[0].grid(linestyle=':')
    axes[0].set_xlabel('Gradient Descent Iteration')
    axes[0].set_ylabel('Score Matching Loss')
    axes[1].plot(xsph[:, 0], xsph[:, 1], '.', alpha=0.01)
    axes[1].plot(xdeq[:, 0], xdeq[:, 1], '.', alpha=0.01)
    axes[1].plot(xamb[:, 0], xamb[:, 1], '.', alpha=0.01)
    axes[1].grid(linestyle=':')
    axes[1].set_xlim(-3., 3.)
    axes[1].set_ylim(-3., 3.)
    plt.tight_layout()
    plt.savefig(os.path.join('images', 'dequantized-samples-elbo-weight-{}-sm-weight-{}.png'.format(args.elbo_weight, args.sm_weight)))
else:
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(121)
    ax.plot(trace)
    ax.grid(linestyle=':')
    ax.set_ylabel('Loss')
    ax.set_xlabel('Gradient Descent Iteration')
    ax = fig.add_subplot(122, projection='3d')
    ax.plot(xsph[:, 0], xsph[:, 1], xsph[:, 2], '.', alpha=0.01)
    ax.plot(xdeq[:, 0], xdeq[:, 1], xdeq[:, 2], '.', alpha=0.01)
    ax.plot(xamb[:, 0], xamb[:, 1], xamb[:, 2], '.', alpha=0.01)
    ax.grid(linestyle=':')
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_zlim(-1.5, 1.5)
    plt.suptitle('KL$(p \Vert q)$ = {:.5f}\nKL$(q\Vert p)$ = {:.5f}'.format(klpq, klqp))
    plt.tight_layout()
    plt.savefig(os.path.join('images', 'dequantized-samples-elbo-weight-{}-sm-weight-{}.png'.format(args.elbo_weight, args.sm_weight)))
