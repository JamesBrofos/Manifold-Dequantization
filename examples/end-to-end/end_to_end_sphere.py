import argparse
import os
from functools import partial
from typing import Callable, Sequence, Tuple

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import tqdm

import jax.numpy as jnp
import jax.scipy.stats as jspst
from jax import lax, random
from jax import jit, value_and_grad
from jax.experimental import optimizers, stax

from prax.bijectors import realnvp, permute
from prax.distributions import lognormal, sphere


def network_factory(rng: jnp.ndarray, num_in: int, num_out: int) -> Tuple:
    """Factory for producing neural networks and their parameterizations.

    Args:
        rng: Pseudo-random number generator seed.
        num_in: Number of inputs to the network.
        num_out: Number of variables to transform by an affine transformation.
            Each variable receives an associated shift and scale.

    Returns:
        out: A tuple containing the network parameters and a callable function
            that returns the neural network output for the given input.

    """
    params_init, fn = stax.serial(
        stax.Dense(512), stax.Relu,
        stax.Dense(512), stax.Relu,
        stax.FanOut(2),
        stax.parallel(stax.Dense(num_out),
                      stax.serial(stax.Dense(num_out), stax.Softplus)))
    _, params = params_init(rng, (-1, num_in))
    return params, fn

def forward(params: Sequence[jnp.ndarray], fns: Sequence[Callable], x:
            jnp.ndarray) -> jnp.ndarray:
    """Forward transformation of composining RealNVP bijectors and a permutation
    bijector between them.

    Args:
        params: List of arrays parameterizing the RealNVP bijectors.
        fns: List of functions that compute the shift and scale of the RealNVP
            affine transformation.
        x: Input to transform according to the composition of RealNVP
            transformations and permutations.

    Returns:
        y: The transformed input.

    """
    y = realnvp.forward(x, 1, params[0], fns[0])
    y = permute.forward(y, jnp.array([2, 0, 1]))
    y = realnvp.forward(y, 1, params[1], fns[1])
    y = permute.forward(y, jnp.array([1, 2, 0]))
    y = realnvp.forward(y, 1, params[2], fns[2])
    y = permute.forward(y, jnp.array([2, 1, 0]))
    y = realnvp.forward(y, 1, params[3], fns[3])
    y = permute.forward(y, jnp.array([1, 0, 2]))
    y = realnvp.forward(y, 1, params[4], fns[4])
    return y

def ambient_flow_log_prob(params: Sequence[jnp.ndarray], fns:
                          Sequence[Callable], y: jnp.ndarray) -> jnp.ndarray:
    """Compute the log-probability of ambient observations under the transformation
    given by composing RealNVP bijectors and a permutation bijector between
    them. Assumes that the base distribution is a standard multivariate normal.

    Args:
        params: List of arrays parameterizing the RealNVP bijectors.
        fns: List of functions that compute the shift and scale of the RealNVP
            affine transformation.
        y: Observations whose likelihood under the composition of bijectors
            should be computed.

    Returns:
        out: The log-probability of the observations given the parameters of the
            bijection composition.

    """
    fldj = 0.
    y = realnvp.inverse(y, 1, params[4], fns[4])
    fldj += realnvp.forward_log_det_jacobian(y, 1, params[4], fns[4])
    y = permute.inverse(y, jnp.array([1, 0, 2]))
    fldj += permute.forward_log_det_jacobian()
    y = realnvp.inverse(y, 1, params[3], fns[3])
    fldj += realnvp.forward_log_det_jacobian(y, 1, params[3], fns[3])
    y = permute.inverse(y, jnp.array([2, 1, 0]))
    fldj += permute.forward_log_det_jacobian()
    y = realnvp.inverse(y, 1, params[2], fns[2])
    fldj += realnvp.forward_log_det_jacobian(y, 1, params[2], fns[2])
    y = permute.inverse(y, jnp.array([1, 2, 0]))
    fldj += permute.forward_log_det_jacobian()
    y = realnvp.inverse(y, 1, params[1], fns[1])
    fldj += realnvp.forward_log_det_jacobian(y, 1, params[1], fns[1])
    y = permute.inverse(y, jnp.array([2, 0, 1]))
    fldj += permute.forward_log_det_jacobian()
    y = realnvp.inverse(y, 1, params[0], fns[0])
    fldj += realnvp.forward_log_det_jacobian(y, 1, params[0], fns[0])
    return jspst.multivariate_normal.logpdf(y, jnp.zeros((3, )), 1.) - fldj

def dequantize(rng: jnp.ndarray, mu_and_sigma_params: Sequence[jnp.ndarray],
               mu_and_sigma_fn: Callable, y: jnp.ndarray) -> jnp.ndarray:
    """Dequantize the observations using a log-normal multiplicative
    dequantizer.

    Args:
        rng: Pseudo-random number generator seed.
        mu_and_sigma_params: Parameters of the mean and scale functions used in
            the log-normal dequantizer.
        mu_and_sigma_fn: Function that computes the mean and scale given its
            parameterization and input.
        y: Observations on the sphere to dequantize.

    Returns:
        x, qcond: A tuple containing observations that are dequantized according
            to multiplication by a log-normal random variable and the
            log-density of the conditional dequantizing distribution.

    """
    mu, sigma = mu_and_sigma_fn(mu_and_sigma_params, y)
    ln = lognormal.sample(rng, mu, sigma, mu.shape)
    x = (1. + ln) * y
    qcond = lognormal.logpdf(ln, mu, sigma).squeeze(-1)
    return x, qcond

def negative_elbo(mu_and_sigma_params: Sequence[jnp.ndarray], bij_params:
                  Sequence[jnp.ndarray], mu_and_sigma_fn: Callable, bij_fns:
                  Sequence[Callable], rng: jnp.ndarray, y: jnp.ndarray) -> float:
    """Compute the negative evidence lower bound of the dequantizing distribution.
    This is the loss function for learning parameters of the dequantizing
    distribution.

    """
    x, qcond = dequantize(rng, mu_and_sigma_params, mu_and_sigma_fn, y)
    pamb = ambient_flow_log_prob(bij_params, bij_fns, x)
    elbo = jnp.mean(pamb - qcond)
    nelbo = -elbo
    return nelbo

# Set random number generation seeds.
rng = random.PRNGKey(0)
rng, rng_bij, rng_y, rng_deq = random.split(rng, 4)
rng, rng_train = random.split(rng, 2)

# Generate random draws from the power spherical distribution.
kappa = 50.
muspha = jnp.array([0., 0., 1.])
musphb = jnp.array([-1., 0., 0.])
obs = jnp.vstack((sphere.powsph(random.fold_in(rng_y, 0), kappa, muspha, [10000]),
                  sphere.powsph(random.fold_in(rng_y, 1), kappa, musphb, [10000])))

# Parameterize the mean and scale of a log-normal multiplicative dequantizer.
mu_and_sigma_params, mu_and_sigma_fn = network_factory(rng_deq, 3, 1)

# Generate the parameters of RealNVP bijectors.
bij_params, bij_fns = [], []
for i in range(5):
    p, f = network_factory(random.fold_in(rng_bij, i), 1, 2)
    bij_params.append(p)
    bij_fns.append(f)

@partial(jit, static_argnums=0)
def train(num_steps: int, lr: float, rng: jnp.ndarray, params: Tuple[jnp.ndarray]) -> Tuple:
    """Training function that estimates both the dequantization and ambient flow
    parameters simultaneously by optimizing the evidence lower bound.

    Args:
        num_steps: The number of training (gradient descent) iterations.
        lr: The gradient descent learning rate.
        rng: Pseudo-random number generator key.
        params: The parameters of the dequantization distribution and the ambient
            flow.

    Returns:
        out: A tuple containing the optimal parameters that maximize the evidence
            lower bound and a trace of the ELBO throughout learning.

    """
    opt_init, opt_update, get_params = optimizers.adam(lr)
    def step(opt_state, it):
        rng_step = random.fold_in(rng, it)
        mu_and_sigma_params, bij_params = get_params(opt_state)
        nelbo, nelbo_grad = value_and_grad(
            negative_elbo, (0, 1))(mu_and_sigma_params, bij_params, mu_and_sigma_fn,
                           bij_fns, rng_step, obs)
        elbo = -nelbo
        opt_state = opt_update(it, nelbo_grad, opt_state)
        return opt_state, elbo
    opt_state, elbo = lax.scan(step, opt_init(params), jnp.arange(num_steps))
    params = get_params(opt_state)
    return params, elbo

params = (mu_and_sigma_params, bij_params)
(mu_and_sigma_params, bij_params), elbo = train(10000, 1e-3, rng_train, params)

x, _ = dequantize(rng, mu_and_sigma_params, mu_and_sigma_fn, obs)
xs = random.normal(rng, [10000, 3])
xs = forward(bij_params, bij_fns, xs)

fig = plt.figure(figsize=(12, 5))
ax = fig.add_subplot(131)
ax.set_title('Evidence Lower Bound')
ax.plot(jnp.arange(len(elbo)), elbo, '-')
ax.grid(linestyle=':')
ax = fig.add_subplot(132, projection='3d')
ax.plot(obs[:, 0], obs[:, 1], obs[:, 2], '.', alpha=0.05, label='Observations')
ax.plot(x[:, 0], x[:, 1], x[:, 2], '.', alpha=0.05, label='Dequantization')
ax.plot(xs[:, 0], xs[:, 1], xs[:, 2], '.', alpha=0.05, label='Ambient Samples')
ax.grid(linestyle=':')
ax.set_xlim((-1.1, 1.1))
ax.set_ylim((-1.1, 1.1))
ax.set_zlim((-1.1, 1.1))
ax.set_title('Dequantization Visualization')
leg = ax.legend()
for lh in leg.legendHandles:
    lh._legmarker.set_alpha(1)

ax = fig.add_subplot(133, projection='3d')
ax.plot(obs[:, 0], obs[:, 1], obs[:, 2], '.', alpha=0.05, label='Observations')
ax.plot(x[:, 0], x[:, 1], x[:, 2], '.', alpha=0.05, label='Dequantization')
ax.plot(xs[:, 0], xs[:, 1], xs[:, 2], '.', alpha=0.05, label='Ambient Samples')
ax.view_init(0., 90.)
ax.grid(linestyle=':')
ax.set_xlim((-1.1, 1.1))
ax.set_ylim((-1.1, 1.1))
ax.set_zlim((-1.1, 1.1))

plt.tight_layout()
plt.savefig(os.path.join('images', 'training-objectives-sphere.png'))

# Visualize samples.
samples = xs / jnp.linalg.norm(xs, axis=-1)[..., jnp.newaxis]
lim = 1.1
fig = plt.figure(figsize=(6, 4))
ax = fig.add_subplot(111, projection='3d')
ax.plot(obs[:, 0], obs[:, 1], obs[:, 2], '.', label='Observations', alpha=0.5)
ax.plot(samples[:, 0], samples[:, 1], samples[:, 2], '.', label='Dequantization Samples', alpha=0.5)
ax.set_title('Sample Comparison')
ax.legend()
ax.grid(linestyle=':')
ax.set_xlim((-lim, lim))
ax.set_ylim((-lim, lim))
ax.set_zlim((-lim, lim))
for i, angle in enumerate(range(0, 360)):
    ax.view_init(30, angle)
    plt.savefig(os.path.join('video-frames', 'spherical-samples-{:05d}.png'.format(i)))
