import argparse
import os
from functools import partial
from typing import Callable, Sequence, Tuple

import matplotlib.pyplot as plt
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
    y = permute.forward(y, jnp.array([1, 0]))
    y = realnvp.forward(y, 1, params[1], fns[1])
    y = permute.forward(y, jnp.array([1, 0]))
    y = realnvp.forward(y, 1, params[2], fns[2])
    y = permute.forward(y, jnp.array([1, 0]))
    y = realnvp.forward(y, 1, params[3], fns[3])
    y = permute.forward(y, jnp.array([1, 0]))
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
    y = permute.inverse(y, jnp.array([1, 0]))
    fldj += permute.forward_log_det_jacobian()
    y = realnvp.inverse(y, 1, params[3], fns[3])
    fldj += realnvp.forward_log_det_jacobian(y, 1, params[3], fns[3])
    y = permute.inverse(y, jnp.array([1, 0]))
    fldj += permute.forward_log_det_jacobian()
    y = realnvp.inverse(y, 1, params[2], fns[2])
    fldj += realnvp.forward_log_det_jacobian(y, 1, params[2], fns[2])
    y = permute.inverse(y, jnp.array([1, 0]))
    fldj += permute.forward_log_det_jacobian()
    y = realnvp.inverse(y, 1, params[1], fns[1])
    fldj += realnvp.forward_log_det_jacobian(y, 1, params[1], fns[1])
    y = permute.inverse(y, jnp.array([1, 0]))
    fldj += permute.forward_log_det_jacobian()
    y = realnvp.inverse(y, 1, params[0], fns[0])
    fldj += realnvp.forward_log_det_jacobian(y, 1, params[0], fns[0])
    return jspst.multivariate_normal.logpdf(y, jnp.zeros((2, )), 1.) - fldj

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

# Generate observations.
scale = 1e-1
obs = jnp.vstack((
    scale*random.normal(random.fold_in(rng, 0), [10000, 2]) + jnp.array([1., 1.]),
    scale*random.normal(random.fold_in(rng, 1), [10000, 2]) + jnp.array([-1., 1.])))
obs /= jnp.linalg.norm(obs, axis=-1)[..., jnp.newaxis]

# Parameterize the mean and scale of a log-normal multiplicative dequantizer.
mu_and_sigma_params, mu_and_sigma_fn = network_factory(rng_deq, 2, 1)

# Generate the parameters of RealNVP bijectors.
bij_params, bij_fns = [], []
for i in range(5):
    p, f = network_factory(random.fold_in(rng_bij, i), 1, 1)
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
xs = random.normal(rng, [10000, 2])
xs = forward(bij_params, bij_fns, xs)

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].set_title('Evidence Lower Bound')
axes[0].plot(jnp.arange(len(elbo)), elbo, '-')
axes[0].grid(linestyle=':')
axes[1].set_title('Dequantization')
axes[1].plot(obs[:, 0], obs[:, 1], '.', alpha=0.05, label='Observations')
axes[1].plot(x[:, 0], x[:, 1], '.', alpha=0.05, label='Dequantization')
axes[1].plot(xs[:, 0], xs[:, 1], '.', alpha=0.05, label='Ambient Samples')
axes[1].grid(linestyle=':')
leg = axes[1].legend()
for lh in leg.legendHandles:
    lh._legmarker.set_alpha(1)

axes[1].set_xlim((-1.1, 1.1))
axes[1].set_ylim((-1.1, 1.1))
axes[1].axis('square')
plt.tight_layout()
plt.savefig(os.path.join('images', 'training-objectives-circle.png'))
