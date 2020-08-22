import argparse
import os
from functools import partial
from typing import Callable, Sequence, Tuple

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import jax.numpy as jnp
import jax.scipy.stats as jspst
from jax import lax, random
from jax import jit, value_and_grad
from jax.experimental import optimizers, stax

from prax.bijectors import realnvp, permute
from prax.distributions import lognormal, sphere


parser = argparse.ArgumentParser(description='Power Spherical Sampling via Dequantization')
parser.add_argument('--num-dequantization-steps', type=int, default=1000, help='Number of training steps for estimating dequantization parameters')
args = parser.parse_args()

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

def ambient_log_prior(x: jnp.ndarray) -> jnp.ndarray:
    """Prior distribution in the ambient space."""
    r = jnp.linalg.norm(x, axis=-1)
    return jspst.norm.logpdf(r, 5.0, 1.0)

def negative_elbo(mu_and_sigma_params: Sequence[jnp.ndarray], mu_and_sigma_fn:
                  Callable, rng: jnp.ndarray, y: jnp.ndarray) -> float:
    """Compute the negative evidence lower bound of the dequantizing distribution.
    This is the loss function for learning parameters of the dequantizing
    distribution.

    """
    mu, sigma = mu_and_sigma_fn(mu_and_sigma_params, y)
    ln = lognormal.sample(rng, mu, sigma, mu.shape)
    x = ln * y
    pamb = ambient_log_prior(x)
    qcond = lognormal.logpdf(ln, mu, sigma).squeeze(-1)
    # The ELBO is the expected difference of the log-prior in the ambient space
    # and the conditional variational dequantizing distribution.
    elbo = jnp.mean(pamb - qcond)
    return -elbo

# Generate random draws from the power spherical distribution.
rng = random.PRNGKey(0)
kappa = 50.
musph = jnp.array([1., -1., 1.])
musph /= jnp.linalg.norm(musph)
y = sphere.powsph(rng, kappa, musph, [10000])

# Parameterize the mean and scale of a log-normal multiplicative dequantizer.
rng, deq_rng = random.split(rng, 2)
mu_and_sigma_params, mu_and_sigma_fn = network_factory(deq_rng, 3, 1)

# Train the variational dequantization distribution.
opt_init, opt_update, get_params = optimizers.adam(1e-3)

def update(it: int, rng: jnp.ndarray, opt_state: optimizers.OptimizerState, y:
           jnp.ndarray) -> Tuple:
    """Update the parameters of the dequantizing distribution by minimizing the
    negative evidence lower bound.

    Args:
        it: Iteration counter.
        rng: Pseudo-random number generator.
        opt_state: Current state of the optimizer.
        y: Manifold-constrained bservations to dequantize.

    Returns:
        out: A tuple containing the optimizer state and the value of the
            evidence lower bound.

    """
    params = get_params(opt_state)
    nelbo, nelbo_grad = value_and_grad(negative_elbo)(params, mu_and_sigma_fn, rng, y)
    elbo = -nelbo
    return opt_update(it, nelbo_grad, opt_state), elbo

@jit
def train():
    """Run gradient descent to minimize the negative evidence lower bound.

    Returns:
        out: A tuple containing the optimal parameterization of the mean and
            variance functions and a trace of the ELBO by iteration.

    """
    opt_state = opt_init(mu_and_sigma_params)
    def step(opt_state, it):
        iter_rng = random.fold_in(rng, it)
        idx = random.permutation(iter_rng, y.shape[0])[:100]
        yb = y[idx]
        opt_state, elbo = update(it, iter_rng, opt_state, yb)
        return opt_state, elbo
    opt_state, trace = lax.scan(step, opt_state, jnp.arange(args.num_dequantization_steps))
    return get_params(opt_state), trace

# Compute the dequantization of the power spherical density.
mu_and_sigma_params, elbo = train()

if args.num_dequantization_steps > 0:
    plt.figure(figsize=(4, 4))
    plt.plot(jnp.arange(len(elbo)), elbo, '-')
    plt.grid(linestyle=':')
    plt.xlabel('Gradient Ascent Iteration')
    plt.title('Evidence Lower Bound')
    plt.savefig(os.path.join('images', 'elbo-maximization.png'))

# Delete irrelevant variables to avoid poluting the namespace.
del (opt_init, opt_update, get_params, train)

# Generate the parameters of two RealNVP bijectors.
params, fns = [], []
for i in range(2):
    p, f = network_factory(random.fold_in(rng, i), 1, 2)
    params.append(p)
    fns.append(f)

def forward(params: Sequence[jnp.ndarray], fns: Sequence[Callable], x:
            jnp.ndarray) -> jnp.ndarray:
    """Forward transformation of two RealNVP bijectors and a permutation bijector
    between them.

    Args:
        params: List of arrays parameterizing the RealNVP bijectors.
        fns: List of functions that compute the shift and scale of the RealNVP
            affine transformation.
        x: Input to transform according to the composition of two RealNVP
            transformations and a permutation.

    Returns:
        y: The transformed input.

    """
    y = realnvp.forward(x, 1, params[0], fns[0])
    y = permute.forward(y, jnp.array([1, 2, 0]))
    y = realnvp.forward(y, 1, params[1], fns[1])
    return y

def negative_log_likelihood(params: Sequence[jnp.ndarray], fns:
                            Sequence[Callable], y: jnp.ndarray) -> float:
    """Compute the negative log-likelihood of observations under the transformation
    given by two RealNVP bijectors and a permutation bijector between them.
    Assumes that the base distribution is a standard multivariate normal.

    Args:
        params: List of arrays parameterizing the RealNVP bijectors.
        fns: List of functions that compute the shift and scale of the RealNVP
            affine transformation.
        y: Observations whose likelihood under the composition of bijectors
            should be computed.

    Returns:
        out: The average negative log-likelihood of the observations given the
            parameters of the bijection composition.

    """
    y = realnvp.inverse(y, 1, params[1], fns[1])
    fldj = realnvp.forward_log_det_jacobian(y, 1, params[1], fns[1])
    y = permute.inverse(y, jnp.array([1, 2, 0]))
    fldj += permute.forward_log_det_jacobian()
    y = realnvp.inverse(y, 1, params[0], fns[0])
    fldj += realnvp.forward_log_det_jacobian(y, 1, params[0], fns[0])
    return -jnp.mean(jspst.multivariate_normal.logpdf(y, jnp.zeros((3, )), 1.) - fldj)

def update(it: int, opt_state: optimizers.OptimizerState, fns:
           Sequence[Callable]) -> Tuple:
    """Compute the gradient of the negative log-likelihood loss function with
    respect to the parameters of the RealNVP bijectors given observations. Take
    a gradient step and record the log-likelihood. The log-likelihood should
    increase as the parameters of the RealNVP bijectors are estimated.

    Args:
        it: Current iteration counter.
        opt_state: Current state of the optimizer.
        fns: List of functions that compute the shift and scale of the RealNVP
            affine transformation.

    Returns:
        out: Subsequent optimizer state and the value of the log-likelihood at
            the previous values of the parameters.

    """
    params = get_params(opt_state)
    mu, sigma = mu_and_sigma_fn(mu_and_sigma_params, y)
    ln = lognormal.sample(random.fold_in(rng, it), mu, sigma, mu.shape)
    x = ln * y
    nll, nll_grad = value_and_grad(negative_log_likelihood)(params, fns, x)
    ll = -nll
    return opt_update(it, nll_grad, opt_state), ll

@partial(jit, static_argnums=(1, 2))
def train(params: Sequence[jnp.ndarray], fns: Sequence[Callable], num_steps:
          int) -> Tuple:
    """Use gradient descent to minimize the negative log-likelihood.

    Args:
        params: List of arrays parameterizing the RealNVP bijectors.
        fns: List of functions that compute the shift and scale of the RealNVP
            affine transformation.
        num_steps: The number of gradient descent iterations.

    Returns:
        out: A tuple containing the optimal parameters for the RealNVP bijectors
            obtained via gradient descent and the trace of the log-likelihood.

    """
    _update = lambda opt_state, it: update(it, opt_state, fns)
    opt_state, trace = lax.scan(_update, opt_init(params), jnp.arange(num_steps))
    return get_params(opt_state), trace

opt_init, opt_update, get_params = optimizers.adam(1e-3)
params, trace = train(params, fns, 1000)
rng, x_rng = random.split(rng, 2)
x = random.normal(x_rng, [10000, 3])
yp = forward(params, fns, x)
yp /= jnp.linalg.norm(yp, axis=-1)[..., jnp.newaxis]

# Compare summary statistics.
emp_mean = yp.mean(0)
thr_mean = sphere.expectation_powsph(kappa, musph)
err = jnp.linalg.norm(emp_mean - thr_mean)
rerr = err / jnp.linalg.norm(thr_mean)
print('difference between means:       {:.5f} - relative error: {:.5f}'.format(err, rerr))
emp_cov = jnp.cov(yp.T)
thr_cov = sphere.variance_powsph(kappa, musph)
err = jnp.linalg.norm(emp_cov - thr_cov)
rerr = err / jnp.linalg.norm(thr_cov)
print('difference between covariances: {:.5f} - relative error: {:.5f}'.format(err, rerr))


# Visualize samples.
lim = 1.1
fig = plt.figure(figsize=(10, 4))
ax = fig.add_subplot(121, projection='3d')
ax.plot(y[:, 0], y[:, 1], y[:, 2], '.', label='Original Samples', alpha=0.5)
ax.plot(yp[:, 0], yp[:, 1], yp[:, 2], '.', label='Dequantization Samples', alpha=0.5)
ax.set_title('Sample Comparison')
ax.legend()
ax.grid(linestyle=':')
ax.set_xlim((-lim, lim))
ax.set_ylim((-lim, lim))
ax.set_zlim((-lim, lim))
ax = fig.add_subplot(122)
ax.plot(jnp.arange(len(trace)) + 1, trace, '-')
ax.grid(linestyle=':')
ax.set_title('Log-Likelihood')
plt.tight_layout()
plt.savefig(os.path.join('images', 'power-spherical-samples-num-dequantization-steps-{}.png'.format(args.num_dequantization_steps)))
