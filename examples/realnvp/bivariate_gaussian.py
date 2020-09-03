import os
from functools import partial
from typing import Callable, Sequence, Tuple

import matplotlib.pyplot as plt

import jax.numpy as jnp
import jax.scipy.stats as jspst
from jax import lax, random
from jax import jit, value_and_grad
from jax.experimental import optimizers, stax

from prax.bijectors import realnvp, permute


"""This code implements training a composition of RealNVP bijectors to model a
multivariate Gaussian distribution. Given observations of the target
multivariate Gaussian, the objective is to maximize the log-likelihood of these
observations under the RealNVP model.

"""

def shift_and_scale_fn_factory(rng: jnp.ndarray, num_in: int, num_out: int) -> Tuple:
    """Factory for producing shift and scale networks and their parameterizations.

    Args:
        rng: Pseudo-random number generator seed.
        num_in: Number of inputs to the network.
        num_out: Number of variables to transform by an affine transformation.
            Each variable receives an associated shift and scale.

    Returns:
        out: A tuple containing the network parameters and a callable function
            that returns the shift and scale for given inputs.

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
    y = permute.forward(y, jnp.array([1, 0]))
    y = realnvp.forward(y, 1, params[1], fns[1])
    return y

def ambient_log_prob(params: Sequence[jnp.ndarray], fns:
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
    y = permute.inverse(y, jnp.array([1, 0]))
    fldj += permute.forward_log_det_jacobian()
    y = realnvp.inverse(y, 1, params[0], fns[0])
    fldj += realnvp.forward_log_det_jacobian(y, 1, params[0], fns[0])
    return jspst.multivariate_normal.logpdf(y, jnp.zeros((2, )), 1.) - fldj


def negative_log_likelihood(params: Sequence[jnp.ndarray], fns:
                            Sequence[Callable], y: jnp.ndarray) -> float:
    return -jnp.mean(ambient_log_prob(params, fns, y))

# Generate observations from a multivariate normal with a given mean and
# covariance structure.
mu = jnp.array([3., 5.])
Cov = jnp.array([[1., 0.5], [0.5, 2.]])
rng = random.PRNGKey(0)
rng, y_rng = random.split(rng, 2)
y = random.multivariate_normal(y_rng, mu, Cov, [10000])

# Generate the parameters of two RealNVP bijectors.
params, fns = [], []
for i in range(2):
    p, f = shift_and_scale_fn_factory(random.fold_in(rng, i), 1, 1)
    params.append(p)
    fns.append(f)

def update(it: int, opt_state: optimizers.OptimizerState, fns: Sequence[Callable],
           y: jnp.ndarray) -> Tuple:
    """Compute the gradient of the negative log-likelihood loss function with
    respect to the parameters of the RealNVP bijectors given observations. Take
    a gradient step and record the log-likelihood. The log-likelihood should
    increase as the parameters of the RealNVP bijectors are estimated.

    Args:
        it: Current iteration counter.
        opt_state: Current state of the optimizer.
        fns: List of functions that compute the shift and scale of the RealNVP
            affine transformation.
        y: Observations whose likelihood under the composition of bijectors
            should be computed.

    Returns:
        out: Subsequent optimizer state and the value of the log-likelihood at
            the previous values of the parameters.

    """
    params = get_params(opt_state)
    nll, nll_grad = value_and_grad(negative_log_likelihood)(params, fns, y)
    ll = -nll
    return opt_update(it, nll_grad, opt_state), ll

@partial(jit, static_argnums=(1, 3, ))
def train(params: Sequence[jnp.ndarray], fns: Sequence[Callable], y: jnp.ndarray,
          num_steps: int) -> Tuple:
    """Use gradient descent to minimize the negative log-likelihood.

    Args:
        params: List of arrays parameterizing the RealNVP bijectors.
        fns: List of functions that compute the shift and scale of the RealNVP
            affine transformation.
        y: Observations whose likelihood under the composition of bijectors
            should be computed.
        num_steps: The number of gradient descent iterations.

    Returns:
        out: A tuple containing the optimal parameters for the RealNVP bijectors
            obtained via gradient descent and the trace of the log-likelihood.

    """
    _update = lambda opt_state, it: update(it, opt_state, fns, y)
    opt_state, trace = lax.scan(_update, opt_init(params), jnp.arange(num_steps))
    return get_params(opt_state), trace

opt_init, opt_update, get_params = optimizers.adam(1e-3)
params, trace = train(params, fns, y, 1000)

# Compute the forward transformation and verify that the samples generated from
# the composition of bijectors resembles the target distribution.
rng, x_rng = random.split(rng, 2)
x = random.normal(x_rng, [10000, 2])
yp = forward(params, fns, x)

figs, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].plot(jnp.arange(len(trace)) + 1., trace, '-')
axes[0].grid(linestyle=':')
axes[0].set_title('Log-Likelihood')
axes[1].plot(y[:, 0], y[:, 1], '.', label='Original Samples', alpha=0.5)
axes[1].plot(yp[:, 0], yp[:, 1], '.', label='RealNVP', alpha=0.5)
axes[1].legend()
axes[1].set_title('RealNVP Samples')
axes[1].grid(linestyle=':')
plt.savefig(os.path.join('images', 'bivariate-gaussian-realnvp.png'))

# Compute an approximation of the KL divergence. Should be small since RealNVP
# can approximate the target multivariate normal distribution.
lpamb = ambient_log_prob(params, fns, yp)
ltarg = jspst.multivariate_normal.logpdf(yp, mu, Cov)
kl = jnp.mean(lpamb - ltarg)
print('kl-divergence: {:.5f}'.format(kl))
