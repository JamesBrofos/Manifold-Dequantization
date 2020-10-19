from typing import Callable, Sequence, Tuple

import jax.numpy as jnp
import jax.scipy.stats as jspst
from jax import random
from jax.experimental import stax

from prax.bijectors import realnvp, permute

from polar import polar


def network_factory(rng: jnp.ndarray, num_in: int, num_out: int, num_hidden: int) -> Tuple:
    """Factory for producing neural networks and their parameterizations used in
    the ambient flow.

    Args:
        rng: Pseudo-random number generator seed.
        num_in: Number of inputs to the network.
        num_out: Number of variables to transform by an affine transformation.
            Each variable receives an associated shift and scale.
        num_hidden: Number of hidden units in the hidden layer.

    Returns:
        out: A tuple containing the network parameters and a callable function
            that returns the neural network output for the given input.

    """
    params_init, fn = stax.serial(
        stax.Dense(num_hidden), stax.Relu,
        stax.Dense(num_hidden), stax.Relu,
        stax.FanOut(2),
        stax.parallel(stax.Dense(num_out),
                      stax.serial(stax.Dense(num_out), stax.Softplus)))
    _, params = params_init(rng, (-1, num_in))
    return params, fn

def forward(params: Sequence[jnp.ndarray], fns: Sequence[Callable], x: jnp.ndarray) -> jnp.ndarray:
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
    num_dims = x.shape[-1]
    num_dims_sq = num_dims**2
    half_num_dims_sq = num_dims_sq // 2
    num_masked = num_dims_sq - half_num_dims_sq
    perm = jnp.roll(jnp.arange(num_dims_sq), 1)
    y = x.reshape((-1, num_dims_sq))
    for i in range(len(fns)):
        y = realnvp.forward(y, num_masked, params[i], fns[i])
        y = permute.forward(y, perm)
    return y.reshape(x.shape)

def log_prob(params: Sequence[jnp.ndarray], fns: Sequence[Callable], y: jnp.ndarray) -> jnp.ndarray:
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
    num_dims = y.shape[-1]
    num_dims_sq = num_dims**2
    half_num_dims_sq = num_dims_sq // 2
    num_masked = num_dims_sq - half_num_dims_sq
    perm = jnp.roll(jnp.arange(num_dims_sq), 1)
    fldj = 0.
    y = y.reshape((-1, num_dims_sq))
    for i in reversed(range(len(fns))):
        y = permute.inverse(y, perm)
        fldj += permute.forward_log_det_jacobian()
        y = realnvp.inverse(y, num_masked, params[i], fns[i])
        fldj += realnvp.forward_log_det_jacobian(y, num_masked, params[i], fns[i])
    logprob = jspst.multivariate_normal.logpdf(y, jnp.zeros((num_dims_sq, )), 1.)
    return logprob - fldj

def sample(rng: jnp.ndarray, num_samples: int, bij_params: Sequence[jnp.ndarray], bij_fns: Sequence[Callable], num_dims: int) -> Tuple[jnp.ndarray]:
    """Generate random samples from the ambient distribution and the projection of
    those samples to O(n).

    Args:
        rng: Pseudo-random number generator seed.
        num_samples: Number of samples to generate.
        bij_params: List of arrays parameterizing the RealNVP bijectors.
        bij_fns: List of functions that compute the shift and scale of the RealNVP
            affine transformation.
        num_dims: Dimensionality of samples.

    Returns:
        xamb, xon: A tuple containing the ambient samples and the projection of
            the samples to O(n).

    """
    num_dims_sq = num_dims**2
    xamb = random.normal(rng, [num_samples, num_dims, num_dims])
    xamb = forward(bij_params, bij_fns, xamb)
    xon, _ = polar(xamb.reshape((-1, num_dims, num_dims)))
    return xamb, xon
