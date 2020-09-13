from typing import Callable, Sequence, Tuple

import jax.numpy as jnp
import jax.scipy.stats as jspst
from jax import random

from prax.bijectors import realnvp, permute

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

def sample_ambient(rng: jnp.ndarray, num_samples: int, bij_params:
                   Sequence[jnp.ndarray], bij_fns: Sequence[Callable]) -> Tuple[jnp.ndarray]:
    """Generate random samples from the ambient distribution and the projection of
    those samples to the sphere.

    Args:
        rng: Pseudo-random number generator seed.
        num_samples: Number of samples to generate.
        bij_params: List of arrays parameterizing the RealNVP bijectors.
        bij_fns: List of functions that compute the shift and scale of the RealNVP
            affine transformation.

    Returns:
        z, zproj: A tuple containing the ambient samples and the projection of
            the samples to the sphere.

    """
    z = random.normal(rng, [num_samples, 3])
    z = forward(bij_params, bij_fns, z)
    zproj = z / jnp.linalg.norm(z, axis=-1)[..., jnp.newaxis]
    return z, zproj
