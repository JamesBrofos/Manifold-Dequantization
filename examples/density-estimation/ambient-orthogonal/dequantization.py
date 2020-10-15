from typing import Callable, Sequence, Tuple

import jax.numpy as jnp
import jax.scipy.stats as jspst
from jax import nn, ops, random
from jax import jacobian, vmap
from jax.experimental import stax

import prax.distributions as pd

from polar import vecpolar, transp


@vmap
def tril_factory(diag, tril):
    """Parameterizes a n x n lower-triangular matrix by its diagonal and
    off-diagonal entries, both presented as vectors.

    Args:
        diag: Vector of the diagonal entries of the matrix.
        tril: Vector of the off-diagonal entries of the matrix.

    Returns:
        L: Lower triangular matrix.

    """
    n = diag.size
    B = jnp.zeros((n, n))
    B = ops.index_update(B, jnp.tril_indices(n, -1), tril)
    L = B + jnp.diag(diag)
    return L

def network(rng: random.PRNGKey, num_in: int, num_hidden: int) -> Tuple:
    """Factory for producing the dequantization neural network and its
    parameterizations.

    Args:
        rng: Pseudo-random number generator seed.
        num_in: Number of inputs to the network.
        num_hidden: Number of hidden units in the hidden layer.

    Returns:
        out: A tuple containing the network parameters and a callable function
            that returns the neural network output for the given input.

    """
    num_in_sq = num_in**2
    num_in_choose_two = num_in * (num_in - 1) // 2
    params_init, fn = stax.serial(
        stax.Flatten,
        stax.Dense(num_hidden), stax.Relu,
        stax.Dense(num_hidden), stax.Relu,
        stax.FanOut(4),
        stax.parallel(stax.Dense(num_in),
                      stax.Dense(num_in_choose_two),
                      stax.serial(stax.Dense(num_in), stax.Softplus),
                      stax.serial(stax.Dense(num_in_choose_two), stax.Softplus),
        ))
    _, params = params_init(rng, (-1, num_in_sq))
    return params, fn

def dequantize(rng: random.PRNGKey, deq_params: Sequence[jnp.ndarray], deq_fn: Callable, xon: jnp.ndarray, num_samples: int):
    """Dequantize O(n) be representing a point in the ambient space a the product
    of a matrix in O(n) and a Cholesky factor. This code places a distribution
    over the cholesky factor.

    Args:
        rng: Pseudo-random number generator seed.
        deq_params: Parameters of the mean and scale functions used in
            the log-normal dequantizer.
        deq_fn: Function that computes the mean and scale of the dequantization
            distribution.
        xon: Observations on O(n).
        num_samples: Number of deqauntization samples to generate.

    Returns:
        xdeq: Dequantized samples.
        lp: The log-density of the dequantization density with a Jacobian
            correction for the ambient space.

    """
    # Compute the dequantization into the ambient space.
    deq = deq_fn(deq_params, xon)
    (dmu, odmu), (dsigma, odsigma) = deq[:2], deq[2:]
    # mulim = 2.5
    # sigmamax = 3.
    # dmu = jnp.clip(dmu, -mulim, mulim)
    # odmu = jnp.clip(odmu, -mulim, mulim)
    # dsigma = jnp.clip(dsigma, 0.1, sigmamax)
    # odsigma = jnp.clip(odsigma, 0.1, sigmamax)
    div = 10.
    dmu, odmu, dsigma, odsigma = dmu / div, odmu / div, dsigma / div, odsigma / div
    rng, rng_d, rng_od = random.split(rng, 3)
    d = pd.lognormal.sample(rng_d, dmu, dsigma, [num_samples] + [len(xon), dmu.shape[-1]])
    od = odmu + odsigma * random.normal(rng_od, [num_samples] + [len(xon), odmu.shape[-1]])
    L = vmap(tril_factory)(d, od)
    R = L@transp(L)
    xdeq = jnp.matmul(xon, R)
    # Compute the Jacobian determinant correction.
    num_dims = xon.shape[-1]
    num_dims_sq = num_dims**2
    xflat = xdeq.reshape((num_samples, -1, num_dims_sq))
    H = vmap(vmap(jacobian(vecpolar)))(xflat)
    ldj = 0.5 * jnp.linalg.slogdet(jnp.swapaxes(H, -1, -2)@H)[1]
    lpd = pd.lognormal.logpdf(d, dmu, dsigma).sum(axis=-1)
    lpod = jspst.norm.logpdf(od, odmu, odsigma).sum(axis=-1)
    lp = lpd + lpod + ldj
    # Return the dequantization samples and the log-density.
    return xdeq, lp
