import argparse
import os
from functools import partial
from typing import Callable, Sequence, Tuple

import matplotlib.pyplot as plt

import jax.numpy as jnp
import jax.scipy.special as jspsp
import jax.scipy.stats as jspst
from jax import lax, nn, random, tree_util
from jax import grad, jacobian, jit, value_and_grad, vmap
from jax.experimental import optimizers, stax

import prax.distributions as pd
from prax.bijectors import realnvp, permute

from coordinates import ang2euclid, euclid2ang
from rejection_sampling import (
    correlated_torus_density,
    unimodal_torus_density,
    multimodal_torus_density,
    embedded_torus_density,
    rejection_sampling)

parser = argparse.ArgumentParser(description='Autoregressive dequantization for distribution on the torus')
parser.add_argument('--num-steps', type=int, default=10000, help='Number of gradient descent iterations for score matching training')
parser.add_argument('--lr', type=float, default=1e-3, help='Gradient descent learning rate')
parser.add_argument('--num-batch', type=int, default=100, help='Number of samples per batch')
parser.add_argument('--density', type=str, default='correlated', help='Indicator of which density function on the torus to use')
parser.add_argument('--elbo-loss', type=int, default=1, help='Flag to indicate using the ELBO loss')
parser.add_argument('--num-importance', type=int, default=20, help='Number of importance samples to draw during training; if the ELBO loss is used, this argument is ignored')
parser.add_argument('--seed', type=int, default=0, help='Pseudo-random number generator seed')
args = parser.parse_args()

torus_density = {
    'correlated': correlated_torus_density,
    'multimodal': multimodal_torus_density,
    'unimodal': unimodal_torus_density
}[args.density]

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

def forward(params: Sequence[jnp.ndarray], fns: Sequence[Callable], x: jnp.ndarray, *args) -> jnp.ndarray:
    """Forward transformation of composining RealNVP bijectors and a permutation
    bijector between them.

    Args:
        params: List of arrays parameterizing the RealNVP bijectors.
        fns: List of functions that compute the shift and scale of the RealNVP
            affine transformation.
        x: Input to transform according to the composition of RealNVP
            transformations and permutations.
        *args: Arguments to the shift-and-scale functions

    Returns:
        y: The transformed input.

    """
    perm = jnp.array([1, 0])
    y = realnvp.forward(x, 1, params[0], fns[0], *args)
    y = permute.forward(y, perm)
    y = realnvp.forward(y, 1, params[1], fns[1], *args)
    y = permute.forward(y, perm)
    y = realnvp.forward(y, 1, params[2], fns[2], *args)
    y = permute.forward(y, perm)
    y = realnvp.forward(y, 1, params[3], fns[3], *args)
    y = permute.forward(y, perm)
    y = realnvp.forward(y, 1, params[4], fns[4], *args)
    return y

def forward_log_prob(params: Sequence[jnp.ndarray], fns: Sequence[Callable], y: jnp.ndarray, *args) -> jnp.ndarray:
    """Compute the log-probability of ambient observations under the transformation
    given by composing RealNVP bijectors and a permutation bijector between
    them. Assumes that the base distribution is a standard multivariate normal.

    Args:
        params: List of arrays parameterizing the RealNVP bijectors.
        fns: List of functions that compute the shift and scale of the RealNVP
            affine transformation.
        y: Observations whose likelihood under the composition of bijectors
            should be computed.
        *args: Arguments to the shift-and-scale functions

    Returns:
        out: The log-probability of the observations given the parameters of the
            bijection composition.

    """
    num_dims = y.shape[-1]
    perm = jnp.array([1, 0])
    fldj = 0.
    y = realnvp.inverse(y, 1, params[4], fns[4], *args)
    fldj += realnvp.forward_log_det_jacobian(y, 1, params[4], fns[4], *args)
    y = permute.inverse(y, perm)
    fldj += permute.forward_log_det_jacobian()
    y = realnvp.inverse(y, 1, params[3], fns[3], *args)
    fldj += realnvp.forward_log_det_jacobian(y, 1, params[3], fns[3], *args)
    y = permute.inverse(y, perm)
    fldj += permute.forward_log_det_jacobian()
    y = realnvp.inverse(y, 1, params[2], fns[2], *args)
    fldj += realnvp.forward_log_det_jacobian(y, 1, params[2], fns[2], *args)
    y = permute.inverse(y, perm)
    fldj += permute.forward_log_det_jacobian()
    y = realnvp.inverse(y, 1, params[1], fns[1], *args)
    fldj += realnvp.forward_log_det_jacobian(y, 1, params[1], fns[1], *args)
    y = permute.inverse(y, perm)
    fldj += permute.forward_log_det_jacobian()
    y = realnvp.inverse(y, 1, params[0], fns[0], *args)
    fldj += realnvp.forward_log_det_jacobian(y, 1, params[0], fns[0], *args)
    logprob = jspst.multivariate_normal.logpdf(y, jnp.zeros((num_dims, )), 1.)
    return logprob - fldj

project = lambda x: x / jnp.linalg.norm(x, axis=-1, keepdims=True)

def sample_ambient(rng: jnp.ndarray,
                   num_samples: int,
                   biju_params: Sequence[jnp.ndarray],
                   biju_fns: Sequence[Callable],
                   bijc_params: Sequence[jnp.ndarray],
                   bijc_fns: Sequence[Callable],
                   num_dims: int) -> Tuple:
    """Generate random samples from the ambient distribution and the projection of
    those samples to the torus.

    Args:
        rng: Pseudo-random number generator seed.
        num_samples: Number of samples to generate.
        biju_params: List of arrays parameterizing the RealNVP bijectors of the
            unconditional distribution.
        biju_fns: List of functions that compute the shift and scale of the RealNVP
            affine transformation of the unconditional distribution.
        bijc_params: List of arrays parameterizing the RealNVP bijectors of the
            conditional distribution.
        bijc_fns: List of functions that compute the shift and scale of the RealNVP
            affine transformation of the conditional distribution.
        num_dims: Dimensionality of samples.

    Returns:
        xu: Ambient samples of the unconditional distribution.
        xc: Ambient samples of the conditional distribution.
        tu: Samples of the unconditional distribution as the first angular
            parameter of the torus.
        tc: Samples of the conditional distribution as the second angular
            parameter of the torus.

    """
    rng, rng_u, rng_c = random.split(rng, 3)
    xu = random.normal(rng_u, [num_samples, num_dims])
    xu = forward(biju_params, biju_fns, xu)
    tu = project(xu)
    xc = random.normal(rng_c, [num_samples, num_dims])
    xc = forward(bijc_params, bijc_fns, xc, tu)
    tc = project(xc)
    return (xu, xc), (tu, tc)

def ambient_log_prob(biju_params: Sequence[jnp.ndarray],
                     biju_fns: Sequence[Callable],
                     bijc_params: Sequence[jnp.ndarray],
                     bijc_fns: Sequence[Callable],
                     xu: jnp.ndarray,
                     xc: jnp.ndarray,
                     tu: jnp.ndarray,
                     tc: jnp.ndarray) -> jnp.ndarray:
    """Compute the log-probability of ambient observations under the transformation
    given by composing RealNVP bijectors and a permutation bijector between
    them. Assumes that the base distribution is a standard multivariate normal.

    Args:
        biju_params: List of arrays parameterizing the RealNVP bijectors of the
            unconditional distribution.
        biju_fns: List of functions that compute the shift and scale of the RealNVP
            affine transformation of the unconditional distribution.
        bijc_params: List of arrays parameterizing the RealNVP bijectors of the
            conditional distribution.
        bijc_fns: List of functions that compute the shift and scale of the RealNVP
            affine transformation of the conditional distribution.
        xu: Ambient samples of the unconditional distribution.
        xc: Ambient samples of the conditional distribution.
        tu: Samples of the unconditional distribution as the first angular
            parameter of the torus.
        tc: Samples of the conditional distribution as the second angular
            parameter of the torus.

    Returns:
        out: The log-probability of the observations given the parameters of the
            bijection composition.

    """
    lpu = forward_log_prob(biju_params, biju_fns, xu)
    lpc = forward_log_prob(bijc_params, bijc_fns, xc, tu)
    return lpu + lpc

def dequantize(rng: jnp.ndarray, dequ_params: Sequence[jnp.ndarray], dequ_fn: Callable, deqc_params: Sequence[jnp.ndarray], deqc_fn: Callable, tu: jnp.ndarray, tc: jnp.ndarray, num_samples: int) -> Tuple:
    """Dequantize observations on the torus into the ambient space. The torus is
    the product manifold of two circles so observations are dequantized
    according to a log-normal dequantizer.

    Args:
        rng: Pseudo-random number generator seed.
        dequ_params: Parameters of the mean and scale functions used in
            the log-normal dequantizer for the unconditional distribution.
        dequ_fn: Function that computes the mean and scale of the unconditional
            dequantization distribution.
        deqc_params: Parameters of the mean and scale functions used in
            the log-normal dequantizer for the conditional distribution.
        deqc_fn: Function that computes the mean and scale of the conditional
            dequantization distribution.
        tu: Samples of the unconditional distribution as the first angular
            parameter of the torus.
        tc: Samples of the conditional distribution as the second angular
            parameter of the torus.
        num_samples: Number of dequantization samples.

    Returns:
        out: A tuple containing the dequantized samples and the log-density of
            the dequantized samples.

    """
    # Dequantization parameters.
    muu, sigmau = dequ_fn(dequ_params, tu)
    muu = nn.softplus(muu)
    muc, sigmac = deqc_fn(deqc_params, tc, tu)
    muc = nn.softplus(muc)
    muu, sigmau = jnp.squeeze(muu), jnp.squeeze(sigmau)
    muc, sigmac = jnp.squeeze(muc), jnp.squeeze(sigmac)
    # Random samples for dequantization.
    rng, rng_ru, rng_rc = random.split(rng, 3)
    ru = pd.lognormal.sample(rng_ru, muu, sigmau, [num_samples] + list(tu.shape[:-1]))
    rc = pd.lognormal.sample(rng_rc, muc, sigmac, [num_samples] + list(tc.shape[:-1]))
    dequ = ru[..., jnp.newaxis] * tu
    deqc  =rc[..., jnp.newaxis] * tc
    # Dequantization densities.
    lpdu = pd.lognormal.logpdf(ru, muu, sigmau) - jnp.log(ru)
    lpdc = pd.lognormal.logpdf(rc, muc, sigmac) - jnp.log(rc)
    lpd = lpdu + lpdc
    return (dequ, deqc), lpd

def negative_elbo(rng: jnp.ndarray,
                  biju_params: Sequence[jnp.ndarray],
                  biju_fns: Sequence[Callable],
                  bijc_params: Sequence[jnp.ndarray],
                  bijc_fns: Sequence[Callable],
                  dequ_params: Sequence[jnp.ndarray],
                  dequ_fn: Callable,
                  deqc_params: Sequence[jnp.ndarray],
                  deqc_fn: Callable,
                  tu: jnp.ndarray,
                  tc: jnp.ndarray) -> jnp.ndarray:
    """Compute the negative evidence lower bound of the dequantizing distribution.

    Args:
        rng: Pseudo-random number generator seed.
        biju_params: List of arrays parameterizing the RealNVP bijectors of the
            unconditional distribution.
        biju_fns: List of functions that compute the shift and scale of the RealNVP
            affine transformation of the unconditional distribution.
        bijc_params: List of arrays parameterizing the RealNVP bijectors of the
            conditional distribution.
        bijc_fns: List of functions that compute the shift and scale of the RealNVP
            affine transformation of the conditional distribution.
        dequ_params: Parameters of the mean and scale functions used in
            the log-normal dequantizer for the unconditional distribution.
        dequ_fn: Function that computes the mean and scale of the unconditional
            dequantization distribution.
        deqc_params: Parameters of the mean and scale functions used in
            the log-normal dequantizer for the conditional distribution.
        deqc_fn: Function that computes the mean and scale of the conditional
            dequantization distribution.
        tu: Samples of the unconditional distribution as the first angular
            parameter of the torus.
        tc: Samples of the conditional distribution as the second angular
            parameter of the torus.

    Returns:
        nelbo: The negative evidence lower bound.

    """
    num_deq = 10
    (dequ, deqc), lpd = dequantize(rng, dequ_params, dequ_fn, deqc_params, deqc_fn, tu, tc, num_deq)
    tutile = jnp.tile(tu, (num_deq, 1, 1))
    lpa = ambient_log_prob(biju_params, biju_fns, bijc_params, bijc_fns, dequ, deqc, tutile, tc)
    elbo = jnp.mean(lpa - lpd, axis=0)
    nelbo = -elbo
    return nelbo

def zero_nans(g):
    """Remove the NaNs in a matrix by replaceing them with zeros.

    Args:
        g: Matrix whose NaN elements should be replaced by zeros.

    Returns:
        out: The input matrix but with NaN elements replaced by zeros.

    """
    g = jnp.where(jnp.isnan(g), jnp.zeros_like(g), g)
    g = jnp.clip(g, -1., 1.)
    return g

def loss(rng: jnp.ndarray,
         biju_params: Sequence[jnp.ndarray],
         biju_fns: Sequence[Callable],
         bijc_params: Sequence[jnp.ndarray],
         bijc_fns: Sequence[Callable],
         dequ_params: Sequence[jnp.ndarray],
         dequ_fn: Callable,
         deqc_params: Sequence[jnp.ndarray],
         deqc_fn: Callable,
         num_samples: int):
    """Loss function implementation.

    Args:
        rng: Pseudo-random number generator seed.
        biju_params: List of arrays parameterizing the RealNVP bijectors of the
            unconditional distribution.
        biju_fns: List of functions that compute the shift and scale of the RealNVP
            affine transformation of the unconditional distribution.
        bijc_params: List of arrays parameterizing the RealNVP bijectors of the
            conditional distribution.
        bijc_fns: List of functions that compute the shift and scale of the RealNVP
            affine transformation of the conditional distribution.
        dequ_params: Parameters of the mean and scale functions used in
            the log-normal dequantizer for the unconditional distribution.
        dequ_fn: Function that computes the mean and scale of the unconditional
            dequantization distribution.
        deqc_params: Parameters of the mean and scale functions used in
            the log-normal dequantizer for the conditional distribution.
        deqc_fn: Function that computes the mean and scale of the conditional
            dequantization distribution.
        num_samples: Number of samples to draw using rejection sampling.

    Returns:
        out: The computed loss function.

    """
    if args.elbo_loss:
        rng, rng_rej, rng_elbo, rng_deq = random.split(rng, 4)
        xang = rejection_sampling(rng_rej, num_samples, torus_density)
        xtor = ang2euclid(xang)
        tu, tc = xtor[..., :2], xtor[..., 2:]
        nelbo = negative_elbo(rng_elbo, biju_params, biju_fns, bijc_params, bijc_fns, dequ_params, dequ_fn, deqc_params, deqc_fn, tu, tc)
        return nelbo.mean()
    else:
        rng, rng_rej, rng_is = random.split(rng, 3)
        xang = rejection_sampling(rng_rej, num_samples, torus_density)
        xtor = ang2euclid(xang)
        tu, tc = xtor[..., :2], xtor[..., 2:]
        log_is = importance_log_density(rng_is, biju_params, biju_fns, bijc_params, bijc_fns, dequ_params, dequ_fn, deqc_params, deqc_fn, args.num_importance, tu, tc)
        log_target = jnp.log(torus_density(xtor))
        return jnp.mean(log_target - log_is)

@partial(jit, static_argnums=(2, 4, 6, 8, 9, 11))
def train(rng: jnp.ndarray,
          biju_params: Sequence[jnp.ndarray],
          biju_fns: Sequence[Callable],
          bijc_params: Sequence[jnp.ndarray],
          bijc_fns: Sequence[Callable],
          dequ_params: Sequence[jnp.ndarray],
          dequ_fn: Callable,
          deqc_params: Sequence[jnp.ndarray],
          deqc_fn: Callable,
          num_steps: int,
          lr: float,
          num_samples: int) -> Tuple:
    """Train the ambient flow with the combined loss function.

    Args:
        rng: Pseudo-random number generator seed.
        bij_params: List of arrays parameterizing the RealNVP bijectors.
        bij_fns: List of functions that compute the shift and scale of the RealNVP
            affine transformation.
        deq_params: Parameters of the mean and scale functions used in
            the log-normal dequantizer.
        deq_fn: Function that computes the mean and scale of the dequantization
            distribution.
        num_steps: Number of gradient descent iterations.
        lr: Gradient descent learning rate.
        num_samples: Number of samples to draw using rejection sampling.

    Returns:
        out: A tuple containing the estimated parameters of the ambient flow
            density and the dequantization distribution. The other element is
            the trace of the loss function.

    """
    opt_init, opt_update, get_params = optimizers.adam(lr)
    def step(opt_state, it):
        step_rng = random.fold_in(rng, it)
        biju_params, bijc_params, dequ_params, deqc_params = get_params(opt_state)
        loss_val, loss_grad = value_and_grad(loss, (1, 3, 5, 7))(step_rng, biju_params, biju_fns, bijc_params, bijc_fns, dequ_params, dequ_fn, deqc_params, deqc_fn, num_samples)
        loss_grad = tree_util.tree_map(zero_nans, loss_grad)
        opt_state = opt_update(it, loss_grad, opt_state)
        return opt_state, loss_val
    params = (biju_params, bijc_params, dequ_params, deqc_params)
    opt_state, trace = lax.scan(step, opt_init(params), jnp.arange(num_steps))
    biju_params, bijc_params, dequ_params, deqc_params = get_params(opt_state)
    return (biju_params, bijc_params, dequ_params, deqc_params), trace

def importance_log_density(rng: jnp.ndarray,
                           biju_params: Sequence[jnp.ndarray],
                           biju_fns: Sequence[Callable],
                           bijc_params: Sequence[jnp.ndarray],
                           bijc_fns: Sequence[Callable],
                           dequ_params: Sequence[jnp.ndarray],
                           dequ_fn: Callable,
                           deqc_params: Sequence[jnp.ndarray],
                           deqc_fn: Callable,
                           num_is: int,
                           tu: jnp.ndarray,
                           tc: jnp.ndarray) -> jnp.ndarray:
    """Use importance samples to estimate the log-density on the torus.

    Args:
        rng: Pseudo-random number generator seed.
        biju_params: List of arrays parameterizing the RealNVP bijectors of the
            unconditional distribution.
        biju_fns: List of functions that compute the shift and scale of the RealNVP
            affine transformation of the unconditional distribution.
        bijc_params: List of arrays parameterizing the RealNVP bijectors of the
            conditional distribution.
        bijc_fns: List of functions that compute the shift and scale of the RealNVP
            affine transformation of the conditional distribution.
        dequ_params: Parameters of the mean and scale functions used in
            the log-normal dequantizer for the unconditional distribution.
        dequ_fn: Function that computes the mean and scale of the unconditional
            dequantization distribution.
        deqc_params: Parameters of the mean and scale functions used in
            the log-normal dequantizer for the conditional distribution.
        deqc_fn: Function that computes the mean and scale of the conditional
            dequantization distribution.
        num_is: Number of importance samples.
        tu: Samples of the unconditional distribution as the first angular
            parameter of the torus.
        tc: Samples of the conditional distribution as the second angular
            parameter of the torus.

    Returns:
        is_log_dens: The estimated log-density on the manifold, computed via
            importance sampling.

    """
    (dequ, deqc), lpd = dequantize(rng, dequ_params, dequ_fn, deqc_params, deqc_fn, tu, tc, num_is)
    tutile = jnp.tile(tu, (num_is, 1, 1))
    lpa = ambient_log_prob(biju_params, biju_fns, bijc_params, bijc_fns, dequ, deqc, tutile, tc)
    is_log_dens = jspsp.logsumexp(lpa - lpd, axis=0) - jnp.log(num_is)
    return is_log_dens

@partial(jit, static_argnums=(2, 4, 6, 8, 9))
def importance_density(rng: jnp.ndarray,
                       biju_params: Sequence[jnp.ndarray],
                       biju_fns: Sequence[Callable],
                       bijc_params: Sequence[jnp.ndarray],
                       bijc_fns: Sequence[Callable],
                       dequ_params: Sequence[jnp.ndarray],
                       dequ_fn: Callable,
                       deqc_params: Sequence[jnp.ndarray],
                       deqc_fn: Callable,
                       num_is: int,
                       tu: jnp.ndarray,
                       tc: jnp.ndarray) -> jnp.ndarray:
    """Compute the estimate of the density on the torus via importance sampling.
    The calculation is encapsulated in a scan so that a large number of
    importance samples may be used without running out of memory.

    Args:
        rng: Pseudo-random number generator seed.
        biju_params: List of arrays parameterizing the RealNVP bijectors of the
            unconditional distribution.
        biju_fns: List of functions that compute the shift and scale of the RealNVP
            affine transformation of the unconditional distribution.
        bijc_params: List of arrays parameterizing the RealNVP bijectors of the
            conditional distribution.
        bijc_fns: List of functions that compute the shift and scale of the RealNVP
            affine transformation of the conditional distribution.
        dequ_params: Parameters of the mean and scale functions used in
            the log-normal dequantizer for the unconditional distribution.
        dequ_fn: Function that computes the mean and scale of the unconditional
            dequantization distribution.
        deqc_params: Parameters of the mean and scale functions used in
            the log-normal dequantizer for the conditional distribution.
        deqc_fn: Function that computes the mean and scale of the conditional
            dequantization distribution.
        num_is: Number of importance samples.
        tu: Samples of the unconditional distribution as the first angular
            parameter of the torus.
        tc: Samples of the conditional distribution as the second angular
            parameter of the torus.

    Returns:
        prob: The importance sampling estimate of the density on the torus.

    """
    def step(it: int, p: jnp.ndarray):
        """Calculate the importance sampling estimate of the density for a single point
        on the torus.

        Args:
            it: Iteration over points on the manifold at which to estimate the
                density.
            p: The observation on the torus.

        Returns:
            out: A tuple containing the next iteration counter and the estimated
                torus density.

        """
        ptu, ptc = p[:2][jnp.newaxis], p[2:][jnp.newaxis]
        rng_step = random.fold_in(rng, it)
        log_prob = importance_log_density(rng_step, biju_params, biju_fns, bijc_params, bijc_fns, dequ_params, dequ_fn, deqc_params, deqc_fn, num_is, ptu, ptc)
        prob = jnp.exp(log_prob)
        return it + 1, prob
    xtor = jnp.concatenate((tu, tc), axis=-1)
    _, prob = lax.scan(step, 0, xtor)
    return jnp.squeeze(prob)


# Set random number generation seeds.
rng = random.PRNGKey(args.seed)
rng, rng_biju, rng_bijc = random.split(rng, 3)
rng, rng_dequ, rng_deqc = random.split(rng, 3)
rng, rng_train = random.split(rng, 2)
rng, rng_amb, rng_obs = random.split(rng, 3)
rng, rng_is, rng_kl = random.split(rng, 3)

# Parameterize unconditional and conditional RealNVP bijectors.
biju_params, biju_fns = [], []
for i in range(5):
    p, f = network_factory(random.fold_in(rng_biju, i), 1, 1)
    biju_params.append(p)
    biju_fns.append(f)

bijc_params, bijc_fns = [], []
for i in range(5):
    p, fp = network_factory(random.fold_in(rng_bijc, i), 3, 1)
    f = lambda params, inputs, cond: fp(params, jnp.concatenate((inputs, cond), axis=-1))
    bijc_params.append(p)
    bijc_fns.append(f)

# Parameterize the mean and scale of log-normal multiplicative dequantizers for
# the conditional and unconditional distributions.
dequ_params, dequ_fn = network_factory(rng_dequ, 2, 1)
deqc_params, deqc_fnp = network_factory(rng_deqc, 4, 1)
deqc_fn = lambda params, inputs, cond: deqc_fnp(params, jnp.concatenate((inputs, cond), axis=-1))

# Estimate parameters of the dequantizers and ambient flow.
(biju_params, bijc_params, dequ_params, deqc_params), trace = train(rng, biju_params, biju_fns, bijc_params, bijc_fns, dequ_params, dequ_fn, deqc_params, deqc_fn, args.num_steps, args.lr, args.num_batch)

# Sample from the learned distribution.
(xu, xc), (tu, tc) = sample_ambient(rng_amb, 10000, biju_params, biju_fns, bijc_params, bijc_fns, 2)
xtor = jnp.concatenate((tu, tc), axis=-1)
xang = euclid2ang(xtor)
xobs = rejection_sampling(rng_obs, len(xang), torus_density)
approx = importance_density(rng_kl, biju_params, biju_fns, bijc_params, bijc_fns, dequ_params, dequ_fn, deqc_params, deqc_fn, 1000, tu, tc)
target = embedded_torus_density(xtor, torus_density)

# Compute comparison statistics.
mean_mse = jnp.square(jnp.linalg.norm(xang.mean(0) - xobs.mean(0)))
cov_mse = jnp.square(jnp.linalg.norm(jnp.cov(xang.T) - jnp.cov(xobs.T)))
w = target / approx
Z = jnp.mean(w)
log_approx = jnp.log(approx)
log_target = jnp.log(target)
kl = jnp.mean(log_approx - log_target) + jnp.log(Z)
ess = jnp.square(jnp.sum(w)) / jnp.sum(jnp.square(w))
ress = 100 * ess / len(w)

# Compute density on a grid.
lin = jnp.linspace(-jnp.pi, jnp.pi)
xx, yy = jnp.meshgrid(lin, lin)
theta = jnp.vstack((xx.ravel(), yy.ravel())).T
ptor = ang2euclid(theta)
prob = importance_density(rng_is, biju_params, biju_fns, bijc_params, bijc_fns, dequ_params, dequ_fn, deqc_params, deqc_fn, 1000, ptor[..., :2], ptor[..., 2:])
aprob = torus_density(theta)

# Visualize learned distribution.
fig, axes = plt.subplots(1, 4, figsize=(16, 5))
axes[0].plot(trace)
axes[0].grid(linestyle=':')
axes[0].set_ylabel('Combined Loss')
axes[0].set_xlabel('Gradient Descent Iteration')
num_plot = 10000
axes[1].plot(xobs[:num_plot, 0], xobs[:num_plot, 1], '.', alpha=0.2, label='Rejection Sampling')
axes[1].plot(xang[:num_plot, 0], xang[:num_plot, 1], '.', alpha=0.2, label='Dequantization Sampling')
axes[1].grid(linestyle=':')
leg = axes[1].legend()
for lh in leg.legendHandles:
    lh._legmarker.set_alpha(1)

axes[2].contourf(xx, yy, jnp.clip(prob, 0., jnp.quantile(prob, 0.95)).reshape(xx.shape))
axes[2].set_title('Importance Sample Density Estimate')
axes[3].contourf(xx, yy, aprob.reshape(xx.shape))
axes[3].set_title('Analytic Density')
plt.suptitle('Mean MSE: {:.5f} - Covariance MSE: {:.5f} - KL$(q\Vert p)$ = {:.5f} - Rel. ESS: {:.2f}%'.format(mean_mse, cov_mse, kl, ress))
plt.tight_layout()
ln = 'elbo' if args.elbo_loss else 'kl'
plt.savefig(os.path.join('images', 'autoregressive-{}-{}-num-batch-{}-num-importance-{}-num-steps-{}-seed-{}.png'.format(ln, args.density, args.num_batch, args.num_importance, args.num_steps, args.seed)))
