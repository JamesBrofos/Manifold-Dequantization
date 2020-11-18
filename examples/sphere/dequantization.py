import argparse
import os
from functools import partial
from typing import Callable, Sequence, Tuple

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import jax.numpy as jnp
import jax.scipy.special as jspsp
import jax.scipy.stats as jspst
from jax import lax, nn, random, tree_util
from jax import grad, jacobian, jit, value_and_grad, vmap
from jax.experimental import optimizers, stax

import prax.distributions as pd
import prax.utils as put
from prax.bijectors import realnvp, permute


parser = argparse.ArgumentParser(description='Density estimation for sphere distribution')
parser.add_argument('--num-steps', type=int, default=1000, help='Number of gradient descent iterations for score matching training')
parser.add_argument('--lr', type=float, default=1e-3, help='Gradient descent learning rate')
parser.add_argument('--num-batch', type=int, default=100, help='Number of samples per batch')
parser.add_argument('--density', type=str, default='sphere', help='Indicator of which density function on the sphere to use')
parser.add_argument('--elbo-loss', type=int, default=1, help='Flag to indicate using the ELBO loss')
parser.add_argument('--num-importance', type=int, default=20, help='Number of importance samples to draw during training; if the ELBO loss is used, this argument is ignored')
parser.add_argument('--num-hidden', type=int, default=512, help='Number of hidden units used in the neural networks')
parser.add_argument('--num-realnvp', type=int, default=5, help='Number of RealNVP bijectors to employ')
parser.add_argument('--seed', type=int, default=0, help='Pseudo-random number generator seed')
args = parser.parse_args()


def importance_log_density(rng: jnp.ndarray, bij_params: Sequence[jnp.ndarray], bij_fns: Sequence[Callable], deq_params: Sequence[jnp.ndarray], deq_fn: Callable, num_is: int, xsph: jnp.ndarray) -> jnp.ndarray:
    """Use importance samples to estimate the log-density on the sphere.

    Args:
        rng: Pseudo-random number generator seed.
        bij_params: List of arrays parameterizing the RealNVP bijectors.
        bij_fns: List of functions that compute the shift and scale of the RealNVP
            affine transformation.
        deq_params: Parameters of the mean and scale functions used in
            the log-normal dequantizer.
        deq_fn: Function that computes the mean and scale of the dequantization
            distribution.
        num_is: Number of importance samples.
        xsph: Observations on the sphere.

    Returns:
        is_log_dens: The estimated log-density on the manifold, computed via
            importance sampling.

    """
    xdeq, deq_log_dens = dequantize(rng, deq_params, deq_fn, xsph, num_is)
    amb_log_dens = ambient_flow_log_prob(bij_params, bij_fns, xdeq)
    is_log_dens = jspsp.logsumexp(amb_log_dens - deq_log_dens, axis=0) - jnp.log(num_is)
    return is_log_dens

@partial(jit, static_argnums=(2, 4, 5))
def importance_density(rng: jnp.ndarray, bij_params: Sequence[jnp.ndarray], bij_fns: Sequence[Callable], deq_params: Sequence[jnp.ndarray], deq_fn: Callable, num_is: int, xsph: jnp.ndarray) -> jnp.ndarray:
    """Compute the estimate of the density on the sphere via importance sampling.
    The calculation is encapsulated in a scan so that a large number of
    importance samples may be used without running out of memory.

    Args:
        rng: Pseudo-random number generator seed.
        bij_params: List of arrays parameterizing the RealNVP bijectors.
        bij_fns: List of functions that compute the shift and scale of the RealNVP
            affine transformation.
        deq_params: Parameters of the mean and scale functions used in
            the log-normal dequantizer.
        deq_fn: Function that computes the mean and scale of the dequantization
            distribution.
        num_is: Number of importance samples.
        xsph: Observations on the sphere.

    Returns:
        prob: The importance sampling estimate of the density on the sphere.

    """
    def step(it: int, p: jnp.ndarray):
        """Calculate the importance sampling estimate of the density for a single point
        on the sphere.

        Args:
            it: Iteration over points on the manifold at which to estimate the
                density.
            p: The observation on the sphere.

        Returns:
            out: A tuple containing the next iteration counter and the estimated
                sphere density.

        """
        rng_step = random.fold_in(rng, it)
        log_prob = importance_log_density(rng_step, bij_params, bij_fns, deq_params, deq_fn, num_is, p)
        prob = jnp.exp(log_prob)
        return it + 1, prob
    _, prob = lax.scan(step, 0, xsph)
    return prob

def network_factory(rng: jnp.ndarray, num_in: int, num_out: int, num_hidden: int) -> Tuple:
    """Factory for producing neural networks and their parameterizations.

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

_project = lambda x: x / jnp.linalg.norm(x, axis=-1)[..., jnp.newaxis]

def project(xamb: jnp.ndarray) -> jnp.ndarray:
    """Projection of points in the ambient space to the sphere.

    Args:
        xamb: Observations in the ambient space.

    Returns:
        out: Projections to the surface of the sphere.

    """
    return _project(xamb)

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
    num_masked = num_dims - 2
    perm = jnp.roll(jnp.arange(num_dims), 1)
    y = x
    for i in range(args.num_realnvp):
        y = realnvp.forward(y, num_masked, params[i], fns[i])
        y = permute.forward(y, perm)
    return y

def ambient_flow_log_prob(params: Sequence[jnp.ndarray], fns: Sequence[Callable], y: jnp.ndarray) -> jnp.ndarray:
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
    num_masked = num_dims - 2
    perm = jnp.roll(jnp.arange(num_dims), 1)
    fldj = 0.
    for i in reversed(range(args.num_realnvp)):
        y = permute.inverse(y, perm)
        fldj += permute.forward_log_det_jacobian()
        y = realnvp.inverse(y, num_masked, params[i], fns[i])
        fldj += realnvp.forward_log_det_jacobian(y, num_masked, params[i], fns[i])
    logprob = jspst.multivariate_normal.logpdf(y, jnp.zeros((num_dims, )), 1.)
    return logprob - fldj

def sample_ambient(rng: jnp.ndarray, num_samples: int, bij_params:
                   Sequence[jnp.ndarray], bij_fns: Sequence[Callable],
                   num_dims: int) -> Tuple[jnp.ndarray]:
    """Generate random samples from the ambient distribution and the projection of
    those samples to the sphere.

    Args:
        rng: Pseudo-random number generator seed.
        num_samples: Number of samples to generate.
        bij_params: List of arrays parameterizing the RealNVP bijectors.
        bij_fns: List of functions that compute the shift and scale of the RealNVP
            affine transformation.
        num_dims: Dimensionality of samples.

    Returns:
        xamb, xsph: A tuple containing the ambient samples and the projection of
            the samples to the sphere.

    """
    xamb = random.normal(rng, [num_samples, num_dims])
    xamb = forward(bij_params, bij_fns, xamb)
    xsph = project(xamb)
    return xamb, xsph

def dequantize(rng: jnp.ndarray, deq_params: Sequence[jnp.ndarray], deq_fn: Callable, xsph: jnp.ndarray, num_samples: int) -> Tuple[jnp.ndarray]:
    """Dequantize observations on the sphere into the ambient space.

    Args:
        rng: Pseudo-random number generator seed.
        deq_params: Parameters of the mean and scale functions used in
            the log-normal dequantizer.
        deq_fn: Function that computes the mean and scale of the dequantization
            distribution.
        xsph: Observations on the sphere.
        num_samples: Number of dequantization samples.

    Returns:
        out: A tuple containing the dequantized samples and the log-density of
            the dequantized samples.

    """
    # Dequantization parameters.
    mu, sigma = deq_fn(deq_params, xsph)
    mu = nn.softplus(mu)
    # Random samples for dequantization.
    rng, rng_rad = random.split(rng, 2)
    mu, sigma = mu[..., 0], sigma[..., 0]
    rad = pd.lognormal.rvs(rng_rad, mu, sigma, [num_samples] + list(xsph.shape[:-1]))
    xdeq = rad[..., jnp.newaxis] * xsph
    # Dequantization density calculation.
    ldj = -(num_dims - 1) * jnp.log(rad)
    logdens = pd.lognormal.logpdf(rad, mu, sigma) + ldj
    return xdeq, logdens

def negative_elbo(rng: jnp.ndarray, bij_params: Sequence[jnp.ndarray], bij_fns: Sequence[Callable], deq_params: Sequence[jnp.ndarray], deq_fn: Callable, xsph: jnp.ndarray) -> jnp.ndarray:
    """Compute the negative evidence lower bound of the dequantizing distribution.

    Args:
        rng: Pseudo-random number generator seed.
        bij_params: List of arrays parameterizing the RealNVP bijectors.
        bij_fns: List of functions that compute the shift and scale of the RealNVP
            affine transformation.
        deq_params: Parameters of the mean and scale functions used in
            the log-normal dequantizer.
        deq_fn: Function that computes the mean and scale of the dequantization
            distribution.
        xsph: Observations on the sphere.

    Returns:
        nelbo: The negative evidence lower bound.

    """
    xdeq, deq_log_dens = dequantize(rng, deq_params, deq_fn, xsph, 1)
    amb_log_dens = ambient_flow_log_prob(bij_params, bij_fns, xdeq)
    elbo = jnp.mean(amb_log_dens - deq_log_dens, axis=0)
    nelbo = -elbo
    return nelbo

def loss(rng: jnp.ndarray, xsph: jnp.ndarray, bij_params: Sequence[jnp.ndarray], bij_fns: Sequence[Callable], deq_params: Sequence[jnp.ndarray], deq_fn: Callable, num_samples: int) -> float:
    """Loss function composed of the evidence lower bound and score matching
    loss.

    Args:
        rng: Pseudo-random number generator seed.
        bij_params: List of arrays parameterizing the RealNVP bijectors.
        bij_fns: List of functions that compute the shift and scale of the RealNVP
            affine transformation.
        deq_params: Parameters of the mean and scale functions used in
            the log-normal dequantizer.
        deq_fn: Function that computes the mean and scale of the dequantization
            distribution.
        num_samples: Number of samples to draw using rejection sampling.

    Returns:
        nelbo: The negative evidence lower bound.

    """
    nelbo = negative_elbo(rng, bij_params, bij_fns, deq_params, deq_fn, xsph).mean()
    return nelbo

@partial(jit, static_argnums=(3, 5, 6, 8))
def train(rng: jnp.ndarray, xsph: jnp.ndarray, bij_params: Sequence[jnp.ndarray], bij_fns: Sequence[Callable], deq_params: Sequence[jnp.ndarray], deq_fn: Callable, num_steps: int, lr: float, num_samples: int) -> Tuple:
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
        num_samples: Number of dequantization samples.

    Returns:
        out: A tuple containing the estimated parameters of the ambient flow
            density and the dequantization distribution. The other element is
            the trace of the loss function.

    """
    opt_init, opt_update, get_params = optimizers.adam(lr)
    def step(opt_state, it):
        step_rng = random.fold_in(rng, it)
        bij_params, deq_params = get_params(opt_state)
        loss_val, loss_grad = value_and_grad(loss, (2, 4))(step_rng, xsph, bij_params, bij_fns, deq_params, deq_fn, num_samples)
        loss_grad = tree_util.tree_map(partial(put.clip_and_zero_nans, clip_value=1.), loss_grad)
        opt_state = opt_update(it, loss_grad, opt_state)
        return opt_state, loss_val
    opt_state, trace = lax.scan(step, opt_init((bij_params, deq_params)), jnp.arange(num_steps))
    bij_params, deq_params = get_params(opt_state)
    return (bij_params, deq_params), trace

def sph2euclid(theta: jnp.ndarray, phi: jnp.ndarray) -> jnp.ndarray:
    """Parameterize a point on the sphere as two angles in spherical coordinates.

    Args:
        theta: First angular coordinate.
        phi: Second angular coordinate.

    Returns:
        out: The point on the sphere parameterized by the two angular
            coordinates.

    """
    return jnp.array([jnp.sin(phi)*jnp.cos(theta),
                      jnp.sin(phi)*jnp.sin(theta),
                      jnp.cos(phi)]).T

def embedded_sphere_density(xsph: jnp.ndarray) -> jnp.ndarray:
    """Unnormalized multimodal density on the sphere.

    Args:
        xsph: Observations on the sphere at which to compute the unnormalized
            density.

    Returns:
        out: The unnormalized density at the provided points on the sphere.

    """
    p = lambda x, mu: jnp.exp(10.*x.dot(mu))
    mua = sph2euclid(0.7, 1.5)
    mub = sph2euclid(-1., 1.)
    muc = sph2euclid(0.6, 0.5)
    mud = sph2euclid(-0.7, 4.)
    return p(xsph, mua) + p(xsph, mub) + p(xsph, muc) + p(xsph, mud)

@partial(jit, static_argnums=(1, 2, 3))
def rejection_sampling(rng: jnp.ndarray, num_samples: int, num_dims: int, sphere_density: Callable) -> jnp.ndarray:
    """Samples from the sphere in embedded coordinates using the uniform
    distribution on the sphere as a proposal density.

    Args:
        rng: Pseudo-random number generator seed.
        num_samples: Number of samples to attempt draw. The number of samples
            returned may be smaller since some samples will be rejected.
        num_dims: Dimensionality of the samples.
        sphere_density: The density on the sphere from which to generate samples.

    Returns:
        samples: Samples from the distribution on the sphere.

    """
    # Precompute certain quantities used in rejection sampling. The upper bound
    # on the density was found according to large-scale simulation.
    # >>> rng = random.PRNGKey(0)
    # >>> embedded_sphere_density(pd.sphere.haar.rvs(rng, [10000000, 3])).max()
    prop_dens = jnp.exp(pd.sphere.haar.logpdf(jnp.array([0., 0., 1.])))
    M = 25000. / prop_dens
    denom = M * prop_dens

    def cond(val):
        """Check whether or not the proposal has been accepted.

        Args:
            val: A tuple containing the previous proposal, whether or not it was
                accepted (it wasn't), and the current iteration of the rejection
                sampling loop.

        Returns:
            out: A boolean for whether or not to continue sampling. If the sample
                was rejected, try again. Otherwise, return the accepted sample.

        """
        _, isacc, _ = val
        return jnp.logical_not(isacc)

    def sample_once(sample_iter, val):
        """Attempt to draw a single sample. If the sample is rejected, this function is
        called in a while loop until a sample is accepted.

        Args:
            sample_iter: Sampling iteration counter.
            val: A tuple containing the previous proposal, whether or not it was
                accepted (it wasn't), and the current iteration of the rejection
                sampling loop.

        Returns:
            out: A tuple containing the proposal, whether or not it was accepted, and
                the next iteration counter.

        """
        _, _, it = val
        rng_sample_once = random.fold_in(random.fold_in(rng, it), sample_iter)
        rng_prop, rng_acc = random.split(rng_sample_once, 2)
        xsph = pd.sphere.haar.rvs(rng_prop, [num_dims])
        numer = sphere_density(xsph)
        alpha = numer / denom
        unif = random.uniform(rng_acc)
        isacc = unif < alpha
        return xsph, isacc, it + 1

    def sample(_, it):
        """Samples in a loop so that the total number of samples has a predictable
        shape. The first argument is ignored. The second argument is the
        sampling iteration.

        Args:
            _: Ignored argument for `lax.scan` compatibility.
            it: Sampling iteration number.

        Returns:
            _, xsph: A tuple containing the ignored input quantity and the
                accepted sample.

        """
        state = lax.while_loop(cond, partial(sample_once, it), (jnp.zeros(num_dims), False, 0))
        xsph, isacc, num_iters = state
        return _, xsph

    _, xsph = lax.scan(sample, None, jnp.arange(num_samples))
    return xsph


# Number of dimensions of Euclidean embedding space.
num_dims = 3

# Set random number generation seeds.
rng = random.PRNGKey(args.seed)
rng, rng_bij, rng_deq = random.split(rng, 3)
rng, rng_train = random.split(rng, 2)
rng, rng_xamb, rng_xobs = random.split(rng, 3)
rng, rng_is, rng_kl, rng_mw = random.split(rng, 4)
rng, rng_rej = random.split(rng, 2)

# Generate the parameters of RealNVP bijectors.
bij_params, bij_fns = [], []
for i in range(args.num_realnvp):
    p, f = network_factory(random.fold_in(rng_bij, i), num_dims - 2, 2, args.num_hidden)
    bij_params.append(p)
    bij_fns.append(f)

# Parameterize the mean and scale of a log-normal multiplicative dequantizer.
deq_params, deq_fn = network_factory(rng_deq, num_dims, 1, args.num_hidden)

# May need to reduce scale of initial parameters for stability.
bij_params = tree_util.tree_map(lambda x: x / 2., bij_params)
deq_params = tree_util.tree_map(lambda x: x / 1., deq_params)

# Compute the number of parameters.
count = lambda x: jnp.prod(jnp.array(x.shape))
num_bij_params = jnp.array(tree_util.tree_map(count, tree_util.tree_flatten(bij_params)[0])).sum()
num_deq_params = jnp.array(tree_util.tree_map(count, tree_util.tree_flatten(deq_params)[0])).sum()
num_params = num_bij_params + num_deq_params
print('dequantization parameters: {} - ambient parameters: {} - number of parameters: {}'.format(num_deq_params, num_bij_params, num_params))
# exit()
# Use rejection sampling to obtain samples from the density.
xsph = rejection_sampling(rng_rej, 100000, num_dims, embedded_sphere_density)

# Estimate parameters of the dequantizer and ambient flow.
(bij_params, deq_params), trace = train(rng_train, xsph, bij_params, bij_fns, deq_params, deq_fn, args.num_steps, args.lr, args.num_batch)

# Sample using dequantization and rejection sampling.
xamb, xsph = sample_ambient(rng_xamb, 100000, bij_params, bij_fns, num_dims)
xobs = rejection_sampling(rng_xobs, len(xsph), num_dims, embedded_sphere_density)

# Compute comparison statistics.
mean_mse = jnp.square(jnp.linalg.norm(xsph.mean(0) - xobs.mean(0)))
cov_mse = jnp.square(jnp.linalg.norm(jnp.cov(xsph.T) - jnp.cov(xobs.T)))
approx = importance_density(rng_kl, bij_params, bij_fns, deq_params, deq_fn, 1000, xsph)
target = embedded_sphere_density(xsph)
w = target / approx
Z = jnp.nanmean(w)
log_approx = jnp.log(approx)
log_target = jnp.log(target)
kl = jnp.nanmean(log_approx - log_target) + jnp.log(Z)
ess = jnp.square(jnp.nansum(w)) / jnp.nansum(jnp.square(w))
ress = 100 * ess / len(w)
print('Mean MSE: {:.5f} - Covariance MSE: {:.5f} - KL$(q\Vert p)$ = {:.5f} - Rel. ESS: {:.2f}%'.format(mean_mse, cov_mse, kl, ress))
