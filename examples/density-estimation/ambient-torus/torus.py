import argparse
import os
from functools import partial
from typing import Callable, Sequence, Tuple

import matplotlib.pyplot as plt

import jax.numpy as jnp
import jax.scipy.special as jspsp
import jax.scipy.stats as jspst
from jax import lax, nn, random
from jax import grad, jacobian, jit, value_and_grad, vmap
from jax.experimental import optimizers, stax

import prax.distributions as pd
import prax.manifolds as pm
from prax.bijectors import realnvp, permute

from coordinates import ang2euclid, euclid2ang
from rejection_sampling import correlated_torus_density, embedded_torus_density, multimodal_torus_density, rejection_sampling, unimodal_torus_density


parser = argparse.ArgumentParser(description='Density estimation for torus distribution')
parser.add_argument('--num-steps', type=int, default=1000, help='Number of gradient descent iterations for score matching training')
parser.add_argument('--lr', type=float, default=1e-3, help='Gradient descent learning rate')
parser.add_argument('--num-batch', type=int, default=100, help='Number of samples per batch')
parser.add_argument('--density', type=str, default='correlated', help='Indicator of which density function on the torus to use')
parser.add_argument('--seed', type=int, default=0, help='Pseudo-random number generator seed')
args = parser.parse_args()

torus_density = {
    'correlated': correlated_torus_density,
    'multimodal': multimodal_torus_density,
    'unimodal': unimodal_torus_density
}[args.density]

def importance_log_density(rng: jnp.ndarray, bij_params: Sequence[jnp.ndarray], bij_fns: Sequence[Callable], deq_params: Sequence[jnp.ndarray], deq_fn: Callable, num_is: int, xtor: jnp.ndarray) -> jnp.ndarray:
    """Use importance samples to estimate the log-density on the torus.

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
        xtor: Observations on the torus.

    Returns:
        is_log_dens: The estimated log-density on the manifold, computed via
            importance sampling.

    """
    xdeq, deq_log_dens = dequantize(rng, deq_params, deq_fn, xtor, num_is)
    amb_log_dens = ambient_flow_log_prob(bij_params, bij_fns, xdeq)
    is_log_dens = jspsp.logsumexp(amb_log_dens - deq_log_dens, axis=0) - jnp.log(num_is)
    return is_log_dens

@partial(jit, static_argnums=(2, 4, 5))
def importance_density(rng: jnp.ndarray, bij_params: Sequence[jnp.ndarray], bij_fns: Sequence[Callable], deq_params: Sequence[jnp.ndarray], deq_fn: Callable, num_is: int, xtor: jnp.ndarray) -> jnp.ndarray:
    """Compute the estimate of the density on the torus via importance sampling.
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
        xtor: Observations on the torus.

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
        rng_step = random.fold_in(rng, it)
        log_prob = importance_log_density(rng_step, bij_params, bij_fns, deq_params, deq_fn, num_is, p)
        prob = jnp.exp(log_prob)
        return it + 1, prob
    _, prob = lax.scan(step, 0, xtor)
    return prob

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

_project = lambda x: x / jnp.linalg.norm(x, axis=-1)[..., jnp.newaxis]

def project(xamb: jnp.ndarray) -> jnp.ndarray:
    """Projection of points in the ambient space to the torus. The torus is the
    product manifold of two circles. Therefore, we project points in
    four-dimensional space to the surface of two circles.

    Args:
        xamb: Observations in the ambient space.

    Returns:
        out: Projections to the surface of two circles.

    """
    xa, xb = xamb[..., :2], xamb[..., 2:]
    return jnp.hstack((_project(xa), _project(xb)))

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
    perm = jnp.array([1, 3, 2, 0])
    y = realnvp.forward(x, 2, params[0], fns[0])
    y = permute.forward(y, perm)
    y = realnvp.forward(y, 2, params[1], fns[1])
    y = permute.forward(y, perm)
    y = realnvp.forward(y, 2, params[2], fns[2])
    y = permute.forward(y, perm)
    y = realnvp.forward(y, 2, params[3], fns[3])
    y = permute.forward(y, perm)
    y = realnvp.forward(y, 2, params[4], fns[4])
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
    num_dims = y.shape[-1]
    perm = jnp.array([1, 3, 2, 0])
    fldj = 0.
    y = realnvp.inverse(y, 2, params[4], fns[4])
    fldj += realnvp.forward_log_det_jacobian(y, 2, params[4], fns[4])
    y = permute.inverse(y, perm)
    fldj += permute.forward_log_det_jacobian()
    y = realnvp.inverse(y, 2, params[3], fns[3])
    fldj += realnvp.forward_log_det_jacobian(y, 2, params[3], fns[3])
    y = permute.inverse(y, perm)
    fldj += permute.forward_log_det_jacobian()
    y = realnvp.inverse(y, 2, params[2], fns[2])
    fldj += realnvp.forward_log_det_jacobian(y, 2, params[2], fns[2])
    y = permute.inverse(y, perm)
    fldj += permute.forward_log_det_jacobian()
    y = realnvp.inverse(y, 2, params[1], fns[1])
    fldj += realnvp.forward_log_det_jacobian(y, 2, params[1], fns[1])
    y = permute.inverse(y, perm)
    fldj += permute.forward_log_det_jacobian()
    y = realnvp.inverse(y, 2, params[0], fns[0])
    fldj += realnvp.forward_log_det_jacobian(y, 2, params[0], fns[0])
    logprob = jspst.multivariate_normal.logpdf(y, jnp.zeros((num_dims, )), 1.)
    return logprob - fldj

def sample_ambient(rng: jnp.ndarray, num_samples: int, bij_params:
                   Sequence[jnp.ndarray], bij_fns: Sequence[Callable],
                   num_dims: int) -> Tuple[jnp.ndarray]:
    """Generate random samples from the ambient distribution and the projection of
    those samples to the torus.

    Args:
        rng: Pseudo-random number generator seed.
        num_samples: Number of samples to generate.
        bij_params: List of arrays parameterizing the RealNVP bijectors.
        bij_fns: List of functions that compute the shift and scale of the RealNVP
            affine transformation.
        num_dims: Dimensionality of samples.

    Returns:
        xamb, xsph: A tuple containing the ambient samples and the projection of
            the samples to the torus.

    """
    xamb = random.normal(rng, [num_samples, num_dims])
    xamb = forward(bij_params, bij_fns, xamb)
    xtor = project(xamb)
    return xamb, xtor

def dequantize(rng: jnp.ndarray, deq_params: Sequence[jnp.ndarray], deq_fn: Callable, xtor: jnp.ndarray, num_samples: int) -> Tuple[jnp.ndarray]:
    """Dequantize observations on the torus into the ambient space. The torus is
    the product manifold of two circles so observations are dequantized
    according to a log-normal dequantizer.

    Args:
        rng: Pseudo-random number generator seed.
        deq_params: Parameters of the mean and scale functions used in
            the log-normal dequantizer.
        deq_fn: Function that computes the mean and scale of the dequantization
            distribution.
        xtor: Observations on the torus.
        num_samples: Number of dequantization samples.

    Returns:
        out: A tuple containing the dequantized samples and the log-density of
            the dequantized samples.

    """
    # Dequantization parameters.
    mu, sigma = deq_fn(deq_params, xtor)
    mu = nn.softplus(mu)
    mua, mub = mu[..., 0], mu[..., 1]
    sigmaa, sigmab = sigma[..., 0], sigma[..., 1]
    # Random samples for dequantization.
    rng, rng_rada, rng_radb = random.split(rng, 3)
    rada = pd.lognormal.sample(rng_rada, mua, sigmaa, [num_samples] + list(xtor.shape[:-1]))
    radb = pd.lognormal.sample(rng_radb, mub, sigmab, [num_samples] + list(xtor.shape[:-1]))
    tora, torb = xtor[..., :2], xtor[..., 2:]
    deqa = rada[..., jnp.newaxis] * tora
    deqb = radb[..., jnp.newaxis] * torb
    xdeq = jnp.concatenate((deqa, deqb), axis=-1)
    # Dequantization density calculation.
    ldj = -(jnp.log(rada) + jnp.log(radb))
    logdens = pd.lognormal.logpdf(rada, mua, sigmaa) + pd.lognormal.logpdf(radb, mub, sigmab) + ldj
    return xdeq, logdens

def negative_elbo(rng: jnp.ndarray, bij_params: Sequence[jnp.ndarray], bij_fns: Sequence[Callable], deq_params: Sequence[jnp.ndarray], deq_fn: Callable, xtor: jnp.ndarray) -> jnp.ndarray:
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
        xtor: Observations on the torus.

    Returns:
        nelbo: The negative evidence lower bound.

    """
    xdeq, deq_log_dens = dequantize(rng, deq_params, deq_fn, xtor, 1)
    amb_log_dens = ambient_flow_log_prob(bij_params, bij_fns, xdeq)
    elbo = jnp.mean(amb_log_dens - deq_log_dens, axis=0)
    nelbo = -elbo
    return nelbo

def loss(rng: jnp.ndarray, bij_params: Sequence[jnp.ndarray], bij_fns: Sequence[Callable], deq_params: Sequence[jnp.ndarray], deq_fn: Callable, num_samples: int) -> float:
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
    rng, rng_rej, rng_elbo, rng_deq = random.split(rng, 4)
    xang = rejection_sampling(rng_rej, num_samples, torus_density)
    xtor = ang2euclid(xang)
    nelbo = negative_elbo(rng_elbo, bij_params, bij_fns, deq_params, deq_fn, xtor).mean()
    return nelbo

@partial(jit, static_argnums=(2, 4, 5, 7))
def train(rng: jnp.ndarray, bij_params: Sequence[jnp.ndarray], bij_fns: Sequence[Callable], deq_params: Sequence[jnp.ndarray], deq_fn: Callable, num_steps: int, lr: float, num_samples: int) -> Tuple:
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
        loss_val, loss_grad = value_and_grad(loss, (1, 3))(step_rng, bij_params, bij_fns, deq_params, deq_fn, num_samples)
        opt_state = opt_update(it, loss_grad, opt_state)
        return opt_state, loss_val
    opt_state, trace = lax.scan(step, opt_init((bij_params, deq_params)), jnp.arange(num_steps))
    bij_params, deq_params = get_params(opt_state)
    return (bij_params, deq_params), trace


# Set random number generation seeds.
rng = random.PRNGKey(args.seed)
rng, rng_bij, rng_deq = random.split(rng, 3)
rng, rng_train = random.split(rng, 2)
rng, rng_xamb, rng_xobs = random.split(rng, 3)
rng, rng_is, rng_kl = random.split(rng, 3)

# Generate the parameters of RealNVP bijectors.
bij_params, bij_fns = [], []
for i in range(5):
    p, f = network_factory(random.fold_in(rng_bij, i), 2, 2)
    bij_params.append(p)
    bij_fns.append(f)

# Parameterize the mean and scale of a log-normal multiplicative dequantizer.
deq_params, deq_fn = network_factory(rng_deq, 4, 2)

# Estimate parameters of the dequantizer and ambient flow.
(bij_params, deq_params), trace = train(rng_train, bij_params, bij_fns, deq_params, deq_fn, args.num_steps, args.lr, args.num_batch)

# Visualize learned distribution.
xamb, xtor = sample_ambient(rng_xamb, 100000, bij_params, bij_fns, 4)
xang = euclid2ang(xtor)
xobs = rejection_sampling(rng_xobs, len(xtor), torus_density)

# Compute comparison statistics.
mean_mse = jnp.square(jnp.linalg.norm(xang.mean(0) - xobs.mean(0)))
cov_mse = jnp.square(jnp.linalg.norm(jnp.cov(xang.T) - jnp.cov(xobs.T)))
approx = importance_density(rng_kl, bij_params, bij_fns, deq_params, deq_fn, 100, xtor[:2000])
target = embedded_torus_density(xtor[:2000], torus_density)
Z = jnp.mean(target / approx)
log_approx = jnp.log(approx)
log_target = jnp.log(target)
kl = jnp.mean(log_approx - log_target) + jnp.log(Z)

# Density on a grid.
lin = jnp.linspace(-jnp.pi, jnp.pi)
xx, yy = jnp.meshgrid(lin, lin)
theta = jnp.vstack((xx.ravel(), yy.ravel())).T
ptor = ang2euclid(theta)
prob = importance_density(rng_is, bij_params, bij_fns, deq_params, deq_fn, 10000, ptor)
aprob = torus_density(theta)

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
plt.suptitle('Mean MSE: {:.5f} - Covariance MSE: {:.5f} - KL$(q\Vert p)$ = {:.5f}'.format(mean_mse, cov_mse, kl))
plt.tight_layout()
plt.savefig(os.path.join('images', '{}.png'.format(args.density)))
