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
import prax.manifolds as pm
import prax.utils as put
from prax.bijectors import realnvp, permute

from distributions import embedded_sphere_density
from rejection_sampling import rejection_sampling

parser = argparse.ArgumentParser(description='Density estimation for sphere distribution')
parser.add_argument('--num-steps', type=int, default=20000, help='Number of gradient descent iterations for score matching training')
parser.add_argument('--lr', type=float, default=1e-3, help='Gradient descent learning rate')
parser.add_argument('--num-batch', type=int, default=100, help='Number of samples per batch')
parser.add_argument('--elbo-loss', type=int, default=1, help='Flag to indicate using the ELBO loss')
parser.add_argument('--num-importance', type=int, default=20, help='Number of importance samples to draw during training; if the ELBO loss is used, this argument is ignored')
parser.add_argument('--num-hidden', type=int, default=70, help='Number of hidden units used in the neural networks')
parser.add_argument('--num-realnvp', type=int, default=3, help='Number of RealNVP bijectors to employ')
parser.add_argument('--seed', type=int, default=0, help='Pseudo-random number generator seed')
args = parser.parse_args()


def importance_log_density(rng: jnp.ndarray, bij_params: Sequence[jnp.ndarray], bij_fns: Sequence[Callable], deq_params: Sequence[jnp.ndarray], deq_fn: Callable, num_is: int, xsph: jnp.ndarray) -> jnp.ndarray:
    """Use importance samples to estimate the log-density on the sphere.

    Args:
        rng: Pseudo-random number generator seed.
        bij_params: List of arrays parameterizing the RealNVP bijectors.
        bij_fns: List of functions that compute the shift and scale of the RealNVP
            affine transformation.
        deq_params: Parameters of the mean and scale functions used in the
            log-normal dequantizer.
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

@partial(jit, static_argnums=(3, ))
def importance_density(rng: jnp.ndarray, bij_params: Sequence[jnp.ndarray], deq_params: Sequence[jnp.ndarray], num_is: int, xsph: jnp.ndarray) -> jnp.ndarray:
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
    rng, rng_rej, rng_loss = random.split(rng, 3)
    xsph = rejection_sampling(rng_rej, num_samples, num_dims, embedded_sphere_density)
    if args.elbo_loss:
        nelbo = negative_elbo(rng_loss, bij_params, bij_fns, deq_params, deq_fn, xsph).mean()
        return nelbo
    else:
        log_is = importance_log_density(rng_loss, bij_params, bij_fns, deq_params, deq_fn, args.num_importance, xsph)
        log_target = jnp.log(embedded_sphere_density(xsph))
        return jnp.mean(log_target - log_is)

@partial(jit, static_argnums=(3, 5))
def train(rng: jnp.ndarray, bij_params: Sequence[jnp.ndarray], deq_params: Sequence[jnp.ndarray], num_steps: int, lr: float, num_samples: int) -> Tuple:
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
        loss_grad = tree_util.tree_map(partial(put.clip_and_zero_nans, clip_value=1.), loss_grad)
        opt_state = opt_update(it, loss_grad, opt_state)
        return opt_state, loss_val
    opt_state, trace = lax.scan(step, opt_init((bij_params, deq_params)), jnp.arange(num_steps))
    bij_params, deq_params = get_params(opt_state)
    return (bij_params, deq_params), trace


# Number of dimensions of Euclidean embedding space.
num_dims = 4

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

# Estimate parameters of the dequantizer and ambient flow.
(bij_params, deq_params), trace = train(rng_train, bij_params, deq_params, args.num_steps, args.lr, args.num_batch)

# Sample using dequantization and rejection sampling.
xamb, xsph = sample_ambient(rng_xamb, 100000, bij_params, bij_fns, num_dims)
xobs = rejection_sampling(rng_xobs, len(xsph), num_dims, embedded_sphere_density)

# Compute comparison statistics.
mean_mse = jnp.square(jnp.linalg.norm(xsph.mean(0) - xobs.mean(0)))
cov_mse = jnp.square(jnp.linalg.norm(jnp.cov(xsph.T) - jnp.cov(xobs.T)))
approx = importance_density(rng_kl, bij_params, deq_params, 1000, xsph)
target = embedded_sphere_density(xsph)
w = target / approx
Z = jnp.nanmean(w)
log_approx = jnp.log(approx)
log_target = jnp.log(target)
klqp = jnp.nanmean(log_approx - log_target) + jnp.log(Z)
ess = jnp.square(jnp.nansum(w)) / jnp.nansum(jnp.square(w))
ress = 100 * ess / len(w)
del w, Z, log_approx, approx, log_target, target
approx = importance_density(rng_kl, bij_params, deq_params, 1000, xobs)
target = embedded_sphere_density(xobs)
w = approx / target
Z = jnp.nanmean(w)
log_approx = jnp.log(approx)
log_target = jnp.log(target)
klpq = jnp.nanmean(log_target - log_approx) + jnp.log(Z)
del w, Z, log_approx, approx, log_target, target
method = 'dequantization ({})'.format('ELBO' if args.elbo_loss else 'KL')
print('{} - Mean MSE: {:.5f} - Covariance MSE: {:.5f} - KL$(q\Vert p)$ = {:.5f} - KL$(p\Vert q)$ = {:.5f} - Rel. ESS: {:.2f}%'.format(method, mean_mse, cov_mse, klqp, klpq, ress))


# Visualize the target density and the approximation.
if args.seed == 0:
    num_slices = 8
    fig = plt.figure(figsize=(10, 1), constrained_layout=True)
    for i, gamma in enumerate(jnp.linspace(0., jnp.pi, num_slices+2)[1:-1]):
        theta = jnp.linspace(-jnp.pi, jnp.pi)[1:-1]
        phi = jnp.linspace(-jnp.pi / 2, jnp.pi / 2)[1:-1]
        xx, yy = jnp.meshgrid(theta, phi)
        grid = jnp.vstack((xx.ravel(), yy.ravel())).T
        G = gamma * jnp.ones_like(grid[..., 0])
        psph = pm.sphere.hsph2euclid(G, grid[..., 0], grid[..., 1])
        ax = fig.add_subplot(1, num_slices, i+1, projection='mollweide')
        ax.contourf(xx, yy, embedded_sphere_density(psph).reshape(xx.shape), cmap=plt.cm.jet)
        ax.set_axis_off()
    ln = 'elbo' if args.elbo_loss else 'kl'
    plt.savefig(os.path.join('images', 'hyper-sphere-target-density-{}.png'.format(ln)))

if args.seed == 0:
    num_slices = 8
    fig = plt.figure(figsize=(10, 1), constrained_layout=True)
    for i, gamma in enumerate(jnp.linspace(0., jnp.pi, num_slices+2)[1:-1]):
        theta = jnp.linspace(-jnp.pi, jnp.pi)[1:-1]
        phi = jnp.linspace(-jnp.pi / 2, jnp.pi / 2)[1:-1]
        xx, yy = jnp.meshgrid(theta, phi)
        grid = jnp.vstack((xx.ravel(), yy.ravel())).T
        G = gamma * jnp.ones_like(grid[..., 0])
        psph = pm.sphere.hsph2euclid(G, grid[..., 0], grid[..., 1])
        dens = importance_density(rng_mw, bij_params, deq_params, 5000, psph)
        ax = fig.add_subplot(1, num_slices, i+1, projection='mollweide')
        ax.contourf(xx, yy, dens.reshape(xx.shape), cmap=plt.cm.jet)
        ax.set_axis_off()
    ln = 'elbo' if args.elbo_loss else 'kl'
    plt.savefig(os.path.join('images', 'hyper-sphere-approx-density-{}.png'.format(ln)))

