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
from prax.bijectors import realnvp, permute

from coordinates import sph2euclid, sph2latlon, hsph2euclid
from rejection_sampling import embedded_earth_density, embedded_sphere_density, embedded_hypersphere_density, rejection_sampling


parser = argparse.ArgumentParser(description='Density estimation for sphere distribution')
parser.add_argument('--num-steps', type=int, default=1000, help='Number of gradient descent iterations for score matching training')
parser.add_argument('--lr', type=float, default=1e-3, help='Gradient descent learning rate')
parser.add_argument('--num-batch', type=int, default=100, help='Number of samples per batch')
parser.add_argument('--density', type=str, default='sphere', help='Indicator of which density function on the sphere to use')
parser.add_argument('--elbo-loss', type=int, default=1, help='Flag to indicate using the ELBO loss')
parser.add_argument('--num-importance', type=int, default=20, help='Number of importance samples to draw during training; if the ELBO loss is used, this argument is ignored')
parser.add_argument('--seed', type=int, default=0, help='Pseudo-random number generator seed')
args = parser.parse_args()

sphere_density, num_dims = {
    'sphere': (embedded_sphere_density, 3),
    'earth': (embedded_earth_density, 3),
    'hyper': (embedded_hypersphere_density, 4)
}[args.density]

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
    """Projection of points in the ambient space to the sphere.

    Args:
        xamb: Observations in the ambient space.

    Returns:
        out: Projections to the surface of the sphere.

    """
    return _project(xamb)

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
    num_dims = x.shape[-1]
    num_masked = num_dims - 2
    perm = jnp.roll(jnp.arange(num_dims), 1)
    y = realnvp.forward(x, num_masked, params[0], fns[0])
    y = permute.forward(y, perm)
    y = realnvp.forward(y, num_masked, params[1], fns[1])
    y = permute.forward(y, perm)
    y = realnvp.forward(y, num_masked, params[2], fns[2])
    y = permute.forward(y, perm)
    y = realnvp.forward(y, num_masked, params[3], fns[3])
    y = permute.forward(y, perm)
    y = realnvp.forward(y, num_masked, params[4], fns[4])
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
    num_masked = num_dims - 2
    perm = jnp.roll(jnp.arange(num_dims), 1)
    fldj = 0.
    y = realnvp.inverse(y, num_masked, params[4], fns[4])
    fldj += realnvp.forward_log_det_jacobian(y, num_masked, params[4], fns[4])
    y = permute.inverse(y, perm)
    fldj += permute.forward_log_det_jacobian()
    y = realnvp.inverse(y, num_masked, params[3], fns[3])
    fldj += realnvp.forward_log_det_jacobian(y, num_masked, params[3], fns[3])
    y = permute.inverse(y, perm)
    fldj += permute.forward_log_det_jacobian()
    y = realnvp.inverse(y, num_masked, params[2], fns[2])
    fldj += realnvp.forward_log_det_jacobian(y, num_masked, params[2], fns[2])
    y = permute.inverse(y, perm)
    fldj += permute.forward_log_det_jacobian()
    y = realnvp.inverse(y, num_masked, params[1], fns[1])
    fldj += realnvp.forward_log_det_jacobian(y, num_masked, params[1], fns[1])
    y = permute.inverse(y, perm)
    fldj += permute.forward_log_det_jacobian()
    y = realnvp.inverse(y, num_masked, params[0], fns[0])
    fldj += realnvp.forward_log_det_jacobian(y, num_masked, params[0], fns[0])
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
    rad = pd.lognormal.sample(rng_rad, mu, sigma, [num_samples] + list(xsph.shape[:-1]))
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
    if args.elbo_loss:
        rng, rng_rej, rng_elbo, rng_deq = random.split(rng, 4)
        xsph = rejection_sampling(rng_rej, num_samples, num_dims, sphere_density)
        nelbo = negative_elbo(rng_elbo, bij_params, bij_fns, deq_params, deq_fn, xsph).mean()
        return nelbo
    else:
        rng, rng_rej, rng_is = random.split(rng, 3)
        xsph = rejection_sampling(rng_rej, num_samples, num_dims, sphere_density)
        log_is = importance_log_density(rng_is, bij_params, bij_fns, deq_params, deq_fn, args.num_importance, xsph)
        log_target = jnp.log(sphere_density(xsph))
        return jnp.mean(log_target - log_is)

def zero_nans(g):
    """Remove the NaNs in a matrix by replaceing them with zeros.

    Args:
        g: Matrix whose NaN elements should be replaced by zeros.

    Returns:
        out: The input matrix but with NaN elements replaced by zeros.

    """
    return jnp.where(jnp.isnan(g), jnp.zeros_like(g), g)

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
        loss_grad = tree_util.tree_map(zero_nans, loss_grad)
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
rng, rng_is, rng_kl, rng_mw = random.split(rng, 4)

# Generate the parameters of RealNVP bijectors.
bij_params, bij_fns = [], []
for i in range(5):
    p, f = network_factory(random.fold_in(rng_bij, i), num_dims - 2, 2)
    bij_params.append(p)
    bij_fns.append(f)

# Parameterize the mean and scale of a log-normal multiplicative dequantizer.
deq_params, deq_fn = network_factory(rng_deq, num_dims, 1)

# Estimate parameters of the dequantizer and ambient flow.
(bij_params, deq_params), trace = train(rng_train, bij_params, bij_fns, deq_params, deq_fn, args.num_steps, args.lr, args.num_batch)

# Sample using dequantization and rejection sampling.
xamb, xsph = sample_ambient(rng_xamb, 100000, bij_params, bij_fns, num_dims)
xobs = rejection_sampling(rng_xobs, len(xsph), num_dims, sphere_density)

# Compute comparison statistics.
mean_mse = jnp.square(jnp.linalg.norm(xsph.mean(0) - xobs.mean(0)))
cov_mse = jnp.square(jnp.linalg.norm(jnp.cov(xsph.T) - jnp.cov(xobs.T)))
approx = importance_density(rng_kl, bij_params, bij_fns, deq_params, deq_fn, 100, xsph)
target = sphere_density(xsph)
w = target / approx
Z = jnp.mean(w)
log_approx = jnp.log(approx)
log_target = jnp.log(target)
kl = jnp.mean(log_approx - log_target) + jnp.log(Z)
ess = jnp.square(jnp.sum(w)) / jnp.sum(jnp.square(w))
ress = 100 * ess / len(w)
print('estimate KL(q||p): {:.5f} - relative effective sample size: {:.2f}%'.format(kl, ress))


# Visualize learned distribution.
if num_dims == 3:
    lat, lon = sph2latlon(xsph)
    theta = jnp.linspace(-jnp.pi, jnp.pi)[1:-1]
    phi = jnp.linspace(-jnp.pi / 2, jnp.pi / 2)[1:-1]
    xx, yy = jnp.meshgrid(theta, phi)
    grid = jnp.vstack((xx.ravel(), yy.ravel())).T
    psph = sph2euclid(grid[..., 0], grid[..., 1])
    dens = importance_density(rng_mw, bij_params, bij_fns, deq_params, deq_fn, 5000, psph)
    fig = plt.figure(figsize=(13, 5))
    ax = fig.add_subplot(141)
    ax.plot(trace)
    ax.grid(linestyle=':')
    ax.set_ylabel('Combined Loss')
    ax.set_xlabel('Gradient Descent Iteration')
    num_plot = 10000
    ax = fig.add_subplot(142, projection='3d')
    ax.plot(xobs[:num_plot, 0], xobs[:num_plot, 1], xobs[:num_plot, 2], '.', alpha=0.2, label='Rejection Sampling')
    ax.plot(xsph[:num_plot, 0], xsph[:num_plot, 1], xsph[:num_plot, 2], '.', alpha=0.2, label='Dequantization Sampling')
    ax.grid(linestyle=':')
    leg = ax.legend()
    for lh in leg.legendHandles:
        lh._legmarker.set_alpha(1)
    ax = fig.add_subplot(143, projection='mollweide')
    ax.scatter(lon, lat, c=approx, vmin=0., vmax=jnp.quantile(approx, 0.97))
    ax.set_axis_off()
    ax.set_title('Approximate Density')
    ax = fig.add_subplot(144, projection='mollweide')
    ax.scatter(lon, lat, c=target)
    ax.set_axis_off()
    ax.set_title('Target Density')
    plt.suptitle('Mean MSE: {:.5f} - Covariance MSE: {:.5f} - KL$(q\Vert p)$ = {:.5f} - Rel. ESS: {:.2f}%'.format(mean_mse, cov_mse, kl, ress))
    plt.tight_layout()
    ln = 'elbo' if args.elbo_loss else 'kl'
    plt.savefig(os.path.join('images', '{}-density-{}-num-importance-{}.png'.format(args.density, ln, args.num_importance)))
elif num_dims == 4:
    num_slices = 8
    fig = plt.figure(figsize=(10, 3))
    for i, gamma in enumerate(jnp.linspace(0., jnp.pi, num_slices+2)[1:-1]):
        theta = jnp.linspace(-jnp.pi, jnp.pi)[1:-1]
        phi = jnp.linspace(-jnp.pi / 2, jnp.pi / 2)[1:-1]
        xx, yy = jnp.meshgrid(theta, phi)
        grid = jnp.vstack((xx.ravel(), yy.ravel())).T
        G = gamma * jnp.ones_like(grid[..., 0])
        psph = hsph2euclid(G, grid[..., 0], grid[..., 1])
        dens = importance_density(rng_mw, bij_params, bij_fns, deq_params, deq_fn, 5000, psph)
        ax = fig.add_subplot(2, num_slices, i+1, projection='mollweide')
        ax.contourf(xx, yy, embedded_hypersphere_density(psph).reshape(xx.shape))
        ax.set_axis_off()
        ax.set_title('$\gamma$ = {:.3f}\nAn.'.format(gamma))
        ax = fig.add_subplot(2, num_slices, i+num_slices+1, projection='mollweide')
        ax.contourf(xx, yy, dens.reshape(xx.shape))
        ax.set_axis_off()
        ax.set_title('App.')
    plt.suptitle('KL$(q\Vert p)$ = {:.5f} - Relative ESS: {:.2f}%'.format(kl, ress))
    plt.tight_layout()
    ln = 'elbo' if args.elbo_loss else 'kl'
    plt.savefig(os.path.join('images', 'hyper-sphere-density-{}.png'.format(ln)))

