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
import prax.manifolds as pm
import prax.utils as put
from prax.bijectors import realnvp, permute

from distributions import correlated_torus_density, multimodal_torus_density, unimodal_torus_density
from rejection_sampling import embedded_torus_density, rejection_sampling


parser = argparse.ArgumentParser(description='Dequantization for distributions on the torus')
parser.add_argument('--num-steps', type=int, default=10000, help='Number of gradient descent iterations for score matching training')
parser.add_argument('--lr', type=float, default=1e-3, help='Gradient descent learning rate')
parser.add_argument('--num-batch', type=int, default=100, help='Number of samples per batch')
parser.add_argument('--density', type=str, default='correlated', help='Indicator of which density function on the torus to use')
parser.add_argument('--elbo-loss', type=int, default=1, help='Flag to indicate using the ELBO loss')
parser.add_argument('--num-importance', type=int, default=20, help='Number of importance samples to draw during training; if the ELBO loss is used, this argument is ignored')
parser.add_argument('--num-hidden', type=int, default=35, help='Number of hidden units used in the neural networks')
parser.add_argument('--num-realnvp', type=int, default=3, help='Number of RealNVP bijectors to employ')
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
    num_dims = x.shape[-1]
    num_masked = num_dims - 2
    perm = jnp.array([1, 3, 2, 0])
    y = x
    for i in range(args.num_realnvp):
        y = realnvp.forward(y, num_masked, params[i], fns[i])
        y = permute.forward(y, perm)
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
    perm = jnp.array([1, 3, 2, 0])
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
    those samples to the torus.

    Args:
        rng: Pseudo-random number generator seed.
        num_samples: Number of samples to generate.
        bij_params: List of arrays parameterizing the RealNVP bijectors.
        bij_fns: List of functions that compute the shift and scale of the RealNVP
            affine transformation.
        num_dims: Dimensionality of samples.

    Returns:
        xamb, xtor: A tuple containing the ambient samples and the projection of
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
    rada = pd.lognormal.rvs(rng_rada, mua, sigmaa, [num_samples] + list(xtor.shape[:-1]))
    radb = pd.lognormal.rvs(rng_radb, mub, sigmab, [num_samples] + list(xtor.shape[:-1]))
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
    """Loss function implementation. Depending on the `elbo_loss` command line
    argument, we will either try to minimize the negative ELBO or the KL(p || q)
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
        out: The computed loss function.

    """
    if args.elbo_loss:
        rng, rng_rej, rng_elbo, rng_deq = random.split(rng, 4)
        xang = rejection_sampling(rng_rej, num_samples, torus_density)
        xtor = pm.torus.ang2euclid(xang)
        nelbo = negative_elbo(rng_elbo, bij_params, bij_fns, deq_params, deq_fn, xtor).mean()
        return nelbo
    else:
        rng, rng_rej, rng_is = random.split(rng, 3)
        xang = rejection_sampling(rng_rej, num_samples, torus_density)
        xtor = pm.torus.ang2euclid(xang)
        log_is = importance_log_density(rng_is, bij_params, bij_fns, deq_params, deq_fn, args.num_importance, xtor)
        log_target = jnp.log(torus_density(xang))
        return jnp.mean(log_target - log_is)

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
        loss_grad = tree_util.tree_map(partial(put.clip_and_zero_nans, clip_value=1.), loss_grad)
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
for i in range(args.num_realnvp):
    p, f = network_factory(random.fold_in(rng_bij, i), 2, 2, args.num_hidden)
    bij_params.append(p)
    bij_fns.append(f)

# Parameterize the mean and scale of a log-normal multiplicative dequantizer.
deq_params, deq_fn = network_factory(rng_deq, 4, 2, args.num_hidden)

if not args.elbo_loss:
    print('rescaling initialization')
    bij_params = tree_util.tree_map(lambda x: x / 2., bij_params)

# Compute the number of parameters.
count = lambda x: jnp.prod(jnp.array(x.shape))
num_bij_params = jnp.array(tree_util.tree_map(count, tree_util.tree_flatten(bij_params)[0])).sum()
num_deq_params = jnp.array(tree_util.tree_map(count, tree_util.tree_flatten(deq_params)[0])).sum()
num_params = num_bij_params + num_deq_params
print('dequantization parameters: {} - ambient parameters: {} - number of parameters: {}'.format(num_deq_params, num_bij_params, num_params))

# Estimate parameters of the dequantizer and ambient flow.
(bij_params, deq_params), trace = train(rng_train, bij_params, bij_fns, deq_params, deq_fn, args.num_steps, args.lr, args.num_batch)

# Sample from the learned distribution.
xamb, xtor = sample_ambient(rng_xamb, 100000, bij_params, bij_fns, 4)
xang = pm.torus.euclid2ang(xtor)
xobs = rejection_sampling(rng_xobs, len(xtor), torus_density)
xobs = pm.torus.euclid2ang(pm.torus.ang2euclid(xobs))

# Compute comparison statistics.
mean_mse = jnp.square(jnp.linalg.norm(xang.mean(0) - xobs.mean(0)))
cov_mse = jnp.square(jnp.linalg.norm(jnp.cov(xang.T) - jnp.cov(xobs.T)))
approx = importance_density(rng_kl, bij_params, bij_fns, deq_params, deq_fn, 1000, xtor)
target = embedded_torus_density(xtor, torus_density)
w = target / approx
Z = jnp.mean(w)
log_approx = jnp.log(approx)
log_target = jnp.log(target)
klqp = jnp.mean(log_approx - log_target) + jnp.log(Z)
ess = jnp.square(jnp.sum(w)) / jnp.sum(jnp.square(w))
ress = 100 * ess / len(w)
del w, Z, log_approx, approx, log_target, target
approx = importance_density(rng_kl, bij_params, bij_fns, deq_params, deq_fn, 1000, pm.torus.ang2euclid(xobs))
target = embedded_torus_density(pm.torus.ang2euclid(xobs), torus_density)
w = approx / target
Z = jnp.mean(w)
log_approx = jnp.log(approx)
log_target = jnp.log(target)
klpq = jnp.mean(log_target - log_approx) + jnp.log(Z)
del w, Z, log_approx, approx, log_target, target
method = 'dequantization ({})'.format('ELBO' if args.elbo_loss else 'KL')
print('{} - {} - seed: {} - Mean MSE: {:.5f} - Covariance MSE: {:.5f} - KL$(q\Vert p)$ = {:.5f} - KL$(p\Vert q)$ = {:.5f} - Rel. ESS: {:.2f}%'.format(method, args.density, args.seed, mean_mse, cov_mse, klqp, klpq, ress))

# Compute density on a grid.
lin = jnp.linspace(-jnp.pi, jnp.pi)
xx, yy = jnp.meshgrid(lin, lin)
theta = jnp.vstack((xx.ravel(), yy.ravel())).T
ptor = pm.torus.ang2euclid(theta)
prob = importance_density(rng_is, bij_params, bij_fns, deq_params, deq_fn, 10000, ptor)
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
plt.suptitle('Mean MSE: {:.5f} - Covariance MSE: {:.5f} - KL$(q\Vert p)$ = {:.5f} - Rel. ESS: {:.2f}%'.format(mean_mse, cov_mse, klqp, ress))
plt.tight_layout()
ln = 'elbo' if args.elbo_loss else 'kl'
plt.savefig(os.path.join('images', '{}-{}-num-batch-{}-num-importance-{}-num-steps-{}-seed-{}.png'.format(ln, args.density, args.num_batch, args.num_importance, args.num_steps, args.seed)))
