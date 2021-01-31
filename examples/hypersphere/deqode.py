import argparse
import os
from functools import partial
from typing import Callable, List, Sequence, Tuple

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import jax.numpy as jnp
import jax.scipy.special as jspsp
import jax.scipy.stats as jspst
from jax import lax, nn, random, tree_util
from jax import grad, jacobian, jit, value_and_grad, vmap
from jax.experimental import optimizers, stax
from jax.experimental.ode import odeint
from jax.config import config
config.update("jax_enable_x64", True)

import prax.distributions as pd
import prax.manifolds as pm
import prax.utils as put
from prax.bijectors import realnvp, permute

from distributions import embedded_sphere_density
from rejection_sampling import rejection_sampling

parser = argparse.ArgumentParser(description='Density estimation for sphere distribution')
parser.add_argument('--num-steps', type=int, default=20000, help='Number of gradient descent iterations for score matching training')
parser.add_argument('--lr', type=float, default=1e-3, help='Gradient descent learning rate')
parser.add_argument('--elbo-loss', type=int, default=1, help='Flag to indicate using the ELBO loss')
parser.add_argument('--num-importance', type=int, default=20, help='Number of importance samples to draw during training; if the ELBO loss is used, this argument is ignored')
parser.add_argument('--num-samples', type=int, default=100, help='Number of samples per batch')
parser.add_argument('--seed', type=int, default=0, help='Pseudo-random number generator seed')
args = parser.parse_args()


deq_init, deq_net = stax.serial(
        stax.Dense(10), stax.Relu,
        stax.Dense(10), stax.Relu,
        stax.FanOut(2),
        stax.parallel(stax.Dense(1),
                      stax.serial(stax.Dense(1), stax.Softplus))
)
net_init, net = stax.serial(
    stax.Dense(100), stax.Tanh,
    stax.Dense(100), stax.Tanh,
    stax.Dense(100), stax.Tanh,
    stax.Dense(4))
opt_init, opt_update, get_params = optimizers.adam(args.lr)

def dequantize(rng: jnp.ndarray, deq_params: Sequence[jnp.ndarray], xsph: jnp.ndarray, num_samples: int) -> Tuple[jnp.ndarray]:
    """Dequantize observations on the sphere into the ambient space.

    Args:
        rng: Pseudo-random number generator seed.
        deq_params: Parameters of the mean and scale functions used in
            the log-normal dequantizer.
        xsph: Observations on the sphere.
        num_samples: Number of dequantization samples.

    Returns:
        out: A tuple containing the dequantized samples and the log-density of
            the dequantized samples.

    """
    num_dims = xsph.shape[-1]
    # Dequantization parameters.
    mu, sigma = deq_net(deq_params, xsph)
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

def primal_to_augmented(vector_field: Callable) -> Callable:
    def divergence(x, t, *args):
        g = lambda x: vector_field(x, t, *args)
        return jnp.trace(vmap(jacobian(g))(x), axis1=-2, axis2=-1).squeeze()

    def divfunc(state, t, *args):
        v, _ = state
        f = vector_field(v, t, *args)
        d = -divergence(v, t, *args)
        return f, d

    return divfunc

stacked = lambda x, t: jnp.concatenate([x, t*jnp.ones(list(x.shape[:-1]) + [1])], axis=-1)
project = lambda x: x / jnp.linalg.norm(x, axis=-1, keepdims=True)

def ode_forward(rng: random.PRNGKey, net_params: List[jnp.ndarray], num_samples: int, num_dims: int) -> Tuple[jnp.ndarray]:
    x = random.normal(rng, [num_samples, num_dims])
    log_prob_prior = jspst.norm.logpdf(x).sum(axis=-1)
    vector_field = lambda x, t: net(net_params, stacked(x, t))
    divfunc = primal_to_augmented(vector_field)
    init = (x, jnp.zeros(len(x)))
    time = jnp.array([0.0, 1.0])
    xfwd, trace = tuple(_[-1] for _ in odeint(divfunc, init, time))
    log_prob = log_prob_prior + trace
    xsph = project(xfwd)
    return xfwd, xsph, log_prob

def ode_reverse(net_params: List[jnp.ndarray], xrev: jnp.ndarray) -> Tuple[jnp.ndarray]:
    num_dims = xrev.shape[-1]
    vector_field = lambda x, t: net(net_params, stacked(x, t))
    divfunc = primal_to_augmented(vector_field)
    revfunc = lambda x, t: tuple(-_ for _ in divfunc(x, 1.0 - t))
    revinit = (xrev, jnp.zeros(len(xrev)))
    time = jnp.array([0.0, 1.0])
    yrev, revtrace = tuple(_[-1] for _ in odeint(revfunc, revinit, time))
    log_prob_prior = jspst.norm.logpdf(yrev).sum(axis=-1)
    rev_log_prob = log_prob_prior - revtrace
    return rev_log_prob

def importance_log_density(rng: jnp.ndarray, net_params: Sequence[jnp.ndarray], deq_params: Sequence[jnp.ndarray], num_is: int, xsph: jnp.ndarray) -> jnp.ndarray:
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
    xdeq, deq_log_dens = dequantize(rng, deq_params, xsph, num_is)
    amb_log_dens = ode_reverse(net_params, xdeq.reshape(-1, 4)).reshape(xdeq.shape[:-1])
    is_log_dens = jspsp.logsumexp(amb_log_dens - deq_log_dens, axis=0) - jnp.log(num_is)
    return is_log_dens

@partial(jit, static_argnums=(3, ))
def importance_density(rng: jnp.ndarray, net_params: Sequence[jnp.ndarray], deq_params: Sequence[jnp.ndarray], num_is: int, xsph: jnp.ndarray) -> jnp.ndarray:
    """Compute the estimate of the density on the sphere via importance sampling.
    The calculation is encapsulated in a scan so that a large number of
    importance samples may be used without running out of memory.

    Args:
        rng: Pseudo-random number generator seed.
        bij_params: List of arrays parameterizing the RealNVP bijectors.
        deq_params: Parameters of the mean and scale functions used in
            the log-normal dequantizer.
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
        log_prob = importance_log_density(rng_step, net_params, deq_params, num_is, p)
        prob = jnp.exp(log_prob)
        return it + 1, prob
    _, prob = lax.scan(step, 0, xsph)
    return prob

def kl_divergence(net_params: List[jnp.ndarray], deq_params: List[jnp.ndarray], rng: random.PRNGKey, num_samples: int) -> float:
    """Computes the KL divergence between the target density and the neural
    manifold ODE's distribution on the sphere. Note that the target density is
    unnormalized.

    Args:
        params: Parameters of the neural manifold ODE.
        rng: Pseudo-random number generator key.
        num_samples: Number of samples use to estimate the KL divergence.

    Returns:
        div: The estimated KL divergence.

    """
    rng, rng_fwd, rng_is = random.split(rng, 3)
    _, xsph, _ = ode_forward(rng_fwd, net_params, num_samples, 4)
    log_prob = importance_log_density(rng_is, net_params, deq_params, 10, xsph)
    log_prob_target = jnp.log(embedded_sphere_density(xsph))
    div = jnp.mean(log_prob - log_prob_target)
    return div

def negative_elbo(rng: jnp.ndarray, net_params: Sequence[jnp.ndarray], deq_params: Sequence[jnp.ndarray], xsph: jnp.ndarray) -> jnp.ndarray:
    """Compute the negative evidence lower bound of the dequantizing distribution.

    Args:
        rng: Pseudo-random number generator seed.
        bij_params: List of arrays parameterizing the RealNVP bijectors.
        deq_params: Parameters of the mean and scale functions used in
            the log-normal dequantizer.
        xsph: Observations on the sphere.

    Returns:
        nelbo: The negative evidence lower bound.

    """
    xdeq, deq_log_dens = dequantize(rng, deq_params, xsph, 1)
    amb_log_dens = ode_reverse(net_params, xdeq.reshape(-1, 4)).reshape(xdeq.shape[:-1])
    elbo = jnp.mean(amb_log_dens - deq_log_dens, axis=0)
    nelbo = -elbo
    return nelbo

def loss(net_params: List[jnp.ndarray], deq_params: List[jnp.ndarray], rng: random.PRNGKey, num_samples: int) -> float:
    rng, rng_rej, rng_loss = random.split(rng, 3)
    xsph = rejection_sampling(rng_rej, num_samples, 4, embedded_sphere_density)
    if args.elbo_loss:
        nelbo = negative_elbo(rng_loss, net_params, deq_params, xsph).mean()
        return nelbo
    else:
        log_is = importance_log_density(rng_loss, net_params, deq_params, args.num_importance, xsph)
        log_target = jnp.log(embedded_sphere_density(xsph))
        return jnp.mean(log_target - log_is)


@partial(jit, static_argnums=(2, ))
def step(opt_state: optimizers.OptimizerState, it: int, num_samples: int):
    """Stochastic batch gradient descent on the KL divergence where the underlying
    model is the neural manifold ODE.

    Args:
        opt_state: The state of the parameters of the neural manifold ODE.
        it: Gradient descent iteration number.
        num_samples: Number of samples to use in each minibatch.

    Returns:
        opt_state: The optimizer state following a single optimization step.
        loss_value: The value of the loss at the current gradient descent iteration.

    """
    rng = random.PRNGKey(it)
    net_params, deq_params = get_params(opt_state)
    loss_value, grads = value_and_grad(loss, (0, 1))(net_params, deq_params, rng, num_samples)
    opt_state = opt_update(it, grads, opt_state)
    return opt_state, loss_value

def statistics(net_params: List[jnp.ndarray], deq_params: List[jnp.ndarray], rng: random.PRNGKey):
    # Split pseudo-random number key.
    rng, rng_sample, rng_xobs, rng_kl = random.split(rng, 4)
    # Compute comparison statistics.
    _, xsph, _ = ode_forward(rng_sample, net_params, 10000, 4)
    xobs = rejection_sampling(rng_xobs, len(xsph), 4, embedded_sphere_density)
    mean_mse = jnp.square(jnp.linalg.norm(xsph.mean(0) - xobs.mean(0)))
    cov_mse = jnp.square(jnp.linalg.norm(jnp.cov(xsph.T) - jnp.cov(xobs.T)))
    approx = importance_density(rng_kl, net_params, deq_params, 10000, xsph)
    log_approx = jnp.log(approx)
    target = embedded_sphere_density(xsph)
    w = target / approx
    Z = jnp.nanmean(w)
    log_approx = jnp.log(approx)
    log_target = jnp.log(target)
    klqp = jnp.nanmean(log_approx - log_target) + jnp.log(Z)
    ess = jnp.square(jnp.nansum(w)) / jnp.nansum(jnp.square(w))
    ress = 100 * ess / len(w)
    del w, Z, log_approx, approx, log_target, target, xsph
    approx = importance_density(rng_kl, net_params, deq_params, 10000, xobs)
    log_approx = jnp.log(approx)
    target = embedded_sphere_density(xobs)
    w = approx / target
    Z = jnp.nanmean(w)
    log_target = jnp.log(target)
    klpq = jnp.nanmean(log_target - log_approx) + jnp.log(Z)
    del w, Z, log_approx, approx, log_target, target
    method = 'deqode ({})'.format('ELBO' if args.elbo_loss else 'KL')
    print('{} - Mean MSE: {:.5f} - Covariance MSE: {:.5f} - KL$(q\Vert p)$ = {:.5f} - KL$(p\Vert q)$ = {:.5f} - Rel. ESS: {:.2f}%'.format(method, mean_mse, cov_mse, klqp, klpq, ress))


def main():
    # Create pseudo-random number generator keys.
    rng = random.PRNGKey(args.seed)
    rng, rng_deq, rng_net = random.split(rng, 3)
    rng, rng_stats = random.split(rng, 2)

    # Initialize parameters of the neural networks.
    _, deq_params = deq_init(rng_deq, (-1, 4))
    _, net_params = net_init(rng_net, (-1, 5))

    # Compute the number of parameters.
    count = lambda x: jnp.prod(jnp.array(x.shape))
    num_net_params = jnp.array(tree_util.tree_map(count, tree_util.tree_flatten(net_params)[0])).sum()
    num_deq_params = jnp.array(tree_util.tree_map(count, tree_util.tree_flatten(deq_params)[0])).sum()
    num_params = num_net_params + num_deq_params
    print('dequantization parameters: {} - ambient parameters: {} - number of parameters: {}'.format(num_deq_params, num_net_params, num_params))

    opt_state = opt_init((net_params, deq_params))

    for it in range(args.num_steps):
        opt_state, loss_value = step(opt_state, it, args.num_samples)
        print('iter.: {} - loss: {:.4f}'.format(it + 1, loss_value))

    net_params, deq_params = get_params(opt_state)
    statistics(net_params, deq_params, rng_stats)


if __name__ == '__main__':
    main()
