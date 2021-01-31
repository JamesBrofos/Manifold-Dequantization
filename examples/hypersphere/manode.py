import argparse
import os
from functools import partial
from typing import Callable, List, Tuple

import matplotlib.pyplot as plt

import jax.numpy as jnp
from jax import random, tree_util
from jax import jacobian, jit, value_and_grad, vmap
from jax.experimental import optimizers, stax
from jax.experimental.ode import odeint
from jax.config import config
config.update("jax_enable_x64", True)

from distributions import embedded_sphere_density
from rejection_sampling import rejection_sampling
from sphere import ambient_to_spherical_vector_field, exp, log, log_det_jac_exp, project_to_sphere, sample_uniform, spherical_to_chart_vector_field, uniform_log_density

parser = argparse.ArgumentParser(description='Mobius spline flow on hypersphere')
parser.add_argument('--num-steps', type=int, default=20000, help='Number of gradient descent iterations for score matching training')
parser.add_argument('--lr', type=float, default=1e-3, help='Gradient descent learning rate')
parser.add_argument('--num-samples', type=int, default=100, help='Number of samples per batch')
parser.add_argument('--seed', type=int, default=0, help='Pseudo-random number generator seed')
args = parser.parse_args()


def kl_divergence(params: List, rng: random.PRNGKey, num_samples: int) -> float:
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
    s, log_prob = manifold_ode_log_prob(params, rng, num_samples)
    log_prob_target = jnp.log(embedded_sphere_density(s))
    div = jnp.mean(log_prob - log_prob_target)
    return div

stacked = lambda x, t: jnp.hstack((x, t*jnp.ones(list(x.shape[:-1]) + [1])))

def manifold_ode_log_prob(params: List, rng: random.PRNGKey, num_samples: int) -> Tuple[jnp.ndarray]:
    """Forward model of the neural manifold ODE. The base distribution is uniform
    on the sphere. Computes both the samples from the forward model and the
    log-probability of the generated samples.

    Args:
        params: Parameters of the neural manifold ODE.
        rng: Pseudo-random number generator key.
        num_samples: Number of samples use to estimate the KL divergence.

    Returns:
        sph: samples generated from the neural manifold ODE under the forward
            model.
        log_prob: The log-probability of the generated samples.

    """
    rng, rng_x, rng_b = random.split(rng, 3)
    x = sample_uniform(rng_x, [num_samples, 4])
    b = project_to_sphere(x + 0.01 * random.normal(rng_b, x.shape))
    v = log(b, x)
    fldj = log_det_jac_exp(b, v)
    vector_field = ambient_to_spherical_vector_field(lambda x, t: net(params, stacked(x, t)))
    cfunc, divfunc = spherical_to_chart_vector_field(b, vector_field)
    init = (v, jnp.zeros(len(v)))
    time = jnp.array([0.0, 1.0])
    tang, trace = tuple(_[-1] for _ in odeint(divfunc, init, time))
    sph = exp(b, tang)
    ildj = -log_det_jac_exp(b, tang)
    log_prob = uniform_log_density(x) + fldj + ildj + trace
    return sph, log_prob

def manifold_reverse_ode_log_prob(params: List, rng: random.PRNGKey, revx: jnp.ndarray) -> jnp.ndarray:
    """Given observations, compute their log-likelihood under the neural manifold
    ODE by integrating the dynamics backwards and applying the
    change-of-variables formula (computed continuously).

    Args:
        params: Parameters of the neural manifold ODE.
        rng: Pseudo-random number generator key.
        revx: Observations whose log-likelihood under the neural ODE model
            should be computed.

    Returns:
        rev_log_prob: The log-probability of the observations.

    """
    b = project_to_sphere(revx + 0.01 * random.normal(rng, revx.shape))
    vrev = log(b, revx)
    revfldj = log_det_jac_exp(b, vrev)
    revinit = (vrev, jnp.zeros(len(vrev)))
    vector_field = ambient_to_spherical_vector_field(lambda x, t: net(params, stacked(x, t)))
    cfunc, divfunc = spherical_to_chart_vector_field(b, vector_field)
    revfunc = lambda x, t: tuple(-_ for _ in divfunc(x, 1.0 - t))
    time = jnp.array([0.0, 1.0])
    revtang, revtrace = tuple(_[-1] for _ in odeint(revfunc, revinit, time))
    revsph = exp(b, revtang)
    revildj = -log_det_jac_exp(b, revtang)
    rev_log_prob = uniform_log_density(revsph) - revfldj - revildj - revtrace
    return rev_log_prob


net_init, net = stax.serial(
    stax.Dense(100), stax.Tanh,
    stax.Dense(100), stax.Tanh,
    stax.Dense(100), stax.Tanh,
    stax.Dense(4))
opt_init, opt_update, get_params = optimizers.adam(args.lr)

@partial(jit, static_argnums=(2, ))
def step(opt_state: optimizers.OptimizerState, it: int, num_samples: int):
    """Stochastic batch gradient descent on the KL divergence where the underlying
    model is the neural manifold ODE.

    Args:
        opt_state: The state of the parameters of the neural manifold ODE.
        it: Gradient descent iteration number.
        num_samples: Number of samples to use in each minibatch.

    Returns:
        out: The optimizer state following a single optimization step.

    """
    rng = random.PRNGKey(it)
    params = get_params(opt_state)
    kl, grads = value_and_grad(kl_divergence)(params, rng, num_samples)
    return opt_update(it, grads, opt_state), kl


def main():
    # Set pseudo-random number generator keys.
    rng = random.PRNGKey(args.seed)
    rng, rng_net = random.split(rng, 2)
    rng, rng_sample, rng_xobs, rng_basis = random.split(rng, 4)
    rng, rng_fwd, rng_rev = random.split(rng, 3)
    rng, rng_kl = random.split(rng, 2)

    # Initialize the parameters of the ambient vector field network.
    _, params = net_init(rng_net, (-1, 5))
    opt_state = opt_init(params)
    count = lambda x: jnp.prod(jnp.array(x.shape))
    num_params = jnp.array(tree_util.tree_map(count, tree_util.tree_flatten(params)[0])).sum()
    print('number of parameters: {}'.format(num_params))

    for it in range(args.num_steps):
        opt_state, kl = step(opt_state, it, args.num_samples)
        print('iter.: {} - kl: {:.4f}'.format(it, kl))

    params = get_params(opt_state)

    # Compute comparison statistics.
    xsph, log_approx = manifold_ode_log_prob(params, rng_sample, 10000)
    xobs = rejection_sampling(rng_xobs, len(xsph), 4, embedded_sphere_density)
    mean_mse = jnp.square(jnp.linalg.norm(xsph.mean(0) - xobs.mean(0)))
    cov_mse = jnp.square(jnp.linalg.norm(jnp.cov(xsph.T) - jnp.cov(xobs.T)))
    approx = jnp.exp(log_approx)
    target = embedded_sphere_density(xsph)
    w = target / approx
    Z = jnp.nanmean(w)
    log_approx = jnp.log(approx)
    log_target = jnp.log(target)
    klqp = jnp.nanmean(log_approx - log_target) + jnp.log(Z)
    ess = jnp.square(jnp.nansum(w)) / jnp.nansum(jnp.square(w))
    ress = 100 * ess / len(w)
    del w, Z, log_approx, approx, log_target, target
    log_approx = manifold_reverse_ode_log_prob(params, rng_kl, xobs)
    approx = jnp.exp(log_approx)
    target = embedded_sphere_density(xobs)
    w = approx / target
    Z = jnp.nanmean(w)
    log_target = jnp.log(target)
    klpq = jnp.nanmean(log_target - log_approx) + jnp.log(Z)
    del w, Z, log_approx, approx, log_target, target
    print('ode - Mean MSE: {:.5f} - Covariance MSE: {:.5f} - KL$(q\Vert p)$ = {:.5f} - KL$(p\Vert q)$ = {:.5f} - Rel. ESS: {:.2f}%'.format(mean_mse, cov_mse, klqp, klpq, ress))

    # x, log_prob = manifold_ode_log_prob(params, rng_fwd, 10)
    # rev_log_prob = manifold_reverse_ode_log_prob(params, rng_rev, x)
    # print(log_prob - rev_log_prob)

    # obs, log_prob = manifold_ode_log_prob(params, rng_sample, 1000000)
    # prob = jnp.exp(log_prob)
    # base = project_to_sphere(random.normal(rng_basis, [4]))
    # B = random.normal(rng, [4, 3])
    # Bp = jnp.concatenate((base[..., jnp.newaxis], B), axis=-1)
    # O = jnp.linalg.qr(Bp)[0][:, 1:]
    # ec = log(base, obs)@O
    # det = jnp.exp(-log_det_jac_exp(base, log(base, obs)))

    # # Compute the density in the tangent space and compare to the empirical
    # # probability of lying in a small region of the tangent space.
    # eprob = prob / det
    # delta = 0.1
    # for i in range(50):
    #     p = ec[i]
    #     pr = eprob[i] * delta**2
    #     idx0 = jnp.abs(ec[:, 0] - p[0]) < delta / 2.
    #     idx1 = jnp.abs(ec[:, 1] - p[1]) < delta / 2.
    #     idx2 = jnp.abs(ec[:, 2] - p[2]) < delta / 2.
    #     pr_est = jnp.mean(idx0 & idx1 & idx2)
    #     print('prob.: {:.10f} - estim. prob.: {:.10f}'.format(pr, pr_est))

if __name__ == '__main__':
    main()
