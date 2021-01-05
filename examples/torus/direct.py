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

from prax.bijectors import realnvp, permute
import prax.utils as put

from distributions import correlated_torus_density, multimodal_torus_density, unimodal_torus_density
from rejection_sampling import embedded_torus_density, rejection_sampling

parser = argparse.ArgumentParser(description='Dequantization for distributions on the torus')
parser.add_argument('--num-steps', type=int, default=20000, help='Number of gradient descent iterations for score matching training')
parser.add_argument('--lr', type=float, default=1e-3, help='Gradient descent learning rate')
parser.add_argument('--num-batch', type=int, default=100, help='Number of samples per batch')
parser.add_argument('--density', type=str, default='unimodal', help='Indicator of which density function on the torus to use')
parser.add_argument('--num-hidden', type=int, default=40, help='Number of hidden units used in the neural networks')
parser.add_argument('--num-realnvp', type=int, default=3, help='Number of RealNVP bijectors to employ')
parser.add_argument('--beta', type=float, default=1., help='Density concentration parameter')
parser.add_argument('--seed', type=int, default=0, help='Pseudo-random number generator seed')
args = parser.parse_args()

torus_density_uw = {
    'correlated': correlated_torus_density,
    'multimodal': multimodal_torus_density,
    'unimodal': unimodal_torus_density
}[args.density]
torus_density = lambda theta: jnp.exp(args.beta * jnp.log(torus_density_uw(theta)))

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
    num_masked = num_dims - 1
    perm = jnp.array([1, 0])
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
    num_masked = num_dims - 1
    perm = jnp.array([1, 0])
    fldj = 0.
    for i in reversed(range(args.num_realnvp)):
        y = permute.inverse(y, perm)
        fldj += permute.forward_log_det_jacobian()
        y = realnvp.inverse(y, num_masked, params[i], fns[i])
        fldj += realnvp.forward_log_det_jacobian(y, num_masked, params[i], fns[i])
    logprob = jspst.multivariate_normal.logpdf(y, jnp.zeros((num_dims, )), 1.)
    return logprob - fldj

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

nan2ninf = lambda g: jnp.where(jnp.isnan(g), jnp.zeros_like(g) - jnp.inf, g)

def induced_torus_log_density(bij_params, bij_fns, xtor):
    lp = jnp.array([
        nan2ninf(ambient_flow_log_prob(bij_params, bij_fns, xtor)),
        # First dimension.
        nan2ninf(ambient_flow_log_prob(bij_params, bij_fns, xtor + jnp.array([2.0*jnp.pi, 0.0]))),
        nan2ninf(ambient_flow_log_prob(bij_params, bij_fns, xtor + jnp.array([4.0*jnp.pi, 0.0]))),
        nan2ninf(ambient_flow_log_prob(bij_params, bij_fns, xtor - jnp.array([2.0*jnp.pi, 0.0]))),
        nan2ninf(ambient_flow_log_prob(bij_params, bij_fns, xtor - jnp.array([4.0*jnp.pi, 0.0]))),
        # Second dimension.
        nan2ninf(ambient_flow_log_prob(bij_params, bij_fns, xtor + jnp.array([0.0, 2.0*jnp.pi]))),
        nan2ninf(ambient_flow_log_prob(bij_params, bij_fns, xtor + jnp.array([0.0, 4.0*jnp.pi]))),
        nan2ninf(ambient_flow_log_prob(bij_params, bij_fns, xtor - jnp.array([0.0, 2.0*jnp.pi]))),
        nan2ninf(ambient_flow_log_prob(bij_params, bij_fns, xtor - jnp.array([0.0, 4.0*jnp.pi]))),
        # Diagonal.
        nan2ninf(ambient_flow_log_prob(bij_params, bij_fns, xtor + jnp.array([2.0*jnp.pi, 2.0*jnp.pi]))),
        nan2ninf(ambient_flow_log_prob(bij_params, bij_fns, xtor + jnp.array([4.0*jnp.pi, 4.0*jnp.pi]))),
        nan2ninf(ambient_flow_log_prob(bij_params, bij_fns, xtor - jnp.array([2.0*jnp.pi, 2.0*jnp.pi]))),
        nan2ninf(ambient_flow_log_prob(bij_params, bij_fns, xtor - jnp.array([4.0*jnp.pi, 4.0*jnp.pi])))])
    lp = jspsp.logsumexp(lp, axis=0)
    return lp

def loss(rng: jnp.ndarray, bij_params: Sequence[jnp.ndarray], bij_fns: Sequence[Callable], num_samples: int) -> float:
    rng, rng_rej = random.split(rng, 2)
    xtor = rejection_sampling(rng_rej, num_samples, torus_density, args.beta)
    log_target = jnp.log(torus_density(xtor))
    log_approx = induced_torus_log_density(bij_params, bij_fns, xtor)
    return jnp.mean(log_target - log_approx)

@partial(jit, static_argnums=(2, 3, 5))
def train(rng: jnp.ndarray, bij_params: Sequence[jnp.ndarray], bij_fns: Sequence[Callable], num_steps: int, lr: float, num_samples: int) -> Tuple:
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
        bij_params = get_params(opt_state)
        loss_val, loss_grad = value_and_grad(loss, 1)(step_rng, bij_params, bij_fns, num_samples)
        loss_grad = tree_util.tree_map(partial(put.clip_and_zero_nans, clip_value=1.), loss_grad)
        opt_state = opt_update(it, loss_grad, opt_state)
        return opt_state, loss_val
    opt_state, trace = lax.scan(step, opt_init(bij_params), jnp.arange(num_steps))
    bij_params = get_params(opt_state)
    return bij_params, trace


# Set random number generation seeds.
rng = random.PRNGKey(args.seed)
rng, rng_bij = random.split(rng, 2)
rng, rng_train = random.split(rng, 2)
rng, rng_xobs, rng_xamb = random.split(rng, 3)

# Generate the parameters of RealNVP bijectors.
bij_params, bij_fns = [], []
for i in range(args.num_realnvp):
    p, f = network_factory(random.fold_in(rng_bij, i), 1, 1, args.num_hidden)
    bij_params.append(p)
    bij_fns.append(f)

# Compute the number of parameters.
count = lambda x: jnp.prod(jnp.array(x.shape))
num_params = jnp.array(tree_util.tree_map(count, tree_util.tree_flatten(bij_params)[0])).sum()
print('number of parameters: {}'.format(num_params))

bij_params = tree_util.tree_map(lambda x: x / 2., bij_params)

# Direct estimation on the torus.
bij_params, trace = train(rng_train, bij_params, bij_fns, args.num_steps, args.lr, 100)

# Sample from the learned distribution.
num_samples = 100000
num_dims = 2
xamb = random.normal(rng_xamb, [num_samples, num_dims])
xamb = forward(bij_params, bij_fns, xamb)
xtor = jnp.mod(xamb, 2.0*jnp.pi)
lp = induced_torus_log_density(bij_params, bij_fns, xtor)
xobs = rejection_sampling(rng_xobs, len(xtor), torus_density, args.beta)

# Compute comparison statistics.
mean_mse = jnp.square(jnp.linalg.norm(xtor.mean(0) - xobs.mean(0)))
cov_mse = jnp.square(jnp.linalg.norm(jnp.cov(xtor.T) - jnp.cov(xobs.T)))
approx = jnp.exp(lp)
target = torus_density(xtor)
w = target / approx
Z = jnp.nanmean(w)
log_approx = jnp.log(approx)
log_target = jnp.log(target)
klqp = jnp.nanmean(log_approx - log_target) + jnp.log(Z)
ess = jnp.square(jnp.nansum(w)) / jnp.nansum(jnp.square(w))
ress = 100 * ess / len(w)
del w, Z, log_approx, log_target
log_approx = induced_torus_log_density(bij_params, bij_fns, xobs)
approx = jnp.exp(log_approx)
target = torus_density(xobs)
log_target = jnp.log(target)
w = approx / target
Z = jnp.mean(w)
klpq = jnp.mean(log_target - log_approx) + jnp.log(Z)
del w, Z, log_approx, log_target
print('direct - {} - seed: {} - Mean MSE: {:.5f} - Covariance MSE: {:.5f} - KL$(q\Vert p)$ = {:.5f} - KL$(p\Vert q)$ = {:.5f} - Rel. ESS: {:.2f}%'.format(args.density, args.seed, mean_mse, cov_mse, klqp, klpq, ress))

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].hist2d(xtor[:, 0], xtor[:, 1], density=True, bins=50)
axes[0].axis('square')
axes[1].scatter(xtor[:, 0], xtor[:, 1], c=jnp.exp(lp))
axes[1].axis('square')
axes[1].set_xlim(0., 2.*jnp.pi)
axes[1].set_ylim(0., 2.*jnp.pi)
plt.tight_layout()
plt.savefig(os.path.join('images', 'direct-torus-{}.png'.format(args.density)))

