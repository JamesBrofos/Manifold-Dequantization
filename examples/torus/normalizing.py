import argparse
import os
from functools import partial
from typing import Callable, Sequence, Tuple

import matplotlib.pyplot as plt

import jax.numpy as jnp
from jax import lax, random, tree_util
from jax import jit, value_and_grad, vmap
from jax.experimental import optimizers, stax

import prax.manifolds as pm

from distributions import correlated_torus_density, multimodal_torus_density, unimodal_torus_density
from mobius import mobius_flow, mobius_log_prob
from rejection_sampling import rejection_sampling

parser = argparse.ArgumentParser(description='Mobius Flows on the Torus')
parser.add_argument('--num-steps', type=int, default=20000, help='Number of gradient descent iterations for score matching training')
parser.add_argument('--lr', type=float, default=1e-3, help='Gradient descent learning rate')
parser.add_argument('--num-batch', type=int, default=100, help='Number of samples per batch')
parser.add_argument('--density', type=str, default='unimodal', help='Indicator for which density to estimate')
parser.add_argument('--num-mobius', type=int, default=9, help='Number of Mobius transforms in convex combination')
parser.add_argument('--num-hidden', type=int, default=64, help='Number of hidden units used in the neural networks')
parser.add_argument('--beta', type=float, default=1., help='Density concentration parameter')
parser.add_argument('--seed', type=int, default=0, help='Pseudo-random number generator seed')
args = parser.parse_args()

torus_density_uw = {
    'unimodal': unimodal_torus_density,
    'correlated': correlated_torus_density,
    'multimodal': multimodal_torus_density
}[args.density]
torus_density = lambda theta: jnp.exp(args.beta * jnp.log(torus_density_uw(theta)))

def network_factory(rng: jnp.ndarray, num_in: int, num_out: int, num_hidden: int) -> Tuple:
    """Factory for producing neural networks and their parameterizations.

    Args:
        rng: Pseudo-random number generator seed.
        num_in: Number of inputs to the network.
        num_out: Number of output variables.
        num_hidden: Number of hidden units in the hidden layer.

    Returns:
        out: A tuple containing the network parameters and a callable function
            that returns the neural network output for the given input.

    """
    params_init, fn = stax.serial(
        stax.Dense(num_hidden), stax.Relu,
        stax.Dense(num_hidden), stax.Relu,
        stax.Dense(num_out))
    _, params = params_init(rng, (-1, num_in))
    return params, fn

compress = lambda w: 0.99 / (1. + jnp.linalg.norm(w, axis=-1, keepdims=True)) * w

def conditional(theta: jnp.ndarray, params: Sequence[jnp.ndarray], fn: Callable) -> jnp.ndarray:
    """Compute the parameters of the conditional distribution of the second angular
    coordinate given the first.

    """
    x = pm.circle.ang2euclid(theta)
    w = fn(params, pm.circle.ang2euclid(theta)).reshape((-1, args.num_mobius, 2))
    w = compress(w)
    return w

def torus_log_prob(wa, wb, unifa, unifb, thetaa, thetab):
    """Compute the log-density on the torus in terms of the two angular
    coordinates.

    """
    lpa = mobius_log_prob(unifa, wa)
    lpb = vmap(mobius_log_prob)(unifb, wb)
    log_prob = lpa + lpb
    return log_prob

def sample_torus(rng, omega, params, fn, num_samples):
    rng, rng_unifa, rng_unifb = random.split(rng, 3)
    unifa = 2.0*jnp.pi*random.uniform(rng_unifa, [num_samples])
    unifb = 2.0*jnp.pi*random.uniform(rng_unifb, [num_samples])
    wa = compress(omega)
    thetaa = mobius_flow(unifa, wa).mean(0)
    wb = conditional(thetaa, params, fn)
    thetab = vmap(mobius_flow, in_axes=(0, 0))(unifb, wb).mean(1)
    return (thetaa, thetab), (unifa, unifb), (wa, wb)

def loss(rng, omega, params, fn, num_samples):
    (thetaa, thetab), (unifa, unifb), (wa, wb) = sample_torus(rng, omega, params, fn, num_samples)
    theta = jnp.stack([thetaa, thetab], axis=-1)
    mlp = torus_log_prob(wa, wb, unifa, unifb, thetaa, thetab)
    target = torus_density(theta)
    log_target = jnp.log(target)
    return jnp.mean(mlp - log_target)

@partial(jit, static_argnums=(3, 4, 5))
def train(rng, omega, params, fn, num_samples, num_steps, lr):
    opt_init, opt_update, get_params = optimizers.adam(lr)
    def step(opt_state, it):
        step_rng = random.fold_in(rng, it)
        omega, params = get_params(opt_state)
        loss_val, loss_grad = value_and_grad(loss, (1, 2))(step_rng, omega, params, fn, num_samples)
        opt_state = opt_update(it, loss_grad, opt_state)
        return opt_state, loss_val
    opt_state, trace = lax.scan(step, opt_init((omega, params)), jnp.arange(num_steps))
    wa, params = get_params(opt_state)
    return (wa, params), trace


# Set random number generation seeds.
rng = random.PRNGKey(args.seed)
rng, rng_torus = random.split(rng, 2)
rng, rng_wa, rng_net = random.split(rng, 3)
rng, rng_train = random.split(rng, 2)
rng, rng_xobs = random.split(rng, 2)

params, fn = network_factory(rng_net, 2, args.num_mobius*2, args.num_hidden)
omega = random.normal(rng_wa, [args.num_mobius, 2])

# Compute number of parameters.
count = lambda x: jnp.prod(jnp.array(x.shape))
num_params_net = jnp.array(tree_util.tree_map(count, tree_util.tree_flatten(params)[0])).sum()
num_omega = count(omega)
num_params = num_params_net + num_omega
print('number of parameters: {}'.format(num_params))

# Train normalizing flow on the torus.
(omega, params), trace = train(rng, omega, params, fn, args.num_batch, args.num_steps, args.lr)

# Compute comparison statistics.
(thetaa, thetab), (unifa, unifb), (wa, wb) = sample_torus(rng_torus, omega, params, fn, 100000)
theta = jnp.stack([thetaa, thetab], axis=-1)
lpa = mobius_log_prob(unifa, wa)
log_approx = torus_log_prob(wa, wb, unifa, unifb, thetaa, thetab)
approx = jnp.exp(log_approx)
target = torus_density(theta)
log_target = jnp.log(target)
w = target / approx
Z = jnp.mean(w)
kl = jnp.mean(log_approx - log_target) + jnp.log(Z)
ess = jnp.square(jnp.sum(w)) / jnp.sum(jnp.square(w))
ress = 100 * ess / len(w)
xobs = rejection_sampling(rng_xobs, len(theta), torus_density, args.beta)
mean_mse = jnp.square(jnp.linalg.norm(theta.mean(0) - xobs.mean(0)))
cov_mse = jnp.square(jnp.linalg.norm(jnp.cov(theta.T) - jnp.cov(xobs.T)))
print('normalizing - {} - seed: {} - Mean MSE: {:.5f} - Covariance MSE: {:.5f} - KL$(q\Vert p)$ = {:.5f} - Rel. ESS: {:.2f}%'.format(args.density, args.seed, mean_mse, cov_mse, kl, ress))


fig, axes = plt.subplots(1, 4, figsize=(13, 4))
axes[0].hist2d(thetaa, thetab, density=True, bins=50)
axes[1].scatter(thetaa, thetab, c=approx)
axes[1].set_xlim(0., 2.*jnp.pi)
axes[1].set_ylim(0., 2.*jnp.pi)
axes[2].hist(thetaa, density=True, bins=50)
axes[2].plot(thetaa, jnp.exp(lpa), '.')
axes[2].grid(linestyle=':')
axes[3].plot(trace)
axes[3].grid(linestyle=':')
plt.suptitle('KL$(q\Vert p)$ = {:.5f} - Rel. ESS: {:.2f}%'.format(kl, ress))
plt.tight_layout()
plt.savefig(os.path.join('images', 'torus-{}-density.png'.format(args.density)))
