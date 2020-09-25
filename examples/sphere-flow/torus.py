import argparse
import os
from functools import partial
from typing import Callable, Sequence, Tuple

import matplotlib.pyplot as plt

import jax.numpy as jnp
from jax import lax, random
from jax import jit, value_and_grad, vmap
from jax.experimental import optimizers, stax

from coordinates import ang2euclid
from mobius import mobius_flow, mobius_log_prob

parser = argparse.ArgumentParser(description='Mobius Flows on the Torus')
parser.add_argument('--density', type=str, default='unimodal', help='Indicator for which density to estimate')
args = parser.parse_args()


def correlated_torus_density(theta: jnp.ndarray) -> jnp.ndarray:
    """Unnormalized correlated density on the torus.

    Args:
        theta: An array containing the two angular coordinates of the torus.

    Returns:
        out: The unnormalized density at the provided angular coordinates.

    """
    thetaa, thetab = theta[..., 0], theta[..., 1]
    return jnp.squeeze(jnp.exp(jnp.cos(thetaa + thetab - 1.94)))

def unimodal_torus_density(theta: jnp.ndarray) -> jnp.ndarray:
    """Unnormalized multimodal density on the torus.

    Args:
        theta: An array containing the two angular coordinates of the torus.

    Returns:
        out: The unnormalized density at the provided angular coordinates.

    """
    p = lambda thetaa, thetab, phia, phib: jnp.exp(jnp.cos(thetaa - phia) + jnp.cos(thetab - phib))
    thetaa, thetab = theta[..., 0], theta[..., 1]
    return jnp.squeeze(p(thetaa, thetab, 4.18, 5.96))

def multimodal_torus_density(theta: jnp.ndarray) -> jnp.ndarray:
    """Unnormalized multimodal density on the torus.

    Args:
        theta: An array containing the two angular coordinates of the torus.

    Returns:
        out: The unnormalized density at the provided angular coordinates.

    """
    p = lambda thetaa, thetab, phia, phib: jnp.exp(jnp.cos(thetaa - phia) + jnp.cos(thetab - phib))
    thetaa, thetab = theta[..., 0], theta[..., 1]
    uprob = (p(thetaa, thetab, 0.21, 2.85) +
             p(thetaa, thetab, 1.89, 6.18) +
             p(thetaa, thetab, 3.77, 1.56)) / 3.
    return jnp.squeeze(uprob)

torus_density = {
    'unimodal': unimodal_torus_density,
    'correlated': correlated_torus_density,
    'multimodal': multimodal_torus_density
}[args.density]

def network_factory(rng: jnp.ndarray, num_in: int, num_out: int) -> Tuple:
    """Factory for producing neural networks and their parameterizations.

    Args:
        rng: Pseudo-random number generator seed.
        num_in: Number of inputs to the network.
        num_out: Number of output variables.

    Returns:
        out: A tuple containing the network parameters and a callable function
            that returns the neural network output for the given input.

    """
    params_init, fn = stax.serial(
        stax.Dense(512), stax.Relu,
        stax.Dense(512), stax.Relu,
        stax.Dense(num_out))
    _, params = params_init(rng, (-1, num_in))
    return params, fn

compress = lambda w: 0.99 / (1. + jnp.linalg.norm(w, axis=-1, keepdims=True)) * w

def conditional(theta: jnp.ndarray, params: Sequence[jnp.ndarray], fn: Callable) -> jnp.ndarray:
    """Compute the parameters of the conditional distribution of the second angular
    coordinate given the first.

    """
    x = ang2euclid(theta)
    w = fn(params, ang2euclid(theta)).reshape((-1, 15, 2))
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

def sample_torus(rng, wa, params, fn, num_samples):
    rng, rng_unifa, rng_unifb = random.split(rng, 3)
    unifa = 2.0*jnp.pi*random.uniform(rng_unifa, [num_samples])
    unifb = 2.0*jnp.pi*random.uniform(rng_unifb, [num_samples])
    thetaa = mobius_flow(unifa, wa).mean(0)
    wb = conditional(thetaa, params, fn)
    thetab = vmap(mobius_flow, in_axes=(0, 0))(unifb, wb).mean(1)
    return (thetaa, thetab), (unifa, unifb), wb

def loss(rng, wa, params, fn, num_samples):
    (thetaa, thetab), (unifa, unifb), wb = sample_torus(rng, wa, params, fn, num_samples)
    theta = jnp.stack([thetaa, thetab], axis=-1)
    mlp = torus_log_prob(wa, wb, unifa, unifb, thetaa, thetab)
    target = torus_density(theta)
    log_target = jnp.log(target)
    return jnp.mean(mlp - log_target)

@partial(jit, static_argnums=(3, 4, 5))
def train(rng, wa, params, fn, num_samples, num_steps, lr):
    opt_init, opt_update, get_params = optimizers.adam(lr)
    def step(opt_state, it):
        step_rng = random.fold_in(rng, it)
        wa, params = get_params(opt_state)
        loss_val, loss_grad = value_and_grad(loss, (1, 2))(step_rng, wa, params, fn, num_samples)
        opt_state = opt_update(it, loss_grad, opt_state)
        return opt_state, loss_val
    opt_state, trace = lax.scan(step, opt_init((wa, params)), jnp.arange(num_steps))
    wa, params = get_params(opt_state)
    return (wa, params), trace


rng = random.PRNGKey(0)
rng, rng_torus = random.split(rng, 2)
rng, rng_wa, rng_net = random.split(rng, 3)
rng, rng_train = random.split(rng, 2)

params, fn = network_factory(rng_net, 2, 15*2)
wa = random.normal(rng_wa, [15, 2])
wa = compress(wa)

num_samples = 100
num_steps = 5000
lr = 1e-3
(wa, params), trace = train(rng, wa, params, fn, num_samples, num_steps, lr)

(thetaa, thetab), (unifa, unifb), wb = sample_torus(rng_torus, wa, params, fn, 100000)
theta = jnp.stack([thetaa, thetab], axis=-1)
lpa = mobius_log_prob(unifa, wa)
log_approx = torus_log_prob(wa, wb, unifa, unifb, thetaa, thetab)
approx = jnp.exp(log_approx)
target = torus_density(theta)
log_target = jnp.log(target)
Z = jnp.mean(target / approx)
kl = jnp.mean(log_approx - log_target) + jnp.log(Z)

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
plt.suptitle('KL$(q\Vert p)$ = {:.5f}'.format(kl))
plt.tight_layout()
plt.savefig(os.path.join('images', 'torus-{}-density.png'.format(args.density)))
