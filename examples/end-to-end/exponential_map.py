import argparse
import os
from functools import partial
from typing import Callable, Sequence, Tuple

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import jax.numpy as jnp
import jax.scipy.stats as jspst
from jax import lax, random, tree_util
from jax import jacobian, jit, value_and_grad, vmap
from jax.experimental import optimizers, stax

from prax.bijectors import realnvp, permute
from prax.distributions import lognormal, sphere


parser = argparse.ArgumentParser(description='Exponential map learning of spherical distribution')
parser.add_argument('--good', type=int, default=1, help='Flag for good or bad tangent space')
args = parser.parse_args()

def sphdist(x: jnp.ndarray, y: jnp.ndarray) -> float:
    """Great circle distance on a sphere. Implementation derived from [1].

    [1] https://www.manopt.org/

    """
    chordal_distance = jnp.linalg.norm(x - y)
    d = jnp.real(2 * jnp.arcsin(0.5 * chordal_distance))
    return d

proj = lambda x, d: d - (x.dot(d)) * x

def logmap(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    """Logarithmic map to compute the coordinates of a point `y` in terms of the
    velocity in the tangent space of `x` which would be required for a geodesic
    to proceed from `x` to `y`. Implementation derived from [1].

    [1] https://www.manopt.org/

    """
    v = proj(x, y - x)
    nv = jnp.linalg.norm(v)
    di = sphdist(x, y)
    return lax.cond(di > 1e-6, lambda _: v * (di / nv), lambda _: v, None)

def expmap(x: jnp.ndarray, v: jnp.ndarray) -> jnp.ndarray:
    """Exponential map to proceed from `x` with velocity `v` along the geodesic.
    Implementation derived from [1].

    [1] https://www.manopt.org/

    """
    nv = jnp.linalg.norm(v)
    return lax.cond(nv > 0., lambda _: x * jnp.cos(nv) + v * (jnp.sin(nv)/ nv), lambda _: x, None)

# Set random number generation seeds.
rng = random.PRNGKey(0)
rng, rng_bij, rng_y, rng_deq = random.split(rng, 4)
rng, rng_train = random.split(rng, 2)

# Generate random draws from the power spherical distribution.
kappa = 50.
muspha = jnp.array([0., 0., 1.])
musphb = jnp.array([-1., 0., 0.])
obs = jnp.vstack((sphere.powsph(random.fold_in(rng_y, 0), kappa, muspha, [10000]),
                  sphere.powsph(random.fold_in(rng_y, 1), kappa, musphb, [10000])))

# Decide to use a good or bad tangent space for the exponential coordinates.
if args.good:
    base = (muspha + musphb) / 2.
else:
    base = -musphb
base /= jnp.linalg.norm(base)
v = logmap(base, obs[0])
assert jnp.allclose(expmap(base, v), obs[0])

basis = jnp.linalg.qr(jnp.hstack((base[..., jnp.newaxis], random.normal(rng, [3, 2]))))[0]
logmap = vmap(logmap, in_axes=(None, 0))
v = logmap(base, obs)
obs = v.dot(basis)[:, 1:]

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
    y = realnvp.forward(x, 1, params[0], fns[0])
    y = permute.forward(y, jnp.array([1, 0]))
    y = realnvp.forward(y, 1, params[1], fns[1])
    y = permute.forward(y, jnp.array([1, 0]))
    y = realnvp.forward(y, 1, params[2], fns[2])
    y = permute.forward(y, jnp.array([1, 0]))
    y = realnvp.forward(y, 1, params[3], fns[3])
    y = permute.forward(y, jnp.array([1, 0]))
    y = realnvp.forward(y, 1, params[4], fns[4])
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
    fldj = 0.
    y = realnvp.inverse(y, 1, params[4], fns[4])
    fldj += realnvp.forward_log_det_jacobian(y, 1, params[4], fns[4])
    y = permute.inverse(y, jnp.array([1, 0]))
    fldj += permute.forward_log_det_jacobian()
    y = realnvp.inverse(y, 1, params[3], fns[3])
    fldj += realnvp.forward_log_det_jacobian(y, 1, params[3], fns[3])
    y = permute.inverse(y, jnp.array([1, 0]))
    fldj += permute.forward_log_det_jacobian()
    y = realnvp.inverse(y, 1, params[2], fns[2])
    fldj += realnvp.forward_log_det_jacobian(y, 1, params[2], fns[2])
    y = permute.inverse(y, jnp.array([1, 0]))
    fldj += permute.forward_log_det_jacobian()
    y = realnvp.inverse(y, 1, params[1], fns[1])
    fldj += realnvp.forward_log_det_jacobian(y, 1, params[1], fns[1])
    y = permute.inverse(y, jnp.array([1, 0]))
    fldj += permute.forward_log_det_jacobian()
    y = realnvp.inverse(y, 1, params[0], fns[0])
    fldj += realnvp.forward_log_det_jacobian(y, 1, params[0], fns[0])
    return jspst.multivariate_normal.logpdf(y, jnp.zeros((2, )), 1.) - fldj

def mixture_density(obs: jnp.ndarray, kappa: float, muspha: jnp.ndarray,
                    musphb: jnp.ndarray) -> jnp.ndarray:
    """Compute the density of the power spherical mixture distribution."""
    densa = jnp.exp(sphere.powsphlogdensity(obs, kappa, muspha))
    densb = jnp.exp(sphere.powsphlogdensity(obs, kappa, musphb))
    return 0.5 * (densa + densb)


# Generate the parameters of RealNVP bijectors.
bij_params, bij_fns = [], []
for i in range(5):
    p, f = network_factory(random.fold_in(rng_bij, i), 1, 1)
    bij_params.append(p)
    bij_fns.append(f)

@partial(jit, static_argnums=(0, 4))
def train(num_steps, lr, rng, bij_params, bij_fns, obs):
    opt_init, opt_update, get_params = optimizers.adam(lr)
    lossfn = lambda params: -ambient_flow_log_prob(params, bij_fns, obs).mean() + 1e-4 * jnp.sum([jnp.square(b).sum() for b in tree_util.tree_flatten(bij_params)[0]])
    def step(opt_state, it):
        rng_step = random.fold_in(rng, it)
        params = get_params(opt_state)
        nll, nll_grad = value_and_grad(lossfn)(params)
        opt_state = opt_update(it, nll_grad, opt_state)
        return opt_state, nll
    opt_state, nll = lax.scan(step, opt_init(bij_params), jnp.arange(num_steps))
    bij_params = get_params(opt_state)
    return bij_params, nll

bij_params, nll = train(10000, 1e-3, rng, bij_params, bij_fns, obs)
xs = random.normal(rng, obs.shape)
xs = forward(bij_params, bij_fns, xs)
inj = jnp.linalg.norm(xs, axis=-1) < jnp.pi
prob_inj = jnp.mean(inj)
xs = xs[inj]

E = basis[:, 1:]
m = xs@basis.T[1:]
J = vmap(lambda m: jacobian(expmap, 1)(base, m))(m)
JE = J@E
det = jnp.linalg.det(jnp.swapaxes(JE, -1, -2)@(JE))

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].plot(jnp.arange(len(nll)), nll, '-')
axes[0].set_title('Negative Log-Likelihood')
axes[0].grid(linestyle=':')
axes[0].set_xlabel('Gradient Descent Iterations')
axes[1].plot(obs[:, 0], obs[:, 1], '.', alpha=0.05, label='Actual Samples')
axes[1].plot(xs[:, 0], xs[:, 1], '.', alpha=0.05, label='RealNVP Samples')
axes[1].grid(linestyle=':')
axes[1].set_xlim((-5, 5))
axes[1].set_ylim((-5, 5))
axes[1].set_title('Distribution in Exponential Coordinates')
log_p_est = ambient_flow_log_prob(bij_params, bij_fns, xs)
p_est = jnp.exp(log_p_est) / prob_inj
cb = axes[2].scatter(xs[:, 0], xs[:, 1], s=1.0, c=p_est)
axes[2].set_xlim((-5, 5))
axes[2].set_ylim((-5, 5))
axes[2].grid(linestyle=':')
plt.colorbar(cb, ax=axes[2])
plt.tight_layout()
plt.savefig(os.path.join('images', 'exponential-coordinates-{}.png'.format('good' if args.good else 'bad')))

expm = vmap(expmap, in_axes=(None, 0))(base, m)
sph_p_est = p_est / jnp.sqrt(det)

log_p_target = jnp.log(mixture_density(expm, kappa, muspha, musphb))
log_sph_p_est = jnp.log(sph_p_est)
notnan = ~jnp.isnan(log_sph_p_est)
kl = jnp.mean(log_sph_p_est[notnan] - log_p_target[notnan])
print('exponential map kl-divergence: {:.5f}'.format(kl))

# Visualize density.
p_target = jnp.exp(log_p_target)
fig = plt.figure(figsize=(10, 4))
ax1 = fig.add_subplot(121, projection='3d')
ax1.set_title('Analytic Density')
cb = ax1.scatter(expm[:, 0], expm[:, 1], expm[:, 2], c=p_target, vmin=0., vmax=2.2)
ax1.grid(linestyle=':')
lim = 1.1
ax1.set_xlim((-lim, lim))
ax1.set_ylim((-lim, lim))
ax1.set_zlim((-lim, lim))
ax2 = fig.add_subplot(122, projection='3d')
ax2.set_title('Approximate Density')
cb = ax2.scatter(expm[:, 0], expm[:, 1], expm[:, 2], c=sph_p_est, vmin=0., vmax=2.2)
ax2.grid(linestyle=':')
ax2.set_xlim((-lim, lim))
ax2.set_ylim((-lim, lim))
ax2.set_zlim((-lim, lim))
plt.tight_layout()
for i, angle in enumerate(range(0, 360)):
    ax1.view_init(30, angle)
    ax2.view_init(30, angle)
    plt.savefig(os.path.join('video-frames', 'exponential-density-{}-{:05d}.png'.format('good' if args.good else 'bad', i)))

