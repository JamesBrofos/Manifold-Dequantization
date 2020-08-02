import argparse
import os
from typing import Tuple, Callable

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import jax.numpy as jnp
import jax.scipy.stats as jspst
from jax import lax, random
from jax import jit, value_and_grad
from jax.experimental import optimizers, stax
from jax.experimental.stax import Dense, FanOut, Relu

from prax.random import powsph
from prax import bijectors


parser = argparse.ArgumentParser(description='Dequantizing the power spherical density')
parser.add_argument('--num-obs', type=int, default=10000, help='Number of observations of power spherical distribution')
parser.add_argument('--kappa', type=float, default=20., help='Power spherical density concentration parameter')
parser.add_argument('--step-size', type=float, default=1e-3, help='Learning rate')
parser.add_argument('--num-steps', type=int, default=1000, help='Number of gradient descent iterations')
parser.add_argument('--seed', type=int, default=0, help='Pseudo-random number generator seed')
args = parser.parse_args()

def realnvp_factory(rng: jnp.ndarray, num_masked: int) -> Tuple[bijectors.RealNVP, Tuple]:
    """Manufacture a RealNVP bijector.

    Args:
        rng: A PRNGKey used as the random key.
        num_masked: The number of inputs to retain as-is and use for
            parameterizing the shift and scale for transforming the remaining
            variables.

    Returns:
        A tuple containing a RealNVP bijection operator and the parameters of
        that bijection.

    """
    num_out = 3 - num_masked
    net_init, net = stax.serial(
        Dense(512), Relu,
        Dense(512), Relu,
        FanOut(2),
        stax.parallel(Dense(num_out), Dense(num_out)))
    _, params = net_init(rng, (None, num_masked))
    return bijectors.RealNVP(num_masked, net), params

def stdnorm_logpdf(x: jnp.ndarray) -> jnp.ndarray:
    """Compute the log-p.d.f. of the standard normal distribution at the input
    points.

    Args:
        x: An array of points at which to compute the log-p.d.f. of the standard
            normal distibution.

    Returns:
        Values of the log-p.d.f. at the input points.

    """
    dim = x.shape[-1]
    return -0.5*jnp.sum(jnp.square(x), axis=-1) - 0.5*dim*jnp.log(2.*jnp.pi)

def dequantize(rng: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
    """Dequantizes points on a sphere by expanding their radius according to an
    exponential random variable.

    Args:
        rng: A PRNGKey used as the random key.
        x: An array of points lying on the unit sphere to dequantize.

    Returns:
        Dequantized observations that lie in the same direction as on the
        sphere but with radii expanded according to an exponential
        distribution.

    """
    rad = 1. + random.exponential(rng, x.shape[:-1])
    return rad[..., jnp.newaxis] * x


def flow_logpdf_factory(rng: jnp.ndarray, num_masked: int) -> Tuple[Callable, Tuple]:
    """Factory function for parameterizing a RealNVP flow.

    Args:
        rng: A PRNGKey used as the random key.
        num_masked: The number of inputs to retain as-is and use for
            parameterizing the shift and scale for transforming the remaining
            variables.

    Returns:
        A tuple containing a callable function that returns the log-p.d.f. of
        the normalizing flow as well as the parameterization of that normalizing
        flow.

    """
    rng, nvp1_rng, nvp2_rng = random.split(rng, 3)
    nvp1, params1 = realnvp_factory(nvp1_rng, num_masked)
    perm1 = bijectors.Permute(jnp.array([2, 0, 1]))
    nvp2, params2 = realnvp_factory(nvp2_rng, num_masked)
    params = (params1, params2)

    def forward(params, x):
        """Compute the forward trajectory."""
        params1, params2 = params
        x_nvp1 = nvp1.forward(x, **{'params': params1})
        x_perm1 = perm1.forward(x_nvp1)
        x_nvp2 = nvp2.forward(x_perm1, **{'params': params2})
        return x_nvp2

    def reverse(params, y):
        """Compute the reverse trajectory."""
        params1, params2 = params
        x_perm1 = nvp2.inverse(y, **{'params': params2})
        x_nvp1 = perm1.inverse(x_perm1)
        x = nvp1.inverse(x_nvp1, **{'params': params1})
        return x, x_perm1, x_nvp1

    def flow_logpdf(params, y):
        # Unpack parameters of the RealNVP flows.
        params1, params2 = params
        x, x_perm1, x_nvp1 = reverse(params, y)
        # Compute the log-p.d.f. of the transformed density.
        base = stdnorm_logpdf(x)
        nvp1_fldj = nvp1.forward_log_det_jacobian(x, **{'params': params1})
        perm1_fldj = perm1.forward_log_det_jacobian(x_nvp1)
        nvp2_fldj = nvp2.forward_log_det_jacobian(x_perm1, **{'params': params2})
        log_pdf = base - nvp1_fldj - perm1_fldj - nvp2_fldj
        return log_pdf

    return flow_logpdf, forward, params

@jit
def train_flow(rng: jnp.ndarray, x: jnp.ndarray) -> Tuple[Tuple, jnp.ndarray]:
    rng, flow_rng, deq_rng = random.split(rng, 3)
    flow_logpdf, flow_forward, params = flow_logpdf_factory(flow_rng, 1)
    negloglik = lambda params, y: -flow_logpdf(params, y).mean()
    opt_init, opt_update, get_params = optimizers.adam(args.step_size)
    def run(opt_state, it):
        y = dequantize(random.fold_in(deq_rng, it), x)
        v, g = value_and_grad(negloglik)(get_params(opt_state), y)
        opt_state = opt_update(it, g, opt_state)
        return opt_state, v
    opt_state, nll = lax.scan(run, opt_init(params), jnp.arange(args.num_steps))
    params = get_params(opt_state)
    return flow_forward, params, nll


rng = random.PRNGKey(args.seed)
rng, powsph_rng, train_rng = random.split(rng, 3)
mu = jnp.ones((3, ))
mu /= jnp.linalg.norm(mu)
x = powsph(rng, args.kappa, mu, [args.num_obs])
forward, params, nll = train_flow(train_rng, x)

rng, z_rng, deq_rng = random.split(rng, 3)
z = random.normal(z_rng, [10000, 2])
yp = forward(params, z)
y = dequantize(deq_rng, x)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(y[:, 0], y[:, 1], y[:, 2], '.')
ax.plot(yp[:, 0], yp[:, 1], yp[:, 2], '.')
ax.grid()
plt.savefig(os.path.join('images', 'real-nvp-flow.png'))
