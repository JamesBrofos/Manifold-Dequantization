import argparse
import os
from functools import partial
from typing import Callable, Sequence, Tuple

import matplotlib.pyplot as plt

import jax.numpy as jnp
from jax import lax, nn, random
from jax import grad, jit, value_and_grad, vmap
from jax.experimental import optimizers, stax

from coordinates import ang2euclid, hsph2euclid, sph2euclid, sph2latlon
from mobius import mobius_flow, mobius_log_prob
from spline import rational_quadratic, grad_rational_quadratic


parser = argparse.ArgumentParser(description='Mobius spline flow on hypersphere')
parser.add_argument('--num-steps', type=int, default=10000, help='Number of gradient descent iterations for score matching training')
parser.add_argument('--lr', type=float, default=1e-3, help='Gradient descent learning rate')
parser.add_argument('--num-samples', type=int, default=100, help='Number of samples per batch')
parser.add_argument('--num-spline', type=int, default=64, help='Number of intervals in spline flow')
parser.add_argument('--num-mobius', type=int, default=32, help='Number of Mobius transforms in convex combination')
parser.add_argument('--seed', type=int, default=0, help='Pseudo-random number generator seed')
args = parser.parse_args()


def sphere_density(xsph: jnp.ndarray) -> jnp.ndarray:
    """Unnormalized multimodal density on the sphere.

    Args:
        xsph: Observations on the sphere at which to compute the unnormalized
            density.

    Returns:
        out: The unnormalized density at the provided points on the sphere.

    """
    p = lambda x, mu: jnp.exp(10.*x.dot(mu))
    mua = hsph2euclid(1.7, -1.5, 2.3)
    mub = hsph2euclid(-3.0, 1.0, 3.0)
    muc = hsph2euclid(0.6, -2.6, 4.5)
    mud = hsph2euclid(-2.5, 3.0, 5.0)
    return p(xsph, mua) + p(xsph, mub) + p(xsph, muc) + p(xsph, mud)

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
    params_init, net = stax.serial(
        stax.Dense(512), stax.Relu,
        stax.Dense(512), stax.Relu,
        stax.Dense(num_out))
    _, params = params_init(rng, (-1, num_in))
    return params, net

def torus2sphere(ra: jnp.ndarray, rb: jnp.ndarray, ang: jnp.ndarray) -> jnp.ndarray:
    """Convert points represented on a torus into points on a sphere.

    Args:
        ra: First radial coordinate.
        rb: Second radial coordinate.
        ang: Angular coordinate.

    Returns:
        out: Conversion of the inputs on a torus into a sphere.

    """
    circ = ang2euclid(ang)
    sph = jnp.sqrt(1. - jnp.square(rb[..., jnp.newaxis])) * circ
    sph = jnp.concatenate((sph, rb[..., jnp.newaxis]), axis=-1)
    hsph = jnp.sqrt(1. - jnp.square(ra[..., jnp.newaxis])) * sph
    hsph = jnp.concatenate((hsph, ra[..., jnp.newaxis]), axis=-1)
    return hsph

def spline_unconstrained_transform(thetax: jnp.ndarray, thetay: jnp.ndarray, thetad: jnp.ndarray) -> jnp.ndarray:
    """Transform the unconstrained parameters of the spline transform into their
    constrained counterparts.

    Args:
        thetax: Unconstrained x-coordinates of the spline intervals.
        thetay: Unconstrained y-coordinates of the spline intervals.
        thetad: Unconstrained derivatives at internal points.

    Returns:
        xk: The x-coordinates of the intervals on which the rational quadratics
            are defined.
        yk: The y-coordinates of the destination intervals of the rational
            quadratic transforms.
        delta: Derivatives at internal points.

    """
    x = jnp.atleast_2d(jnp.cumsum(2*nn.softmax(thetax), axis=-1) - 1.)
    x = jnp.hstack((-jnp.ones((x.shape[0], 1)), x))
    y = jnp.atleast_2d(jnp.cumsum(2*nn.softmax(thetay), axis=-1) - 1.)
    y = jnp.hstack((-jnp.ones((y.shape[0], 1)), y))
    delta = nn.softplus(thetad)
    return jnp.squeeze(x), jnp.squeeze(y), jnp.squeeze(delta)

def spline_conditional(ra: jnp.ndarray, paramsr: Sequence[jnp.ndarray], netr: Callable, num_spline: int) -> Tuple[jnp.ndarray]:
    """Partition the output of the spline parameterization neural network into
    components corresponding to unconstrained parameterizations of the knots
    and internal derivatives.

    Args:
        ra: The first radial parameter which parameterizes the second.
        paramsr: Parameters of the neural network giving the conditional
            distribution of the second radius parameter.
        netr: Neural network to compute the second radius parameter.
        num_spline: Number of intervals in the spline flow.

    Returns:
        out: A tuple containing constrained parameterizations of the knots and
            internal derivatives of the conditional distribution of the second
            radial parameter.

    """
    o = netr(paramsr, ra[..., jnp.newaxis])
    thetax = o[..., :num_spline]
    thetay = o[..., num_spline:2*num_spline]
    thetad = o[..., 2*num_spline:]
    return spline_unconstrained_transform(thetax, thetay, thetad)

compress = lambda w: 0.99 / (1. + jnp.linalg.norm(w, axis=-1, keepdims=True)) * w

def mobius_conditional(ra: jnp.ndarray, rb: jnp.ndarray, paramsm: Sequence[jnp.ndarray], netm: Callable) -> jnp.ndarray:
    """Compute the parameters of the Mobius transformation of the angular parameter
    given the radial parameters.

    Args:
        ra: The first radial parameter.
        rb: The second radial parameter.
        paramsm: Parameters of the neural network giving the conditional
            distribution of the angular parameter.
        netm: Neural network to compute the angular parameter.

    Returns:
        w: Parameterization of the Mobius transformation given the two radial
            parameters.

    """
    x = jnp.stack((ra, rb), axis=-1)
    w = netm(paramsm, x).reshape((-1, args.num_mobius, 2))
    w = compress(w)
    return w

def mobius_spline_sample(rng: jnp.ndarray, num_samples: int, xk: jnp.ndarray, yk: jnp.ndarray, deltak: jnp.ndarray, paramsm: Sequence[jnp.ndarray], netm: Callable, paramsr: Sequence[jnp.ndarray], netr: Callable):
    """Sample from the Mobius spline transforms on the sphere.

    Args:
        rng: Pseudo-random number generator seed.
        num_samples: Number of samples to generate.
        xk: The x-coordinates of the intervals on which the rational quadratics
            are defined for the first radial parameter.
        yk: The y-coordinates of the destination intervals of the rational
            quadratic transforms for the first radial parameter.
        deltak: Derivatives at internal points for the first radial parameter.
        paramsm: Parameters of the neural network giving the conditional
            distribution of the angular parameter.
        netm: Neural network to compute the angular parameter.
        paramsr: Parameters of the neural network giving the conditional
            distribution of the second radius parameter.
        netr: Neural network to compute the second radius parameter.

    Returns:
        ra: Sampled first radial parameter.
        rb: Sampled second radial parameter.
        ang: Sampled angular parameter.
        raunif: Uniform first radial parameters that are transformed by the
            spline flow.
        rbunif: Uniform second radial parameters that are transformed by the
            spline flow.
        angunif: Uniform angular parameter that are transformed by the Mobius
            transformation.
        w: Mobius transformation parameters.
        xl: The x-coordinates of the intervals on which the rational quadratics
            are defined for the second radial parameter.
        yl: The y-coordinates of the destination intervals of the rational
            quadratic transforms for the second radial parameter.
        deltal: Derivatives at internal points for the second radial parameter.

    """
    rng, rng_ra, rng_rb, rng_ang = random.split(rng, 4)
    raunif = random.uniform(rng_ra, [num_samples], minval=-1., maxval=1.)
    rbunif = random.uniform(rng_rb, [num_samples], minval=-1., maxval=1.)
    angunif = random.uniform(rng_ang, [num_samples], minval=0., maxval=2.*jnp.pi)
    ra = rational_quadratic(raunif, xk, yk, deltak)
    xl, yl, deltal = spline_conditional(ra, paramsr, netr, args.num_spline)
    rb = vmap(rational_quadratic)(rbunif, xl, yl, deltal)
    w = mobius_conditional(ra.reshape((-1, 1)), rb.reshape((-1, 1)), paramsm, netm)
    ang = vmap(mobius_flow)(angunif, w).mean(1)
    return (ra, rb, ang), (raunif, rbunif, angunif), (w, xl, yl, deltal)

def mobius_spline_log_prob(ra: jnp.ndarray, raunif: jnp.ndarray, rb: jnp.ndarray, rbunif: jnp.ndarray, ang: jnp.ndarray, angunif: jnp.ndarray, w: jnp.ndarray, xk: jnp.ndarray, yk: jnp.ndarray, deltak: jnp.ndarray, xl: jnp.ndarray, yl: jnp.ndarray, deltal: jnp.ndarray):
    """Compute the log-density of the Mobius spline transformation on the sphere.

    """
    lpra = -jnp.log(2.) - jnp.log(grad_rational_quadratic(raunif, xk, yk, deltak))
    lprb = -jnp.log(2.) - jnp.log(vmap(grad_rational_quadratic)(rbunif, xl, yl, deltal))
    lpang = vmap(mobius_log_prob)(angunif, w)
    ldj =  -(3./2 - 1.) * jnp.log(1 - jnp.square(ra))
    log_prob = lpra + lprb + lpang + ldj
    return log_prob

def loss(rng: jnp.ndarray,
         thetax: jnp.ndarray,
         thetay: jnp.ndarray,
         thetad: jnp.ndarray,
         paramsm: Sequence[jnp.ndarray],
         netm: Callable,
         paramsr: Sequence[jnp.ndarray],
         netr: Callable,
         num_samples: int):
    """KL(q || p) loss function to minimize. This is computable up to a constant
    when the target density is known up to proportionality.

    Args:
        rng: Pseudo-random number generator seed.
        thetax: Unconstrained x-coordinates of the spline intervals for radial coordinate.
        thetay: Unconstrained y-coordinates of the spline intervals for radial coordinate.
        thetad: Unconstrained derivatives at internal points for radial coordinate.
        paramsm: Parameters of the neural network giving the conditional
            distribution of the angular parameter.
        netm: Neural network to compute the angular parameter.
        num_samples: Number of samples to draw.

    Returns:
        out: A Monte Carlo estimate of KL(q || p).

    """
    xk, yk, deltak = spline_unconstrained_transform(thetax, thetay, thetad)
    (ra, rb, ang), (raunif, rbunif, angunif), (w, xl, yl, deltal) = mobius_spline_sample(rng, num_samples, xk, yk, deltak, paramsm, netm, paramsr, netr)
    xsph = torus2sphere(ra, rb, ang)
    mslp = mobius_spline_log_prob(ra, raunif, rb, rbunif, ang, angunif, w, xk, yk, deltak, xl, yl, deltal)
    t = sphere_density(xsph)
    lt = jnp.log(t)
    return jnp.mean(mslp - lt)

@partial(jit, static_argnums=(5, 7, 8, 9))
def train(rng: jnp.ndarray,
          thetax: jnp.ndarray,
          thetay: jnp.ndarray,
          thetad: jnp.ndarray,
          paramsm: Sequence[jnp.ndarray],
          netm: Callable,
          paramsr: Sequence[jnp.ndarray],
          netr: Callable,
          num_samples: int,
          num_steps: int,
          lr: float):
    opt_init, opt_update, get_params = optimizers.adam(lr)
    def step(opt_state, it):
        step_rng = random.fold_in(rng, it)
        thetax, thetay, thetad, paramsm, paramsr = get_params(opt_state)
        loss_val, loss_grad = value_and_grad(loss, (1, 2, 3, 4, 6))(step_rng, thetax, thetay, thetad, paramsm, netm, paramsr, netr, num_samples)
        opt_state = opt_update(it, loss_grad, opt_state)
        return opt_state, loss_val
    params = (thetax, thetay, thetad, paramsm, paramsr)
    opt_state, trace = lax.scan(step, opt_init(params), jnp.arange(num_steps))
    thetax, thetay, thetad, paramsm, paramsr = get_params(opt_state)
    return (thetax, thetay, thetad, paramsm, paramsr), trace


rng = random.PRNGKey(0)
rng, rng_netm, rng_netr = random.split(rng, 3)
rng, rng_thetax, rng_thetay, rng_thetad = random.split(rng, 4)
rng, rng_ms, rng_train = random.split(rng, 3)

paramsr, netr = network_factory(rng_netr, 1, 3*args.num_spline-1)
paramsm, netm = network_factory(rng_netm, 2, args.num_mobius*2)
thetax = random.uniform(rng_thetax, [args.num_spline])
thetay = random.uniform(rng_thetay, [args.num_spline])
thetad = random.uniform(rng_thetad, [args.num_spline - 1])

(thetax, thetay, thetad, paramsm, paramsr), trace = train(rng_train, thetax, thetay, thetad, paramsm, netm, paramsr, netr, args.num_samples, args.num_steps, args.lr)
num_samples = 100000
xk, yk, deltak = spline_unconstrained_transform(thetax, thetay, thetad)

(ra, rb, ang), (raunif, rbunif, angunif), (w, xl, yl, deltal) = mobius_spline_sample(rng_ms, num_samples, xk, yk, deltak, paramsm, netm, paramsr, netr)
xsph = torus2sphere(ra, rb, ang)
log_approx = mobius_spline_log_prob(ra, raunif, rb, rbunif, ang, angunif, w, xk, yk, deltak, xl, yl, deltal)
approx = jnp.exp(log_approx)
target = sphere_density(xsph)
log_target = jnp.log(target)
w = target / approx
Z = jnp.mean(w)
kl = jnp.mean(log_approx - log_target) + jnp.log(Z)
ess = jnp.square(jnp.sum(w)) / jnp.sum(jnp.square(w))
ress = 100 * ess / len(w)
print('estimated KL(q || p) = {:.5f}'.format(kl))

plt.figure()
plt.title('KL$(q\Vert p)$ = {:.5f} - Rel. ESS: {:.2f}%'.format(kl, ress))
plt.plot(trace)
plt.grid(linestyle=':')
plt.xlabel('Gradient Descent Iteration')
plt.ylabel('Loss')
plt.savefig(os.path.join('images', 'hypersphere-mobius-spline.png'))
