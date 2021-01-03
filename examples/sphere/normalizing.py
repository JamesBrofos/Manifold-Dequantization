import argparse
import os
from functools import partial
from typing import Callable, Sequence, Tuple

import matplotlib.pyplot as plt

import jax.numpy as jnp
from jax import lax, nn, random, tree_util
from jax import grad, jacobian, jit, value_and_grad, vmap
from jax.experimental import optimizers, stax

import prax.manifolds as pm

from distributions import embedded_sphere_density
from rejection_sampling import rejection_sampling


parser = argparse.ArgumentParser(description='Mobius spline flow on hypersphere')
parser.add_argument('--num-steps', type=int, default=20000, help='Number of gradient descent iterations for score matching training')
parser.add_argument('--lr', type=float, default=1e-3, help='Gradient descent learning rate')
parser.add_argument('--num-samples', type=int, default=100, help='Number of samples per batch')
parser.add_argument('--num-spline', type=int, default=32, help='Number of intervals in spline flow')
parser.add_argument('--num-mobius', type=int, default=12, help='Number of Mobius transforms in convex combination')
parser.add_argument('--num-hidden', type=int, default=64, help='Number of hidden units used in the neural networks')
parser.add_argument('--seed', type=int, default=0, help='Pseudo-random number generator seed')
args = parser.parse_args()


def mobius_circle(x: jnp.ndarray, w: jnp.ndarray) -> jnp.ndarray:
    """Implements the Mobius transformation on a circle embedded in the plane. The
    implementation is vectorized so that if `x` and `w` are both matrices then
    the output is the individual Mobius transformation of `x` by each center
    individually.

    Args:
        x: Observations on the circle.
        w: Center points of the Mobius transformation.

    Returns:
        xp: The observations on the circle transformed under the Mobius
            transformation.

    """
    w = jnp.atleast_2d(w)
    x = jnp.atleast_2d(x)
    w_sqnorm = jnp.square(jnp.linalg.norm(w, axis=-1))
    xmw = jnp.moveaxis(x[:, jnp.newaxis] - w, 0, 1)
    xmw_sqnorm = jnp.square(jnp.linalg.norm(xmw, axis=-1))
    ratio = ((1 - w_sqnorm)[..., jnp.newaxis] / xmw_sqnorm)[..., jnp.newaxis]
    xp = ratio * xmw - w[:, jnp.newaxis]
    return xp

def mobius_angle(theta: jnp.ndarray, w: jnp.ndarray) -> jnp.ndarray:
    """The Mobius transformation on the circle as viewed as an angle in the
    interval [0, 2pi).

    Args:
        theta: The angle parameterizing a point on the circle.
        w: Center points of the Mobius transformation.

    Returns:
        out: The Mobius transformation as a transformation on angles.

    """
    x = pm.sphere.ang2euclid(theta)
    xp = mobius_circle(x, w)
    return jnp.squeeze(pm.sphere.euclid2ang(xp))

def mobius_flow(theta: jnp.ndarray, w: jnp.ndarray) -> jnp.ndarray:
    """Following [1] the Mobius flow must have the property that the endpoints of
    the interval [0, 2pi] are mapped to themselves. This transformation
    enforces this property by appling a backwards rotation to bring the zero
    angle back to itself.

    Args:
        theta: The angle parameterizing a point on the circle.
        w: Center points of the Mobius transformation.

    Returns:
        out: The Mobius flow as a transformation on angles leaving the endpoints
            of the interval invariant.

    """
    theta = jnp.hstack((0., theta))
    omega = mobius_angle(theta, w)
    omega -= omega[..., [0]]
    return jnp.squeeze(jnp.mod(omega[..., 1:], 2.*jnp.pi))

def mobius_log_prob(theta: jnp.ndarray, w: jnp.ndarray) -> jnp.ndarray:
    """Compute the log-density of the Mobius flow assuming the base distribution is
    uniform on the interval [0, 2pi).

    Args:
        theta: The angle parameterizing a point on the circle.
        w: Center points of the Mobius transformation.

    Returns:
        out: The log-density of the transformation of a uniformly-distributed
            angle under a Mobius transformation.

    """
    theta = jnp.atleast_1d(theta)
    log_base = -jnp.log(2.*jnp.pi)
    jac = vmap(jacobian(lambda t: mobius_flow(t, w)))(theta).mean(-1)
    fldj = jnp.log(jac)
    return jnp.squeeze(log_base - fldj)

def rational_quadratic_parameters(x: jnp.ndarray, xk: jnp.ndarray, yk: jnp.ndarray, delta: jnp.ndarray) -> jnp.ndarray:
    """Compute necessary intermediate parameters used for computing the spline
    flow. For details on the rational quadratic transformation, consult [1].

    [1] https://arxiv.org/pdf/1906.04032.pdf

    """
    idxn = jnp.searchsorted(xk, x)
    idx = idxn - 1
    xi = (x - xk[idx]) / (xk[idxn] - xk[idx])
    ym = yk[idxn] - yk[idx]
    sk = ym / (xk[idxn] - xk[idx])
    dk = delta[idx]
    dkp = delta[idxn]
    dp = dkp + dk
    xib = xi * (1.0 - xi)
    xisq = jnp.square(xi)
    return (idx, xi, xib, xisq, dk, dkp, dp, sk, ym)

def rational_quadratic(x: jnp.ndarray, xk: jnp.ndarray, yk: jnp.ndarray, delta: jnp.ndarray) -> jnp.ndarray:
    """Compute the rational quadratic diffeomorphism of the interval [-1, +1]. The
    rational quadratic is defined as a series of segments, each of which is
    equipped with a rational quadratic function. The derivative of the rational
    quadratic function is constructed so as to be strictly positive.

    Args:
        x: Points to transform by the rational quadratic function.
        xk: The x-coordinates of the intervals on which the rational quadratics
            are defined.
        yk: The y-coordinates of the destination intervals of the rational
            quadratic transforms.
        delta: Derivatives at internal points.

    Returns:
        rq: The rational quadratic transformation of the [-1, +1] interval
            evaluated at the inputs `x` given the knots defined by `xk` and `yk`
            and derivatives `delta`.

    """
    idx, xi, xib, xisq, dk, dkp, dp, sk, ym = rational_quadratic_parameters(x, xk, yk, delta)
    rq = yk[idx] + ym * (sk * xisq + dk * xib) / (sk + (dp - 2.0 * sk) * xib)
    return rq

def grad_rational_quadratic(x: jnp.ndarray, xk: jnp.ndarray, yk: jnp.ndarray, delta: jnp.ndarray) -> jnp.ndarray:
    """The gradient of the rational quadratic transformation of the interval [-1, +1].

    Args:
        x: Points to transform by the rational quadratic function.
        xk: The x-coordinates of the intervals on which the rational quadratics
            are defined.
        yk: The y-coordinates of the destination intervals of the rational
            quadratic transforms.
        delta: Derivatives at internal points.

    Returns:
        drq: The derivative of the rational quadratic function.

    """
    idx, xi, xib, xisq, dk, dkp, dp, sk, ym = rational_quadratic_parameters(x, xk, yk, delta)
    drq = jnp.square(sk) * (dkp * xisq + 2.0 * sk * xib + dk * jnp.square(1.0 - xi)) / jnp.square(sk + (dp - 2.0 * sk) * xib)
    return drq

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
    params_init, net = stax.serial(
        stax.Dense(num_hidden), stax.Relu,
        stax.Dense(num_hidden), stax.Relu,
        stax.Dense(num_out))
    _, params = params_init(rng, (-1, num_in))
    return params, net

def torus2sphere(ra: jnp.ndarray, ang: jnp.ndarray) -> jnp.ndarray:
    """Convert points represented on a two-dimensional torus into points on a
    two-dimensional sphere.

    Args:
        ra: First radial coordinate.
        ang: Angular coordinate.

    Returns:
        out: Conversion of the inputs on a torus into a sphere.

    """
    circ = pm.sphere.ang2euclid(ang)
    sph = jnp.sqrt(1. - jnp.square(ra[..., jnp.newaxis])) * circ
    sph = jnp.concatenate((sph, ra[..., jnp.newaxis]), axis=-1)
    return sph

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
    xk = jnp.atleast_2d(jnp.cumsum(2*nn.softmax(thetax), axis=-1) - 1.)
    xk = jnp.hstack((-jnp.ones((xk.shape[0], 1)), xk))
    yk = jnp.atleast_2d(jnp.cumsum(2*nn.softmax(thetay), axis=-1) - 1.)
    yk = jnp.hstack((-jnp.ones((yk.shape[0], 1)), yk))
    delta = nn.softplus(thetad)
    return jnp.squeeze(xk), jnp.squeeze(yk), jnp.squeeze(delta)

compress = lambda w: 0.99 / (1. + jnp.linalg.norm(w, axis=-1, keepdims=True)) * w

def mobius_conditional(ra: jnp.ndarray, params: Sequence[jnp.ndarray], net: Callable) -> jnp.ndarray:
    """Compute the parameters of the Mobius transformation of the angular parameter
    given the radial parameters.

    Args:
        ra: The radial parameter.
        paramsm: Parameters of the neural network giving the conditional
            distribution of the angular parameter.
        netm: Neural network to compute the angular parameter.

    Returns:
        w: Parameterization of the Mobius transformation given the two radial
            parameters.

    """
    w = net(params, ra).reshape((-1, args.num_mobius, 2))
    w = compress(w)
    return w

def mobius_spline_sample(rng: jnp.ndarray, num_samples: int, xk: jnp.ndarray, yk: jnp.ndarray, delta: jnp.ndarray, paramsm: Sequence[jnp.ndarray], netm: Callable):
    """Sample from the Mobius spline transforms on the sphere.

    Args:
        rng: Pseudo-random number generator seed.
        num_samples: Number of samples to generate.
        xk: The x-coordinates of the intervals on which the rational quadratics
            are defined.
        yk: The y-coordinates of the destination intervals of the rational
            quadratic transforms.
        delta: Derivatives at internal points.
        paramsm: Parameters of the neural network giving the conditional
            distribution of the angular parameter.
        netm: Neural network to compute the angular parameter.

    Returns:
        ra: Sampled radial parameter.
        ang: Sampled angular parameter.
        raunif: Uniform radial parameters that are transformed by the spline
            flow.
        angunif: Uniform angular parameter that are transformed by the Mobius
            transformation.
        w: Mobius transformation parameters.

    """
    rng, rng_ra, rng_ang = random.split(rng, 3)
    raunif = random.uniform(rng_ra, [num_samples], minval=-1., maxval=1.)
    angunif = random.uniform(rng_ang, [num_samples], minval=0., maxval=2.*jnp.pi)
    ra = rational_quadratic(raunif, xk, yk, delta)
    w = mobius_conditional(ra.reshape((-1, 1)), paramsm, netm)
    ang = vmap(mobius_flow)(angunif, w).mean(1)
    return (ra, ang), (raunif, angunif), w

def mobius_spline_log_prob(ra: jnp.ndarray, raunif: jnp.ndarray, ang: jnp.ndarray, angunif: jnp.ndarray, w: jnp.ndarray, xk: jnp.ndarray, yk: jnp.ndarray, delta: jnp.ndarray):
    """Compute the log-density of the Mobius spline transformation on the sphere.

    """
    lpx = -jnp.log(2.) - jnp.log(grad_rational_quadratic(raunif, xk, yk, delta))
    lpang = vmap(mobius_log_prob)(angunif, w)
    log_prob = lpx + lpang
    return log_prob

def loss(rng: jnp.ndarray,
         thetax: jnp.ndarray,
         thetay: jnp.ndarray,
         thetad: jnp.ndarray,
         paramsm: Sequence[jnp.ndarray],
         netm: Callable,
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
    xk, yk, delta = spline_unconstrained_transform(thetax, thetay, thetad)
    (ra, ang), (raunif, angunif), w = mobius_spline_sample(rng, num_samples, xk, yk, delta, paramsm, netm)
    xsph = torus2sphere(ra, ang)
    mslp = mobius_spline_log_prob(ra, raunif, ang, angunif, w, xk, yk, delta)
    t = embedded_sphere_density(xsph)
    lt = jnp.log(t)
    return jnp.mean(mslp - lt)

@partial(jit, static_argnums=(5, 6, 7))
def train(rng: jnp.ndarray,
          thetax: jnp.ndarray,
          thetay: jnp.ndarray,
          thetad: jnp.ndarray,
          paramsm: Sequence[jnp.ndarray],
          netm: Callable,
          num_samples: int,
          num_steps: int,
          lr: float):
    opt_init, opt_update, get_params = optimizers.adam(lr)
    def step(opt_state, it):
        step_rng = random.fold_in(rng, it)
        thetax, thetay, thetad, paramsm = get_params(opt_state)
        loss_val, loss_grad = value_and_grad(loss, (1, 2, 3, 4))(step_rng, thetax, thetay, thetad, paramsm, netm, num_samples)
        opt_state = opt_update(it, loss_grad, opt_state)
        return opt_state, loss_val
    params = (thetax, thetay, thetad, paramsm)
    opt_state, trace = lax.scan(step, opt_init(params), jnp.arange(num_steps))
    thetax, thetay, thetad, paramsm = get_params(opt_state)
    return (thetax, thetay, thetad, paramsm), trace


# Set random number generation seeds.
rng = random.PRNGKey(args.seed)
rng, rng_netm = random.split(rng, 2)
rng, rng_thetax, rng_thetay, rng_thetad = random.split(rng, 4)
rng, rng_ms, rng_train = random.split(rng, 3)
rng, rng_xobs = random.split(rng, 2)

paramsm, netm = network_factory(rng_netm, 1, args.num_mobius*2, args.num_hidden)
thetax = random.uniform(rng_thetax, [args.num_spline])
thetay = random.uniform(rng_thetay, [args.num_spline])
thetad = random.uniform(rng_thetad, [args.num_spline - 1])

# Compute number of parameters.
count = lambda x: jnp.prod(jnp.array(x.shape))
num_paramsm = jnp.array(tree_util.tree_map(count, tree_util.tree_flatten(paramsm)[0])).sum()
num_theta = count(thetax) + count(thetay) + count(thetad)
num_params = num_theta + num_paramsm
print('number of parameters: {}'.format(num_params))

# Train normalizing flow on the sphere.
(thetax, thetay, thetad, paramsm), trace = train(rng_train, thetax, thetay, thetad, paramsm, netm, args.num_samples, args.num_steps, args.lr)
num_samples = 100000
xk, yk, delta = spline_unconstrained_transform(thetax, thetay, thetad)

# Compute comparison statistics.
(ra, ang), (raunif, angunif), w = mobius_spline_sample(rng_ms, num_samples, xk, yk, delta, paramsm, netm)
xsph = torus2sphere(ra, ang)
log_approx = mobius_spline_log_prob(ra, raunif, ang, angunif, w, xk, yk, delta)
approx = jnp.exp(log_approx)
target = embedded_sphere_density(xsph)
log_target = jnp.log(target)
w = target / approx
Z = jnp.mean(w)
kl = jnp.mean(log_approx - log_target) + jnp.log(Z)
ess = jnp.square(jnp.sum(w)) / jnp.sum(jnp.square(w))
ress = 100 * ess / len(w)
xobs = rejection_sampling(rng_xobs, len(xsph), 3, embedded_sphere_density)
mean_mse = jnp.square(jnp.linalg.norm(xsph.mean(0) - xobs.mean(0)))
cov_mse = jnp.square(jnp.linalg.norm(jnp.cov(xsph.T) - jnp.cov(xobs.T)))
print('normalizing - Mean MSE: {:.5f} - Covariance MSE: {:.5f} - KL$(q\Vert p)$ = {:.5f} - Rel. ESS: {:.2f}%'.format(mean_mse, cov_mse, kl, ress))
