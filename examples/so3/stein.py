import argparse
import os
from functools import partial

import jax
import jax.numpy as np
from jax import lax, random
from jax.config import config
config.update("jax_enable_x64", True)

import matplotlib.pyplot as plt

import kernel as kut
import rodrigues as sonut


parser = argparse.ArgumentParser(description='SO(3) Learning with Stein Variational Gradient Descent')
parser.add_argument('--num-samples', type=int, default=10, help='Number of samples o generate')
parser.add_argument('--num-obs', type=int, default=100, help='Number of observations to generate')
parser.add_argument('--noise-scale', type=float, default=1.0, help='Standard deviation of noise corruption')
parser.add_argument('--learning-rate', type=float, default=1e-2, help='Gradient descent learning rate')
parser.add_argument('--num-steps', type=int, default=1000, help='Number of steps of Stein variational gradient descent')
parser.add_argument('--seed', type=int, default=0, help='Pseudo-random number generator seed')
parser.add_argument('--euclid', type=int, default=1, help='Flag to enable Euclidean or Lie kernel')
args = parser.parse_args()


def median_dist(v):
    rsq = np.square(v[:, np.newaxis] - v).sum(-1)
    s = np.sqrt(np.sort(rsq.ravel()))
    return s[(s.size-1) // 2]

def generate_rotation_data(rng, num_samples, num_dims, noise_scale):
    x_rng, rot_rng, noise_rng = random.split(rng, 3)
    x = random.normal(x_rng, [num_samples, num_dims])
    v = random.normal(rot_rng, [num_dims])
    v *= np.sqrt(2.)*np.pi / np.linalg.norm(v)
    R = sonut.rodrigues(sonut.euclid2skew(v))
    y = (R.dot(x.T)).T
    y += noise_scale * random.normal(noise_rng, y.shape)
    return x, y, R, v

rng = random.PRNGKey(args.seed)
theta_rng, data_rng = random.split(rng, 2)
theta = 0.1*random.normal(theta_rng, [args.num_samples, 3])
x, y, R, v = generate_rotation_data(data_rng, args.num_obs, 3, args.noise_scale)

def potential(theta):
    R = sonut.rodrigues(sonut.euclid2skew(theta))
    e = R.dot(x.T).T - y
    return -0.5*np.sum(e**2, axis=-1).sum() - 0.5*np.sum(theta**2)


@partial(jax.jit, static_argnums=(2, 3))
def svgd(theta, lr, num_steps, euclid):
    alpha = 0.9
    def step(theta_and_hist, it):
        theta, hist = theta_and_hist
        rng = random.PRNGKey(it)
        if euclid:
            K, dK = kut.euclid_kernel_and_grad(rng, theta, median_dist(theta))
        else:
            K, dK = kut.lie_kernel_and_grad(rng, theta, median_dist(theta))
        G = jax.vmap(lambda t: jax.grad(potential)(t))(theta)
        phi = K.dot(G) + dK.sum(0)
        hist = lax.cond(it == 0,
                        lambda _: np.square(phi),
                        lambda _: alpha*hist + (1.-alpha)*np.square(phi),
                        None)
        phi = phi / (1e-6 + np.sqrt(hist))
        theta = jax.vmap(sonut.retract)(theta + lr * phi)
        pot = jax.vmap(lambda t: potential(t))(theta)
        norm = np.linalg.norm(G)
        return [theta, hist], (pot, norm)

    hist = np.zeros_like(theta)
    theta_and_hist, trace = lax.scan(step, [theta, hist], np.arange(num_steps))
    return theta_and_hist[0], trace

theta, trace = svgd(theta, args.learning_rate, args.num_steps, args.euclid)

f, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(trace[0].mean(1), '-')
axes[0].grid(linestyle=':')
axes[0].set_title('Average Potential')
axes[0].set_xlabel('Iteration Number')
axes[1].plot(trace[1], '-')
axes[1].grid(linestyle=':')
axes[1].set_title('Gradient Size')
axes[1].set_xlabel('Iteration Number')
plt.tight_layout()
plt.savefig(os.path.join('images', 'trace.png'))
