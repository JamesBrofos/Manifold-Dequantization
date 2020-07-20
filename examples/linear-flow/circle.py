import argparse
import os
from functools import partial

import matplotlib.pyplot as plt

import jax.numpy as np
import jax.scipy.stats as spst
from jax import grad, jit, lax, random, vmap, value_and_grad
from jax.experimental.optimizers import adam


parser = argparse.ArgumentParser(description='Projecting ambient distribution to circle')
parser.add_argument('--step-size', type=float, default=1e-2, help='Linear flow learning rate')
parser.add_argument('--num-steps', type=int, default=1000, help='Number of learning steps')
parser.add_argument('--num-samples', type=int, default=10000, help='Number of samples to use for computing KL divergence')
parser.add_argument('--seed', type=int, default=0, help='Pseudo-random number generator seed')
args = parser.parse_args()

def project(x):
    norm = np.sqrt(np.square(x).sum(-1))
    return x / norm[..., np.newaxis]

def density(x, mu, sigmasq):
    Sigma = np.diag(sigmasq)
    return spst.multivariate_normal.pdf(x, mu, Sigma)

def circle_density(y, mu, sigmasq, radii):
    delta = radii[1] - radii[0]
    halfdelta = delta / 2.
    mid = radii - halfdelta
    def _cd(radius):
        return radius * density(radius * y, mu, sigmasq)
    return delta * vmap(_cd)(mid).sum(0)

def sample(rng, mu, sigmasq, num_samples):
    sigma = np.sqrt(sigmasq)
    x = sigma * random.normal(rng, [num_samples, 2]) + mu
    y = project(x)
    return x, y

def kl_divergence(rng, mu, log_sigmasq):
    sigmasq = np.exp(log_sigmasq)
    x, y = sample(rng, mu, sigmasq, args.num_samples)
    m = lax.stop_gradient(np.maximum(x.max() + 1., 10.))
    radii = np.linspace(0., m, 20)[1:]
    logp = np.log(circle_density(y, mup, sigmasqp, radii))
    logq = np.log(circle_density(y, mu, sigmasq, radii))
    return np.mean(logq - logp)

@partial(jit)
def flow(mu, log_sigmasq):
    init_fun, update_fun, get_params = adam(args.step_size)
    opt_state = init_fun((mu, log_sigmasq))
    def step(opt_state, it):
        mu, log_sigmasq = get_params(opt_state)
        sigmasq = np.exp(log_sigmasq)
        rng_ = random.fold_in(rng, it)
        kl, grads = value_and_grad(kl_divergence, (1, 2))(rng_, mu, log_sigmasq)
        opt_state = update_fun(it, grads, opt_state)
        return opt_state, kl
    opt_state, kl = lax.scan(step, opt_state, np.arange(args.num_steps))
    return get_params(opt_state), kl

rng = random.PRNGKey(args.seed)
mu = np.array([0., 0.])
sigmasq = np.array([1., 1.])

mup = np.array([1.5, 0.5])
sigmasqp = np.array([2., 0.2])
(mu, log_sigmasq), kl = flow(mu, np.log(sigmasq))
sigmasq = np.exp(log_sigmasq)
kl.block_until_ready()

theta = np.linspace(-np.pi, np.pi, 1000)
y = np.vstack((np.cos(theta), np.sin(theta))).T
radii = np.linspace(0., 10., 100)[1:]
pdens = circle_density(y, mup, sigmasqp, radii)
qdens = circle_density(y, mu, sigmasq, radii)

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].plot(np.arange(kl.size) + 1, kl, '-')
axes[0].grid(linestyle=':')
axes[0].set_xlabel('Gradient Descent Iteration')
axes[0].set_ylabel('KL-Divergence')
axes[0].set_title('KL-Divergence by Iteration')
axes[1].plot(theta, pdens, '-', label='Target Density')
axes[1].plot(theta, qdens, '--', label='Approx. Density')
axes[1].legend()
axes[1].grid(linestyle=':')
axes[1].set_xlabel('Angle')
axes[1].set_ylabel('Probability Density')
axes[1].set_title('Target vs. Approximation')
plt.tight_layout()
plt.savefig(os.path.join('images', 'linear-flow.png'))
