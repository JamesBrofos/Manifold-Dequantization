import jax.numpy as jnp
from jax import random
from jax import jacobian, vmap

import prax.distributions as pd
import prax.manifolds as pm


# Set pseudo-random number generator keys.
rng = random.PRNGKey(1)
rng, rng_base, rng_obs = random.split(rng, 3)

# Base space of the exponential map tangent space. Also compute a basis of the
# tangent space.
# base = pd.sphere.haarsph(rng_base, [3])
base = jnp.array([0., -1., 0.])
B = random.normal(rng, [3, 2])
Bp = jnp.concatenate((base[..., jnp.newaxis], B), axis=-1)
O = jnp.linalg.qr(Bp)[0][:, 1:]

# Sample from the power spherical distribution and compute the density at
# sampled points.
musph = jnp.array([0., 0., 1.])
kappa = 50.0
obs = pd.sphere.powsph(rng_obs, kappa, musph, [1000000])
logprob = pd.sphere.powsphlogdensity(obs, kappa, musph)
prob = jnp.exp(logprob)

# Compute exponential coordinates.
f = lambda x: pm.sphere.logmap(base, x)@O
ec = vmap(f)(obs)

# Compute determinant correction for samples in the exponential coordinates
# representation.
J = vmap(jacobian(f))(obs)
B = random.normal(rng, [len(obs), 3, 2])
Bp = jnp.concatenate((obs[..., jnp.newaxis], B), axis=-1)
E = jnp.linalg.qr(Bp)[0][..., 1:]
JE = J@E
det = jnp.sqrt(jnp.linalg.det(jnp.swapaxes(JE, -1, -2)@(JE)))
assert jnp.abs(jnp.einsum('ij,ijk->ik', obs, E)).max() < 1e-6

# Compute the density in the tangent space and compare to the empirical
# probability of lying in a small region of the tangent space.
eprob = prob / det
delta = 0.1
for i in range(50):
    p = ec[i]
    pr = eprob[i] * delta**2
    idx0 = jnp.abs(ec[:, 0] - p[0]) < delta / 2.
    idx1 = jnp.abs(ec[:, 1] - p[1]) < delta / 2.
    pr_est = jnp.mean(idx0 & idx1)
    print('prob.: {:.10f} - estim. prob.: {:.10f}'.format(pr, pr_est))
