import jax.numpy as jnp
from jax import random
from jax import jacobian, vmap

import prax.distributions as pd
import prax.manifolds as pm


"""What does it mean to have a density on the sphere? In a Euclidean vector
space, if one is provided with samples and a normalized density function, one
can readily verify that the density function agrees with the samples: Look at a
very small region in space and compute whether or not the observed proportion
of observations in that space agrees with the density over that small region.

For manifolds, which may be sets of measure zero in the ambient Euclidean
space, I am not aware of such an easy check. In this module, we consider the
sphere, which is a two dimensional manifold, and we transform every point on
the sphere into a vector in the tangent space of some base vector, a
two-dimensional vector space; these are called exponential map coordinates.
Using the formulas in [1] we can compute the Jacobian correction which
transforms the density on the sphere into a density on the tangent space.
Notice that the density on the tangent space is necessarily containined in the
radius of injectivity of exponential map.

Because the tangent space is two-dimensional, we can compute empirical
proportions of observations in small regions and verify that these are in
agreement with the transformed density on the tangent space. This allows us to
construct a density on the tangent space of the manifold which agrees with the
usual understanding of densities.

[1] https://arxiv.org/pdf/2002.02428.pdf

"""

# Set pseudo-random number generator keys.
rng = random.PRNGKey(1)
rng, rng_base, rng_obs = random.split(rng, 3)

# Base space of the exponential map tangent space. Also compute a basis of the
# tangent space.
base = pd.sphere.haarsph(rng_base, [3])
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
