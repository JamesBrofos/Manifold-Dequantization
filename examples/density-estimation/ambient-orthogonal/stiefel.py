import jax.numpy as jnp
from jax import random

from polar import polar

rng = random.PRNGKey(0)
A = random.normal(rng, [10, 3])
o, chol = polar(A)
