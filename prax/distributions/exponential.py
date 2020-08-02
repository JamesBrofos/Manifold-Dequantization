import jax.numpy as jnp
from jax import random
from .distribution import Distribution

class Exponential(Distribution):
    def rvs(self, key, shape, scale):
        return scale * random.exponential(key, shape)

    def log_prob(self, x, scale):
        log_scale = jnp.log(scale)
        return log_scale - x / self.scale
