import jax.numpy as jnp
from .bijector import Bijector


class LinearDiagonal(Bijector):
    def __init__(self, shift, scale):
        self.shift = shift
        self.scale = scale
        self.abs_scale = jnp.abs(scale)
        super(LinearDiagonal, self).__init__()

    def forward(self, x, **kwargs):
        return x * self.scale + self.shift

    def inverse(self, y, **kwargs):
        return (y - self.shift) / self.scale

    def forward_log_det_jacobian(self, x, **kwargs):
        return jnp.sum(jnp.log(self.abs_scale), axis=-1)

    def inverse_log_det_jacobian(self, y, **kwargs):
        return -jnp.sum(jnp.log(self.abs_scale), axis=-1)
