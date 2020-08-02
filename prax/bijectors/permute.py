import jax.numpy as jnp

from .bijector import Bijector


class Permute(Bijector):
    def __init__(self, permutation):
        self.permutation = permutation
        self.ipermutation = jnp.argsort(permutation)
        super(Permute, self).__init__()

    def forward(self, x, **kwargs):
        return x[..., self.permutation]

    def inverse(self, y, **kwargs):
        return y[..., self.ipermutation]

    def forward_log_det_jacobian(self, x, **kwargs):
        return 0.

    def inverse_log_det_jacobian(self, y, **kwargs):
        return 0.
