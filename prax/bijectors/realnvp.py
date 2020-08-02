import jax.numpy as jnp
from .bijector import Bijector
from .linear import LinearDiagonal


class RealNVP(Bijector):
    def __init__(self, num_masked, shift_and_log_scale_fn):
        self.num_masked = num_masked

        def bijector_fn(paramdict, x0):
            params = paramdict['params']
            shift, log_scale = shift_and_log_scale_fn(params, x0)
            scale = jnp.exp(log_scale)
            return LinearDiagonal(shift, scale)
        self.bijector_fn = bijector_fn

        super(RealNVP, self).__init__()

    def forward(self, x, **kwargs):
        x0, x1 = x[..., :self.num_masked], x[..., self.num_masked:]
        y1 = self.bijector_fn(kwargs, x0).forward(x1)
        y = jnp.concatenate([x0, y1], axis=-1)
        return y

    def inverse(self, y, **kwargs):
        y0, y1 = y[..., :self.num_masked], y[..., self.num_masked:]
        x1 = self.bijector_fn(kwargs, y0).inverse(y1)
        x = jnp.concatenate([y0, x1], axis=-1)
        return x

    def forward_log_det_jacobian(self, x, **kwargs):
        x0, x1 = x[..., :self.num_masked], x[..., self.num_masked:]
        return self.bijector_fn(kwargs, x0).forward_log_det_jacobian(x1)

    def inverse_log_det_jacobian(self, y, **kwargs):
        y0, y1 = y[..., :self.num_masked], y[..., self.num_masked:]
        return self.bijector_fn(kwargs, y0).inverse_log_det_jacobian(y1)
