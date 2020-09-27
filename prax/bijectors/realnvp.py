from typing import Callable, Tuple

import jax.numpy as jnp

from . import affine


def forward(x: jnp.ndarray, num_masked: int, shift_and_scale_params:
            Tuple[jnp.ndarray], shift_and_scale_fn: Callable, *args) -> jnp.ndarray:
    """Forward transformation of the RealNVP bijector. The input is separated into
    a component that is held constant and a component that is transformed. The
    input that is constant parameterizes an affine transformation of the
    transformed component.

    Args:
        x: An array to transform according to the RealNVP bijector.
        num_masked: The elements of the input up to this index are held constant
            and the remaining elements are transformed; this applies to the last
            axis of the input.
        shift_and_scale_params: The parameters of the function that produces the
            shift and scale of the affine transformation given the constant
            component of the input.
        shift_and_scale_fn: Function that computes the shift and scale of the
            affine transformation given parameters and the constant component of
            the input.
        *args: Arguments for `shift_and_scale_fn`.

    Returns:
        y: The first part of the output consists of the constant component of the
            input. The second part is the affine transformation of the
            transformed component of the input.

    """
    xconst, xtrans = x[..., :num_masked], x[..., num_masked:]
    shift, scale = shift_and_scale_fn(shift_and_scale_params, xconst, *args)
    ytrans = affine.forward(xtrans, shift, scale)
    y = jnp.concatenate([xconst, ytrans], axis=-1)
    return y

def inverse(y: jnp.ndarray, num_masked: int, shift_and_scale_params:
            Tuple[jnp.ndarray], shift_and_scale_fn: Callable, *args) -> jnp.ndarray:
    """Inverse transformation of the RealNVP bijector.

    Args:
        y: An array on which to apply the inverse transformation of the RealNVP
            bijector.
        num_masked: The elements of the input up to this index are held constant
            and the remaining elements are transformed; this applies to the last
            axis of the input.
        shift_and_scale_params: The parameters of the function that produces the
            shift and scale of the affine transformation given the constant
            component of the input.
        shift_and_scale_fn: Function that computes the shift and scale of the
            affine transformation given parameters and the constant component of
            the input.
        *args: Arguments for `shift_and_scale_fn`.

    Returns:
        x: The inverse transformation of the RealNVP bijector applied to the
            input.

    """
    yconst, ytrans = y[..., :num_masked], y[..., num_masked:]
    shift, scale = shift_and_scale_fn(shift_and_scale_params, yconst, *args)
    xtrans = affine.inverse(ytrans, shift, scale)
    x = jnp.concatenate([yconst, xtrans], axis=-1)
    return x

def forward_log_det_jacobian(x: jnp.ndarray, num_masked: int,
                             shift_and_scale_params: Tuple[jnp.ndarray],
                             shift_and_scale_fn: Callable, *args) -> jnp.ndarray:
    """Computes the Jacobian correction for a random variable that is transformed
    in the forward direction by the RealNVP bijector. This is just the forward
    log-determinant of the Jacobian of an affine transformation parameterized
    by the shift and scale function.

    Args:
        x: An array to transform according to the RealNVP bijector.
        num_masked: The elements of the input up to this index are held constant
            and the remaining elements are transformed; this applies to the last
            axis of the input.
        shift_and_scale_params: The parameters of the function that produces the
            shift and scale of the affine transformation given the constant
            component of the input.
        shift_and_scale_fn: Function that computes the shift and scale of the
            affine transformation given parameters and the constant component of
            the input.
        *args: Arguments for `shift_and_scale_fn`.

    Returns:
        out: The log-determinant of the Jacobian of the RealNVP bijector in the
            forward direction.

    """
    xconst, xtrans = x[..., :num_masked], x[..., num_masked:]
    shift, scale = shift_and_scale_fn(shift_and_scale_params, xconst, *args)
    return affine.forward_log_det_jacobian(scale)
