import jax.numpy as jnp


def ang2euclid(theta: jnp.ndarray) -> jnp.ndarray:
    """Convert an angle into a point on the unit circle in the plane. The
    conversion is made via polar coordinates with the radius fixed at one.

    Args:
        theta: The angle parameterizing a point on the circle.

    Returns:
        out: The point on the circle determined by the polar coordinate angle.

    """
    return jnp.stack([jnp.cos(theta), jnp.sin(theta)], axis=-1)

def euclid2ang(x: jnp.ndarray) -> jnp.ndarray:
    """Converts points on the circle embedded in the plane into an angular
    representation in polar coordinates. Note that the output is shifted so
    that angles lie in the interval [0, 2pi).

    Args:
        x: Observations on the circle.

    Returns:
        out: Angular representation of the input.

    """
    return jnp.arctan2(x[..., 1], x[..., 0]) + jnp.pi
