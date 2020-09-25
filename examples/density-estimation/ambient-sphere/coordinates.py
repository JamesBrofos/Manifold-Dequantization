from typing import Tuple

import jax.numpy as jnp


def sph2latlon(xsph: jnp.ndarray) -> Tuple[jnp.ndarray]:
    """Converts a point on the unit sphere into latitude and longitude measured in
    radians.

    Args:
        xsph: Observations on the sphere.

    Returns:
        lat, lon: A tuple containing the latitude and longitude coordinates of the input.

    """
    x, y, z = xsph[..., 0], xsph[..., 1], xsph[..., 2]
    lat = jnp.arctan2(z, jnp.sqrt(jnp.square(x) + jnp.square(y)))
    lon = jnp.arctan2(y, x)
    return lat, lon

def sph2euclid(theta: jnp.ndarray, phi: jnp.ndarray) -> jnp.ndarray:
    """Parameterize a point on the sphere as two angles in spherical coordinates.

    Args:
        theta: First angular coordinate.
        phi: Second angular coordinate.

    Returns:
        out: The point on the sphere parameterized by the two angular
            coordinates.

    """
    return jnp.array([jnp.sin(phi)*jnp.cos(theta),
                      jnp.sin(phi)*jnp.sin(theta),
                      jnp.cos(phi)]).T

def hsph2euclid(theta: jnp.ndarray, phi: jnp.ndarray, gamma: jnp.ndarray) -> jnp.ndarray:
    """Parameterize a point on the four-dimensional sphere as three angles in hyper-spherical coordinates.

    Args:
        theta: First angular coordinate.
        phi: Second angular coordinate.
        gamma: Third angular coordinate.

    Returns:
        out: The point on the sphere parameterized by the two angular
            coordinates.

    """
    return jnp.array([
        jnp.cos(theta),
        jnp.sin(theta)*jnp.cos(phi),
        jnp.sin(theta)*jnp.sin(phi)*jnp.cos(gamma),
        jnp.sin(theta)*jnp.sin(phi)*jnp.sin(gamma)]).T

