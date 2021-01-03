import jax.numpy as jnp


def correlated_torus_density(theta: jnp.ndarray) -> jnp.ndarray:
    """Unnormalized correlated density on the torus.

    Args:
        theta: An array containing the two angular coordinates of the torus.

    Returns:
        out: The unnormalized density at the provided angular coordinates.

    """
    thetaa, thetab = theta[..., 0], theta[..., 1]
    return jnp.squeeze(jnp.exp(jnp.cos(thetaa + thetab - 1.94)))

def unimodal_torus_density(theta: jnp.ndarray) -> jnp.ndarray:
    """Unnormalized multimodal density on the torus.

    Args:
        theta: An array containing the two angular coordinates of the torus.

    Returns:
        out: The unnormalized density at the provided angular coordinates.

    """
    p = lambda thetaa, thetab, phia, phib: jnp.exp(jnp.cos(thetaa - phia) + jnp.cos(thetab - phib))
    thetaa, thetab = theta[..., 0], theta[..., 1]
    return jnp.squeeze(p(thetaa, thetab, 4.18, 5.96)) / 3.0

def multimodal_torus_density(theta: jnp.ndarray) -> jnp.ndarray:
    """Unnormalized multimodal density on the torus.

    Args:
        theta: An array containing the two angular coordinates of the torus.

    Returns:
        out: The unnormalized density at the provided angular coordinates.

    """
    p = lambda thetaa, thetab, phia, phib: jnp.exp(jnp.cos(thetaa - phia) + jnp.cos(thetab - phib))
    thetaa, thetab = theta[..., 0], theta[..., 1]
    uprob = (p(thetaa, thetab, 0.21, 2.85) +
             p(thetaa, thetab, 1.89, 6.18) +
             p(thetaa, thetab, 3.77, 1.56)) / 3.
    return jnp.squeeze(uprob)
