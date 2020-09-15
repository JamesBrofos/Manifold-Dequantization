import jax.numpy as jnp


def rodrigues(axis, angle):
    """Implements the Rodrigues formula to convert an axis-angle representation
    into a rotation formula. The angle is represented as a point on the circle.

    Args:
        axis: A vector on the sphere representing a fixed point of the rotation;
            that is, the axis of the rotation.
        angle: A vector on the circle representing the rotation around the axis
            of rotation.

    Returns:
        R: The rotation matrix.

    """
    theta = jnp.arctan2(angle[1], angle[0])
    Id = jnp.eye(3)
    K = jnp.array([[      0., -axis[2],  axis[1]],
                   [ axis[2],       0., -axis[0]],
                   [-axis[1],  axis[0],       0.]])
    Ksq = K@K
    R = Id + jnp.sin(theta)*K + (1. - jnp.cos(theta))*Ksq
    return R


if __name__ == '__main__':
    from jax import random
    import prax.distributions as pd

    rng = random.PRNGKey(1)
    rng, rng_axis, rng_angle = random.split(rng, 3)
    axis = pd.sphere.haarsph(rng_axis, [3])
    angle = pd.sphere.haarsph(rng_angle, [3])
    R = rodrigues(axis, angle)

    assert jnp.allclose(R.T@R, jnp.eye(3), atol=1e-7)
    assert jnp.allclose(jnp.linalg.det(R), 1., atol=1e-7)
