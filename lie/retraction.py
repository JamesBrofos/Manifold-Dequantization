import jax.numpy as np


def polar_retr(x, v):
    """Given an element `x` in O(n) and an element `v` in the Lie algebra, proceed
    in the direction of `v` to compute `x + v` and retract back to O(n) using
    the singular value decomposition.

    """
    m = x + v
    u, s, vh = np.linalg.svd(m)
    return u@vh

if __name__ == '__main__':
    from jax import random
    import rand
    n = 3
    for it in range(10):
        rng = random.PRNGKey(it)
        x = rand.randson(rng, n)
        v = rand.randlie(rng, n)
        r = polar_retr(x, v)
        det = np.linalg.det(r)
        print('determinant: {:.5f}'.format(det))
