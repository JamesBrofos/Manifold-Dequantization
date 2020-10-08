import jax.numpy as jnp
from jax import random
from jax import jacobian

from jax.config import config
config.update("jax_enable_x64", True)


"""This code shows how to compute the Jacobian of the QR decomposition in two
different ways. We could first consider the mapping from O(n) x Tri+(n) ->
GL(n) constructed simply as the matrix product A = QR. Alternatively, we could
compute the Jacobian of the map GL(n) -> O(n) x Tri+(n) by computing the unique
QR-decomposition of a matrix A.

The first method is somewhat simpler because GL(n) has as its tangent space at
every point the set of n x n matrices; we know how to identify a basis of this
tangent space. In contrast, O(n) x Tri+(n) has a tangent space which is more
involved, thereby complicating the calculation of the Jacobian determinant.
I'll compute the Jacobian determinant in both ways to show that they are
consistent. (The one virtue that the mapping O(n) x Tri+ -> GL(n) does have is
that the actual expression of the function is much simpler.)

"""

def uqr(A):
    """This is the implementation of the unique QR decomposition as proposed in
    [1], modified for JAX compatibility.

    [1] https://github.com/numpy/numpy/issues/15628

    """
    Q, R = jnp.linalg.qr(A)
    signs = 2 * (jnp.diag(R) >= 0) - 1
    Q = Q * signs[jnp.newaxis, :]
    R = R * signs[:, jnp.newaxis]
    return Q, R

def constraint(Q_and_R):
    """Note that Q and R are both n x n matrices. Therefore, we can express O(n) x
    Tri+(n) as a constrained subset of the (2*n*n)-dimensional real numbers.
    This function implements that constraint. Namely, the subset of variables
    corresponding to the Q matrix must satisfy an orthogonality constraint
    whereas the lower-triangular subset of the R matrix must be equal to
    zero.

    One can readily compute that there are n*n + n*(n-1) / 2 constraint
    functions.

    """
    n = jnp.sqrt(Q_and_R.size / 2).astype(int)
    Q, R = jnp.hsplit(Q_and_R.reshape((n, 2*n)), 2)
    ortho = jnp.ravel(Q.T@Q - jnp.eye(n))
    tril = R[jnp.tril_indices(n, -1)]
    return jnp.hstack((ortho, tril))


# Set the pseudo-random number key and the dimensionality of the matrices.
rng = random.PRNGKey(0)
n = 4
# Generate a random matrix with standard normal entries. Assert that it is
# invertible.
A = random.normal(rng, [n, n])
assert not jnp.allclose(jnp.linalg.det(A), 0.)

# Compute the unique QR decomposition, which exists because we checked that the
# matrix is invertible.
Q, R = uqr(A)

# Convert the QR decomposition into a vectorized representation as a
# (2*n*n)-dimensional vector. From this vector space representation, we can
# compute QR (also as a vector representation) as a (n*n)-dimensional vector.
# Thus, we have a mapping from (2*n*n)-dimensional vectors to (n*n)-dimensional
# vectors.
Q_and_R = jnp.hstack((Q, R)).ravel()
f = lambda Q_and_R: jnp.matmul(*jnp.hsplit(Q_and_R.reshape((n, 2*n)), 2)).ravel()
J = jacobian(f)(Q_and_R)

# Convert the matrix in GL(n) into a vector representation. We can regard the
# unique QR decomposition as a mapping from n*n-dimensional vectors to
# (2*n*n)-dimensional vectors. We can compute the Jacobian of this
# transformation and compute the determinant. Because the tangent space of
# GL(n) is the set n x n matrices, no further work is required in terms of
# finding bases of the tangent space.
a = A.ravel()
g = lambda a: jnp.hstack(uqr(a.reshape((n, n)))).ravel()
H = jacobian(g)(a)
detg = jnp.sqrt(jnp.linalg.det(H.T@H))

# Compute the Jacobian of the constraint function defining the subspace of
# (2*n*n)-dimensional vectors corresponding to the vectorized representation of
# O(n) x Tri+(n). The Jacobian of the constraint function allows us to compute
# an orthogonal basis of the tangent space.
G = jacobian(constraint)(Q_and_R)
u, s, vT = jnp.linalg.svd(G, full_matrices=True)
E = vT[(n**2):].T
# Transform the orthogonal basis of the tangent space by the Jacobian of the
# transformation O(n) x Tri+(n) -> GL(n). Then compute the Jacobian
# determinant.
JE = J@E
detf = jnp.sqrt(jnp.linalg.det(JE.T@JE))

# Assert that the two calculations of the Jacobian determinant are in
# agreement.
assert jnp.allclose(detf, 1. / detg)
