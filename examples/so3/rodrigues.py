import jax
import jax.numpy as np
from jax import lax, random



def euclid2skew(v):
    return np.array([[   0.,  v[0], v[1]],
                     [-v[0],    0., v[2]],
                     [-v[1], -v[2],   0.]]) / np.sqrt(2.)

def skew2euclid(X):
    return np.sqrt(2.)*np.array([X[0, 1], X[0, 2], X[1, 2]])

def expm(X):
    L, V = np.linalg.eig(X)
    Y = V.dot(np.diag(np.exp(L)).dot(V.conj().T)).real
    return Y

def logm(R, S):
    L, V = np.linalg.eig(R.T.dot(S))
    Y = V.dot(np.diag(np.log(L)).dot(V.conj().T)).real
    return Y

def rodrigues(K):
    Ksq = K.dot(K)
    Id = np.eye(K.shape[0])
    norm = np.linalg.norm(np.sqrt(0.5)*K)
    half_norm = 0.5*norm
    exp = Id + np.sin(norm) / norm * K + 0.5*(np.sin(half_norm) / half_norm)**2*Ksq
    return lax.cond(norm > 1e-10,
                    lambda _: exp,
                    lambda _: Id,
                    None)

def irodrigues(R, S):
    """This function is based on, but is not equivalent to, the discussion at [1].

    [1] https://math.stackexchange.com/questions/3031999/proof-of-logarithm-map-formulae-from-so3-to-mathfrak-so3
    """
    U = R.T.dot(S)
    tn = np.arccos(0.5*(np.trace(U) - 1.))
    xi = tn * np.array([U[0, 1] - U[1, 0],
                        U[0, 2] - U[2, 0],
                        U[1, 2] - U[2, 1]]) / (np.sqrt(2.)*np.sin(tn))
    return lax.cond(np.linalg.norm(R-S) < 1e-10,
                    lambda _: np.zeros_like(U),
                    lambda _: euclid2skew(xi),
                    None)

def retract(v):
    Id = np.eye(v.size)
    return skew2euclid(irodrigues(Id, rodrigues(euclid2skew(v))))

def rotdist(R, S):
    return np.linalg.norm(irodrigues(R, S))

def skewdist(X, Y):
    d = rotdist(rodrigues(X), rodrigues(Y))
    return d

def liedist(x, y):
    return skewdist(euclid2skew(x), euclid2skew(y))

def randrot(rng, num_dims=3):
    def cond(val):
        return np.abs(np.linalg.det(val[0]) - 1.) > 1e-10

    def body(val):
        _, it = val
        Q, _ = np.linalg.qr(random.normal(random.fold_in(rng, it), [num_dims, num_dims]))
        return [Q, it + 1]

    Q = np.zeros((num_dims, num_dims))
    return lax.while_loop(cond, body, [Q, 0])[0]


if __name__ == '__main__':
    from jax.config import config
    config.update("jax_enable_x64", True)

    @jax.jit
    def check():
        def step(_, it):
            rng = random.PRNGKey(it)
            X = random.normal(random.fold_in(rng, 0), [3, 3])
            X = (X - X.T) / 2.
            Y = random.normal(random.fold_in(rng, 1), [3, 3])
            Y = (Y - Y.T) / 2.
            R = rodrigues(X)
            S = rodrigues(Y)
            Xp = irodrigues(R, S)
            err = np.linalg.norm(Xp - logm(R, S))
            return _, (err, X, Y)
        return lax.scan(step, None, np.arange(100000))[1]

    res = check()
    print('maximum difference: {:.10f}'.format(res[0].max()))

    R = randrot(random.PRNGKey(0))
    X = irodrigues(np.eye(3), R)
    x = skew2euclid(X)
    S = randrot(random.PRNGKey(1))
    Y = irodrigues(np.eye(3), S)
    y = skew2euclid(Y)
    print('rotation distance: {:.5f} - skew distance: {:.5f} - lie distance: {:.5f}'.format(
        rotdist(R, S), skewdist(X, Y), liedist(x, y)))

    @jax.jit
    def descend(x, y, num_steps):
        lr = 1e-3
        def step(it, x):
            g = jax.grad(lambda x, y: liedist(x, y)**2)(x, y)
            g = lax.cond(np.any(np.isnan(g)), lambda _: np.zeros_like(g), lambda _: g, None)
            return x - lr * g

        return lax.fori_loop(0, num_steps, step, x)

    xp = descend(x, y, 1000000)
    delta = liedist(xp, y)
    print('descent distance: {:.5f}'.format(delta))
