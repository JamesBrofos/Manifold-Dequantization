import jax.numpy as jnp
import jax.scipy.stats as jspst
from jax import random
from jax.experimental import stax
from jax.experimental.stax import Dense, FanOut, Relu


from prax.bijectors import LinearDiagonal, Permute, RealNVP


def mvn_log_prob(x, mu, Cov):
    return jspst.multivariate_normal.logpdf(x, mu, Cov)

def permute_test():
    rng = random.PRNGKey(0)
    x = random.normal(rng, [10, 5, 3])
    perm = random.permutation(rng, jnp.arange(x.shape[-1]))
    bij = Permute(perm)
    y = bij.forward(x)
    xr = bij.inverse(y)
    assert jnp.allclose(x, xr)

def linear_diagonal_test():
    rng = random.PRNGKey(0)
    x = random.normal(rng, [10, 5, 3])
    scale = random.normal(random.fold_in(rng, 0 ), [x.shape[-1]])
    shift = random.normal(random.fold_in(rng, 1), [x.shape[-1]])
    bij = LinearDiagonal(shift, scale)
    y = bij.forward(x)
    xr = bij.inverse(y)
    assert jnp.allclose(x, xr)

    x = random.normal(rng, [3])
    y = bij.forward(x)
    Id = jnp.eye(3)
    Cov = jnp.diag(jnp.square(scale))
    zero = jnp.zeros((3, ))
    log_prob = mvn_log_prob(x, zero, Id) - bij.forward_log_det_jacobian(x)
    log_prob_ = mvn_log_prob(y, shift, Cov)
    assert jnp.allclose(log_prob, log_prob_)

def real_nvp_test():
    num_dims = 5
    num_masked = num_dims // 2
    num_out = num_dims - num_masked
    rng = random.PRNGKey(0)
    net_init, net = stax.serial(
        Dense(512), Relu,
        Dense(512), Relu,
        FanOut(2),
        stax.parallel(Dense(num_out), Dense(num_out)))
    _, params = net_init(rng, (None, num_masked))

    x = random.normal(rng, [10, num_dims])
    bij = RealNVP(num_masked, net)
    y = bij.forward(x, params=params)
    xr = bij.inverse(y, params=params)
    assert jnp.allclose(x, xr)

    zero = jnp.zeros((num_dims, ))
    Id = jnp.eye(num_dims)

    fldj = bij.forward_log_det_jacobian(x, **{'params': params})
    ildj = bij.inverse_log_det_jacobian(y, params=params)
    assert jnp.allclose(fldj, -ildj)


if __name__ == '__main__':
    real_nvp_test()
    permute_test()
    linear_diagonal_test()
