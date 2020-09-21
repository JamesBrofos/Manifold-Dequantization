import jax.numpy as jnp
import jax.scipy.special as jspsp
import jax.scipy.stats as jspst
from jax import random
from jax import vmap


def logpdf(x: jnp.ndarray, mu: jnp.ndarray, sigma: jnp.ndarray, weights: jnp.ndarray) -> jnp.ndarray:
    lpmvn = vmap(lambda _: vmap(
        lambda m, s: jspst.multivariate_normal.logpdf(_, m, s))(mu, sigma))(x)
    return jspsp.logsumexp(lpmvn, axis=1, b=weights[..., jnp.newaxis])

def sample(rng: jnp.ndarray, mu: jnp.ndarray, sigma: jnp.ndarray, weights:
           jnp.ndarray, num_samples: int) -> jnp.ndarray:
    rng, rng_comp, rng_normal = random.split(rng, 3)
    num_comp = len(mu)
    comp = random.choice(rng_comp, num_comp, [num_samples], p=weights)
    x = vmap(lambda m, s: random.multivariate_normal(rng_normal, m, s, [num_samples]))(mu, sigma)
    x = x[comp, jnp.arange(num_samples)]
    return x


# if __name__ == '__main__':
#     rng = random.PRNGKey(0)
#     mu = jnp.array([[1., 0.], [0., 1.], [-2., -2.]])
#     sigma = jnp.array([jnp.eye(2), 2. * jnp.eye(2), 0.1 * jnp.eye(2)])
#     weights = jnp.array([0.1, 0.3, 0.6])
#     num_samples = 1000
#     x = sample(rng, mu, sigma, weights, num_samples)
#     lp = logpdf(x, mu, sigma, weights)
