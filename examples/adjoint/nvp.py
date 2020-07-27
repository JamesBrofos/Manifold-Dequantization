"""This code reproduced from the tutorial [1].

[1] https://blog.evjang.com/2019/07/nf-jax.html
"""
import jax.numpy as np
import jax.scipy.stats as spst
from jax import lax, nn, random
from jax.experimental import stax
from jax.experimental.stax import Dense, Relu
from jax.nn.initializers import glorot_normal, normal


def shift_and_log_scale_fn(params, x):
    """Shift function and log-scale function."""
    (W0, b0), (W1, b1), (W2, b2) = params
    h = nn.relu(x@W0 + b0)
    h = nn.relu(h@W1 + b1)
    o = h@W2 + b2
    return np.hsplit(o, 2)

def forward(params, x):
    """RealNVP forward transformation."""
    d = x.shape[-1] // 2
    x1, x2 = x[:, :d], x[:, d:]
    shift, log_scale = shift_and_log_scale_fn(params, x1)
    y2 = x2 * np.exp(log_scale) + shift
    y = np.concatenate([x1, y2], axis=-1)
    return y

def inverse(params, y):
    """RealNVP inverse transformation."""
    d = y.shape[-1] // 2
    y1, y2 = y[:, :d], y[:, d:]
    shift, log_scale = shift_and_log_scale_fn(params, y1)
    x2 = (y2 - shift) * np.exp(-log_scale)
    x = np.concatenate([y1, x2], axis=-1)
    return x, log_scale

def init_nvp_flow(rng, num_dims, num_hidden, dtype):
    """Initialize a single RealNVP transformation."""
    # Only half of the variables are transformed give the shift and scale of
    # the other half of the variables.
    in_shape = (-1, num_dims // 2)
    init, net = stax.serial(
        Dense(num_hidden, glorot_normal(dtype=dtype), normal(dtype=dtype)),
        Relu,
        Dense(num_hidden, glorot_normal(dtype=dtype), normal(dtype=dtype)),
        Relu,
        Dense(2*in_shape[1], glorot_normal(dtype=dtype), normal(dtype=dtype)))
    out_shape, params = init(rng, in_shape)
    params = [params[0], params[2], params[4]]
    return params

def init_nvp_chain(rng, num_chains, num_dims, num_hidden, dtype):
    """Initialize a chain of RealNVP transformations. Each transformation depends
    on the parameters of the underlying the neural network.

    """
    params = []
    for i in range(num_chains):
        p = init_nvp_flow(random.fold_in(rng, i), num_dims, num_hidden, dtype)
        params.append(p)
    return params

def ambient_nvp_flow_log_density(params, base_log_prob_fn, y):
    """Log-density computed from the change-of-variables formula for the RealNVP
    transformation.

    """
    x, log_scale = inverse(params, y)
    ildj = -np.sum(log_scale, axis=-1)
    return base_log_prob_fn(x) + ildj

def ambient_nvp_chain_log_density(params, y):
    """Log-density computed from the change-of-variables formula for a chain of
    RealNVP transform.

    """
    log_prob_fn = lambda x: spst.norm.logpdf(x, 0., 1.).sum(-1)
    for i, p in enumerate(params):
        perm = random.permutation(random.PRNGKey(i), y.shape[-1])
        iperm = np.argsort(perm)
        log_prob_fn = log_prob_fn_factory(p, log_prob_fn)
        log_prob_fn = permutation_factory(log_prob_fn, iperm)
    return log_prob_fn(y)

def ambient_nvp_chain_density(params, y):
    """Compute the density in the ambient space by exponentiating the log-density
    in the embedding space.

    """
    lp = ambient_nvp_chain_log_density(params, y)
    return np.exp(lp)

def ambient_nvp_chain_sample(rng, params, shape):
    """Sample from the RealNVP chain by applying each transformation in
    succession.

    """
    base_sample_fn = lambda rng, shape: random.normal(rng, shape)
    x = base_sample_fn(rng, shape)
    for i, p in enumerate(params):
        perm = random.permutation(random.PRNGKey(i), x.shape[-1])
        x = forward(p, x)
        x = x[..., perm]
    return x

def log_prob_fn_factory(params, log_prob_fn):
    """Returns a function that computes the log-density of the RealNVP
    transformation given a log-density function for the base distribution.

    """
    return lambda x: ambient_nvp_flow_log_density(params, log_prob_fn, x)

def permutation_factory(log_prob_fn, iperm):
    """Under a permutation of the variables, the log-density is the original
    log-density evaluated at the un-permutated variables. Notice that
    permutation is a volume-preserving operation.

    """
    return lambda y: log_prob_fn(y[..., iperm])
