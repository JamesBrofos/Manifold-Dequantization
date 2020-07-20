"""This code reproduced from the tutorial [1].

[1] https://blog.evjang.com/2019/07/nf-jax.html
"""
import jax.numpy as np
from jax.experimental import stax
from jax.experimental.stax import Dense, Relu
from jax.nn.initializers import glorot_normal, normal

def nvp_forward(net_params, shift_and_log_scale_fn, x, flip=False):
    d = x.shape[-1]//2
    x1, x2 = x[:, :d], x[:, d:]
    if flip:
        x2, x1 = x1, x2
    shift, log_scale = shift_and_log_scale_fn(net_params, x1)
    y2 = x2*np.exp(log_scale) + shift
    if flip:
        x1, y2 = y2, x1
    y = np.concatenate([x1, y2], axis=-1)
    return y

def nvp_inverse(net_params, shift_and_log_scale_fn, y, flip=False):
    d = y.shape[-1]//2
    y1, y2 = y[:, :d], y[:, d:]
    if flip:
        y1, y2 = y2, y1
    shift, log_scale = shift_and_log_scale_fn(net_params, y1)
    x2 = (y2-shift)*np.exp(-log_scale)
    if flip:
        y1, x2 = x2, y1
    x = np.concatenate([y1, x2], axis=-1)
    return x, log_scale

def sample_nvp(net_params, shift_log_scale_fn, base_sample_fn, N, flip=False):
    x = base_sample_fn(N)
    return nvp_forward(net_params, shift_log_scale_fn, x, flip=flip)

def log_prob_nvp(net_params, shift_log_scale_fn, base_log_prob_fn, y, flip=False):
    x, log_scale = nvp_inverse(net_params, shift_log_scale_fn, y, flip=flip)
    ildj = -np.sum(log_scale, axis=-1)
    return base_log_prob_fn(x) + ildj

def init_nvp(rng):
    D = 2
    net_init, net_apply = stax.serial(
        Dense(512, glorot_normal(dtype=np.float64), normal(dtype=np.float64)),
        Relu,
        Dense(512, glorot_normal(dtype=np.float64), normal(dtype=np.float64)),
        Relu,
        Dense(D, glorot_normal(dtype=np.float64), normal(dtype=np.float64)))
    in_shape = (-1, D//2)
    out_shape, net_params = net_init(rng, in_shape)
    def shift_and_log_scale_fn(net_params, x1):
        s = net_apply(net_params, x1)
        return np.split(s, 2, axis=1)
    return net_params, shift_and_log_scale_fn

def init_nvp_chain(rng, n):
    flip = False
    ps, configs = [], []
    for i in range(n):
        p, f = init_nvp(rng)
        ps.append(p), configs.append((f, flip))
        flip = not flip
    return ps, configs

def sample_nvp_chain(rng, ps, configs, base_sample_fn, N):
    x = base_sample_fn(rng, N)
    for p, config in zip(ps, configs):
        shift_log_scale_fn, flip = config
        x = nvp_forward(p, shift_log_scale_fn, x, flip=flip)
    return x

def make_log_prob_fn(p, log_prob_fn, config):
    shift_log_scale_fn, flip = config
    return lambda x: log_prob_nvp(p, shift_log_scale_fn, log_prob_fn, x, flip=flip)

def log_prob_nvp_chain(ps, configs, base_log_prob_fn, y):
    log_prob_fn = base_log_prob_fn
    for p, config in zip(ps, configs):
        log_prob_fn = make_log_prob_fn(p, log_prob_fn, config)
    return log_prob_fn(y)

def prob_nvp_chain(ps, configs, base_log_prob_fn, y):
    return np.exp(log_prob_nvp_chain(ps, configs, base_log_prob_fn, y))
