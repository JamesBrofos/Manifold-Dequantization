
def _lp(lp, bij):
    return lambda y: lp(bij.inverse(y)) + bij.inverse_log_det_jacobian(y)

def log_prob(base_log_prob, bijectors, params):
    lp = lambda x: base_log_prob(x)
    for (bij, param) in zip(bijectors, params):
        lp = _lp(lp, bij)
    return lp
