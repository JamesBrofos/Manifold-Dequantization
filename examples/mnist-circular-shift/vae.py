import jax
import jax.numpy as np
from jax import random
from jax.experimental import stax
from jax.experimental.stax import Dense, FanOut, Relu, LogSoftmax, Softplus


def gaussian_kl(qm, qv, pm, pv):
    """KL divergence from a diagonal Gaussian to the standard Gaussian."""
    element_wise = 0.5 * (np.log(pv) - np.log(qv) + qv / pv + np.power(qm - pm, 2) / pv - 1)
    kl = element_wise.sum(-1)
    return kl

def gaussian_sample(rng, mu, sigmasq):
    """Sample a diagonal Gaussian."""
    return mu + np.sqrt(sigmasq) * random.normal(rng, mu.shape)

def bernoulli_logpdf(logits, data):
    """Bernoulli log pdf of data given logits."""
    bce = np.maximum(logits, 0) - logits * data + np.log(1 + np.exp(-np.abs(logits)))
    log_prob = -bce.sum(-1)
    return log_prob

class SO2VAE:
    def __init__(self, rng, num_latent):
        self.num_latent = num_latent
        encoder_init, self.encode = stax.serial(
            Dense(512), Relu,
            Dense(512), Relu,
            FanOut(3),
            stax.parallel(
                Dense(2*2),
                Dense(num_latent), stax.serial(Dense(num_latent), Softplus)))
        decoder_init, self.decode = stax.serial(
            Dense(512), Relu,
            Dense(512), Relu,
            Dense(28**2))
        enc_rng, dec_rng = random.split(rng, 2)
        _, self.encoder_params = encoder_init(enc_rng, (-1, 28 * 28))
        _, self.decoder_params = decoder_init(dec_rng, (-1, num_latent))

    def z_given_x(self, rng, mu_z, sigmasq_z, R):
        z = gaussian_sample(rng, mu_z, sigmasq_z)
        R = R.reshape((-1, 2, 2))
        U, _, VH = np.linalg.svd(R, full_matrices=False)
        R = U@VH
        rotated = (R@(z[:, :2][..., np.newaxis])).squeeze()
        return np.hstack((rotated, z[:, 2:]))

    def evidence_lower_bound(self, rng, params, images):
        enc_params, dec_params = params
        R, mu_z, sigmasq_z = self.encode(enc_params, images)
        rotated = self.z_given_x(rng, mu_z, sigmasq_z, R)
        logits_x = self.decode(dec_params, rotated)
        kl = gaussian_kl(mu_z, sigmasq_z, 0., 5.).sum(0)
        log_prob = bernoulli_logpdf(logits_x, images).sum(0)
        elbo = (log_prob - kl) / images.shape[0]
        return elbo

    @property
    def params(self):
        return (self.encoder_params, self.decoder_params)
