import argparse
import os
import tqdm
from functools import partial

import matplotlib.pyplot as plt

import jax
import jax.numpy as np
from jax import jit, lax, random, vmap
from jax.experimental import optimizers

import dataset
import vae

parser = argparse.ArgumentParser(description='Learning rotation matrices with a VAE')
parser.add_argument('--num-latent', type=int, default=2, help='Dimensionality of latent space')
parser.add_argument('--step-size', type=float, default=1e-3, help='Gradient descent step size')
parser.add_argument('--num-epochs', type=int, default=100, help='Number of learning epochs')
parser.add_argument('--num-batches', type=int, default=500, help='Number of batches per epoch')
parser.add_argument('--batch-size', type=int, default=128, help='Minibatch size')
parser.add_argument('--seed', type=int, default=0, help='Pseudo-random number generator seed')
args = parser.parse_args()


@vmap
def circular_shift(im, theta):
    im = im.reshape((28, 28))
    roll = np.ceil(28 * theta / (2.*np.pi)).astype(np.int32)
    im = np.roll(im, roll, axis=1).ravel()
    return im

def binarize(rng, images):
    return random.bernoulli(rng, images)

def next_batch(images, it):
    batch = lax.dynamic_slice_in_dim(images, it * args.batch_size, args.batch_size)
    return batch

@jit
def run_epoch(rng, opt_state, images):
    def body_fun(it, opt_state):
        elbo_rng, theta_rng, bin_rng = random.split(random.fold_in(rng, it), 3)
        theta = 2.*np.pi*random.uniform(theta_rng, [args.batch_size])
        batch = next_batch(images, it)
        shifted = circular_shift(batch, theta)
        binbatch = binarize(bin_rng, shifted)
        loss = lambda params: -model.evidence_lower_bound(elbo_rng, params, binbatch)
        g = jax.grad(loss)(get_params(opt_state))
        return opt_update(it, g, opt_state)
    return lax.fori_loop(0, args.num_batches, body_fun, opt_state)



rng = random.PRNGKey(args.seed)
(x_train, y_train), (x_test, y_test) = dataset.mnist()
model = vae.SO2VAE(rng, args.num_latent)
opt_init, opt_update, get_params = optimizers.momentum(args.step_size, mass=0.9)
opt_state = opt_init(model.params)

with tqdm.tqdm(total=args.num_epochs) as pbar:
    for epoch in range(args.num_epochs):
        epoch_rng = random.fold_in(rng, epoch)
        x_train = random.permutation(epoch_rng, x_train)
        opt_state = run_epoch(epoch_rng, opt_state, x_train)
        test_elbo = model.evidence_lower_bound(epoch_rng, get_params(opt_state), x_test)
        pbar.set_postfix(test_elbo=test_elbo)
        pbar.update(1)

        if epoch % 10 == 0:
            theta = 2.*np.pi*random.uniform(rng, [x_test.shape[0]])
            shift_test = circular_shift(x_test, theta)
            encoder_params, decoder_params = get_params(opt_state)
            R, mu_z, sigmasq_z = model.encode(encoder_params, shift_test)
            rotated = model.z_given_x(rng, mu_z, sigmasq_z, R)
            logits_x = model.decode(decoder_params, rotated)
            recon = jax.nn.sigmoid(logits_x)

            plt.figure(figsize=(6, 6))
            for i in range(10):
                idx = y_test == i
                plt.plot(rotated[idx, 0], rotated[idx, 1], '.', label=str(i))
            plt.grid(linestyle=':')
            plt.legend()
            plt.savefig(os.path.join('images', 'latent-space.png'))
            plt.close()

            fig, axes = plt.subplots(2, 4, figsize=(8, 4))
            for i in range(axes.shape[1]):
                axes[0, i].imshow(shift_test[i].reshape((28, 28)), cmap=plt.cm.gray)
                axes[0, i].axis('off')
                axes[1, i].imshow(recon[i].reshape((28, 28)), cmap=plt.cm.gray)
                axes[1, i].axis('off')
                if i == 0:
                    axes[0, i].set_ylabel('Original')
                    axes[1, i].set_ylabel('Reconstruction')

            plt.tight_layout()
            plt.savefig(os.path.join('images', 'reconstruction.png'))
            plt.close()

            Rp = vmap(
                lambda theta: np.array([[np.cos(theta), np.sin(theta)],
                                        [-np.sin(theta), np.cos(theta)]])
            )(np.linspace(0., 2.*np.pi))
            mu_zp = np.tile(mu_z[0], (Rp.shape[0], 1))
            rotated = np.hstack(((Rp@(mu_zp[:, :2][..., np.newaxis])).squeeze(), mu_zp[:, 2:]))
            dec = jax.nn.sigmoid(model.decode(decoder_params, rotated))

            fig, axes = plt.subplots(10, 10, figsize=(12, 12))
            axes = axes.ravel()
            for i in range(len(axes)):
                axes[i].imshow(dec[i].reshape((28, 28)), cmap=plt.cm.gray)
                axes[i].axis('off')
            plt.tight_layout()
            plt.savefig(os.path.join('images', 'rotation.png'))
            plt.close()
