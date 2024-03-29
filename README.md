# Manifold Dequantization for Sampling

We investigate dequantization as a method for matching distributions on manifolds.

## Reproducing Results

Every directory in the `examples` folder contains a file called `joblist.txt`. Each line of this file corresponds to a different parameter configuration or comparison that we report in our paper. We use [dSQ](https://github.com/ycrc/dSQ) in order to run these jobs in a cluster environment.

## Computational Environment

We use Singularity to manage our computational environment. The definition file for our Singularity container can be found in `manifold-dequantization.def `. To build the Singularity container itself, you may wish to use the [Singularity remote builder](https://cloud.sylabs.io/builder):
```
singularity build --remote manifold-dequantization.sif manifold-dequantization.def
```
The path to `manifold-dequantization.sif` is assumed to be in the environment variable `$SINGULARITY_CONTAINERS`.
