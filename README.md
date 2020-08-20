# Manifold Dequantization for Sampling

We investigate dequantization as a method for matching distributions on manifolds.

## Computational Environment

We use Singularity to manage our computational environment. The definition file for our Singularity container can be found in `manifold-dequantization.def `. To build the Singularity container itself, you may wish to use the [Singularity remote builder](https://cloud.sylabs.io/builder):
```
singularity build --remote manifold-dequantization.sif manifold-dequantization.def
```
