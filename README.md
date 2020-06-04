# Stein Variational Gradient Descent for Lie Algebras

We use Singularity to manage our computational environment. The definition file for our Singularity container can be found in `lie-stein.def`. To build the Singularity container itself, you may wish to use the [Singularity remote builder](https://cloud.sylabs.io/builder):
```
singularity build --remote lie-stein.sif lie-stein.def
```
