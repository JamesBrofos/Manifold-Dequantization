# Density Estimation on Spheres


This module examines densities on the spherical manifold. The experiments seeks to replicate the experimental setup in [Normalizing Flows on Tori and Spheres](https://arxiv.org/abs/2002.02428). To reproduce these experiments invoke the following command:
```
dSQ --jobfile joblist.txt -p gpu --gres=gpu:1 --gres-flags=enforce-binding -t 24:00:00 -c 10 --job-name sphere -o output/sphere-%A-%J.log --submit --suppress-stats-file
```
