# Dequantizing Densities on Torus

This module examines dequantizing densities defined on tori. A torus can be represented as a the product manifold of two circles. For dequantization, our approach will be to dequantize these circles into an ambient Euclidean space. We consider three distributions defined on the torus: (i) a unimodal density, (ii) a multimodal density, and (iii) a correlated density, which I have aimed to reproduce from Table 2 in Normalizing Flows on Tori and Spheres.

In these experiments I compare dequantization of the two circles jointly with another method that I call "autoregressive dequantization" wherein the first circle is dequantized which then informs the dequantization of the second circle.

## Unimodal Density

### ELBO Minimization Joint Dequatization

![](images/elbo-unimodal-num-batch-100-num-importance-0-num-steps-10000-seed-0.png)

### KL(p || q) Minimization Joint Dequatization

![](images/kl-unimodal-num-batch-100-num-importance-100-num-steps-10000-seed-0.png)

### ELBO Minimization Autoregressive Dequatization

![](images/autoregressive-elbo-unimodal-num-batch-100-num-importance-0-num-steps-10000-seed-0.png)

### KL(p || q) Minimization Autoregressive Dequatization

![](images/autoregressive-kl-unimodal-num-batch-100-num-importance-100-num-steps-10000-seed-0.png)

Here is the comparison against the Mobius-spline flow for the unimodal density.

![](images/torus-unimodal-density.png)

## Multimodal Density

### ELBO Minimization Joint Dequatization

![](images/elbo-multimodal-num-batch-100-num-importance-0-num-steps-10000-seed-0.png)

### KL(p || q) Minimization Joint Dequatization

![](images/kl-multimodal-num-batch-100-num-importance-100-num-steps-10000-seed-0.png)

### ELBO Minimization Autoregressive Dequatization

![](images/autoregressive-elbo-multimodal-num-batch-100-num-importance-0-num-steps-10000-seed-0.png)

### KL(p || q) Minimization Autoregressive Dequatization

![](images/autoregressive-kl-multimodal-num-batch-100-num-importance-100-num-steps-10000-seed-0.png)

Here is the comparison against the Mobius-spline flow for the multimodal density.

![](images/torus-multimodal-density.png)

## Correlated Density

### ELBO Minimization Joint Dequatization

![](images/elbo-correlated-num-batch-100-num-importance-0-num-steps-10000-seed-0.png)

### KL(p || q) Minimization Joint Dequatization

![](images/kl-correlated-num-batch-100-num-importance-100-num-steps-10000-seed-0.png)

### ELBO Minimization Autoregressive Dequatization

![](images/autoregressive-elbo-correlated-num-batch-100-num-importance-0-num-steps-10000-seed-0.png)

### KL(p || q) Minimization Autoregressive Dequatization

![](images/autoregressive-kl-correlated-num-batch-100-num-importance-100-num-steps-10000-seed-0.png)

Here is the comparison against the Mobius-spline flow for the correlated density.

![](images/torus-correlated-density.png)

Invoke the following to reproduce these results.
```
dSQ --jobfile joblist.txt -p gpu --gres=gpu:1 --gres-flags=enforce-binding -t 24:00:00 -c 10 --job-name torus -o output/torus-%A-%J.log --submit --suppress-stats-file
cp ../../sphere-flow/images/torus-unimodal-density.png images/
cp ../../sphere-flow/images/torus-multimodal-density.png images/
cp ../../sphere-flow/images/torus-correlated-density.png images/
```
