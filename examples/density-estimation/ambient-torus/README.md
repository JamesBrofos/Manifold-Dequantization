```
dSQ --jobfile joblist.txt -p gpu --gres=gpu:1 --gres-flags=enforce-binding -t 24:00:00 -c 10 --submit --suppress-stats-file 
```
```
singularity exec --nv ~/scratch60/singularity-containers/manifold-dequantization.sif python torus.py --num-steps 5000 --elbo-loss 0 --density correlated
singularity exec --nv ~/scratch60/singularity-containers/manifold-dequantization.sif python torus.py --num-steps 5000 --elbo-loss 0 --density multimodal --num-importance 40
```
