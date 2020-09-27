jobstr = 'singularity exec --nv ~/scratch60/singularity-containers/manifold-dequantization.sif python {} --density {} --num-steps 10000 --elbo-loss {} --num-importance {} --lr 1e-5 --seed {}\n'

with open('joblist.txt', 'w') as f:
    for elbo in [0, 1]:
        for fn in ['torus.py', 'autoregressive.py']:
            for density in ['correlated', 'multimodal', 'unimodal']:
                for seed in range(5):
                    if elbo == 0:
                        for num_importance in [10, 20, 50, 100]:
                            f.write(jobstr.format(fn, density, elbo, num_importance, seed))
                    else:
                        f.write(jobstr.format(fn, density, elbo, 0, seed))
