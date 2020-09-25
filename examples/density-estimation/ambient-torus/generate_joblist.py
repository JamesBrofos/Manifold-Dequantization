job = 'singularity exec --nv ~/scratch60/singularity-containers/manifold-dequantization.sif python torus.py --num-steps {} --elbo-loss 0 --density {} --num-importance {} --num-batch {}\n'

with open('joblist.txt', 'w') as f:
    for dens in ['unimodal', 'multimodal', 'correlated']:
        for num_steps in [1000, 5000, 10000]:
            for num_imp in [10, 20, 30, 40]:
                for num_batch in [100, 500, 1000]:
                    f.write(job.format(num_steps, dens, num_imp, num_batch))
