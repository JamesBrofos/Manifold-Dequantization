job = 'singularity exec --nv $SINGULARITY_CONTAINERS/manifold-dequantization.sif python {}.py --num-steps {} --density {} --seed {}'

with open('joblist.txt', 'w') as f:
    for num_steps in [10000]:
        for density in ['unimodal', 'multimodal', 'correlated']:
            for seed in range(10):
                for method in ['normalizing', 'direct', 'dequantization']:
                    if method == 'dequantization':
                        f.write(job.format(method, num_steps, density, seed) + ' --elbo-loss 0' + '\n')
                        f.write(job.format(method, num_steps, density, seed) + ' --elbo-loss 1' + '\n')
                    else:
                        f.write(job.format(method, num_steps, density, seed) + '\n')
