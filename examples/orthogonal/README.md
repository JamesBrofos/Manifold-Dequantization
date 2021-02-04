# Density Estimation on Orthogonal Group


This module examines densities on the orthogonal group. To reproduce these experiments invoke the following command:
```
dSQ --jobfile joblist.txt -p gpu --gres=gpu:v100:1 --gres-flags=enforce-binding -t 24:00:00 -c 10 --job-name orthogonal -o output/orthogonal-%A-%J.log --submit --suppress-stats-file
```
One may also invoke any single line job from `joblist.txt` in order to run that experiment.
