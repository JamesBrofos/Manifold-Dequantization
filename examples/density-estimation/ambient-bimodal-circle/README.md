# Ambient Score Matching 

Run these commands to reproduce:
```
dSQ --jobfile joblist.txt -p gpu --gres=gpu:1 --gres-flags=enforce-binding --max-jobs 1000 -c 10 -t 2:00:00 --job-name circle-dequantization -o output/circle-dequantization-%A-%J.log --suppress-stats-file --submit
```
