Try these commands:
```
singularity exec ~/scratch60/singularity-containers/lie-stein.sif python sphere.py --num-steps 2000 --num-samples 50 --step-size 1e-4 --num-discrete 50 --num-hidden 10
```

```
singularity exec ~/scratch60/singularity-containers/lie-stein.sif python circle.py --num-steps 100 --real-nvp 1 --num-samples 100 --step-size 1e-3 --num-discrete 50 --seed 1
```

Movie:
```
singularity exec ~/scratch60/singularity-containers/lie-stein.sif ffmpeg -y -r 60 -f image2 -s 1920x1080 -i images/sphere-samples-%d.png -vcodec libx264 -crf 25  -pix_fmt yuv420p sphere-samples.mp4
```
