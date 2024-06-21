# Neuro-Symbolic AQA (NS-AQA)
This repository contains the Python code implementation of NS-AQA for platform diving.

Technical Paper:

Demo:

## Overview
We propose a neuro-symbolic paradigm for AQA.
![NS-AQA Concept](teaser_fig.png)

## Run NS-AQA
Score Report is saved as an HTML file at "./output/"

Run NS-AQA on a single dive clip.
```
python nsaqa.py path/to/video.mp4
```
Run NS-AQA on a single dive from the [FineDiving Dataset](https://github.com/xujinglin/FineDiving). Each dive in the dataset has an instance id (x, y).
```
python nsaqa_finediving.py x y

# e.g. if the instance id is ('01', 1)
python nsaqa_finediving.py 01 1
```
