# LNLFace (Under Review)
---
LNLFace: Enhanced Blind Face Restoration With Local and Non-local Lookups


## Dependencies
+ Python 3.6
+ PyTorch >= 1.7.0
+ matplotlib
+ opencv
+ torchvision
+ numpy


## Datasets are provided in [here](https://github.com/wzhouxiff/RestoreFormer?tab=readme-ov-file#preparations-of-dataset-and-models)


## Train and Test (based on [Basicsr](https://github.com/XPixelGroup/BasicSR))

    python lnlface/train.py -opt options/train/[xx].yml --auto_resume
    python inference.py

**ps: the path configs should be changed to your own path**




