# LNLFace (ICASSP25)
---
LNLFace: Enhanced Blind Face Restoration With Local and Non-local Lookups

Paper (comming) | [Projtect](https://github.com/yanwd628/LNLFace)

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

Our pretrained model is available [Google Drive](https://drive.google.com/file/d/1C27l1RdvUXdKCb1RVPbXy2ANo4nxCQGE/view?usp=sharing)



