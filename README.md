# LNLFace (ICASSP25)
---
LNLFace: Enhanced Blind Face Restoration With Local and Non-local Lookups

[Paper](https://ieeexplore.ieee.org/document/10889799) | [Projtect](https://github.com/yanwd628/LNLFace)

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

```
@INPROCEEDINGS{10889799,
  author={Yan, Weidan and Shao, Wenze and Zhang, Dengyin},
  booktitle={ICASSP 2025 - 2025 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)}, 
  title={LNLFace: Enhanced Blind Face Restoration With Local and Non-local Lookups}, 
  year={2025},
  volume={},
  number={},
  pages={1-5},
  keywords={Degradation;Codes;Dictionaries;Dams;Speech coding;Semantics;Signal processing;Image restoration;Speech processing;Faces;blind face restoration;component dictionaries;feature codebook;local and non-local},
  doi={10.1109/ICASSP49660.2025.10889799}}
```

