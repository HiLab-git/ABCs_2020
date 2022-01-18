# The Second Place of MICCAI 2020 ABCs Challenge
[nnUNet_link]:https://github.com/MIC-DKFZ/nnUNetdescribe
[ABCs_link]:https://abcs.mgh.harvard.edu/
This repository provides source code for MICCAI 2020 Anatomical brainBarriers to Cancer spread (ABCs) challenge. Method will be briefly introduced below, and our method won the 2nd place of [ABCs](ABCs_link).
Our method is based on [nnUNet][nnUNet_link], a self-adaptive segmentation method for medical images.

<img src='./HMRnet.png'  width="1100">

# Method overview
Our solution coatriaons corse and fine stage. We first use a localization model based on [nnUNet][nnUNet_link]to obtain a rough localization of different structures, and then use a High- and Multi-Resolution Network(HMRNet)  to  segment  each  structure  around  its  local  regionrespectively, where a Bidirectional Feature Calibration (BFC)block  is  introduced  for  better  interaction  between  features  inthe two branches
