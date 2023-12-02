# MoMA-M3T
**Delving into Motion-Aware Matching for Monocular 3D Object Tracking** (ICCV 2023) [[paper](https://arxiv.org/abs/2308.11607)]\
Kuan-Chih Huang, Ming-Hsuan Yang, Yi-Hsuan Tsai.

<img src="resources/arch.png" alt="vis" style="zoom:50%;" />

## Setup

Please refer to [SETUP.md](SETUP.md) for installation and data preparation.
Download checkpoints and detections [here](https://drive.google.com/drive/folders/1wAYLXk3aWJtYNsNElVuPdO_QHskR5iyy?usp=sharing) to root folder.

## nuScenes Dataset

To evaluate on the validation set:

 ```bash
 sh infer_eval_nusc_mini.sh  #for mini set
 sh infer_eval_nusc_val.sh  #for val set
 ```

## KITTI Dataset

To evaluate on the subval set (for 01,04,11,12,13,14,15,18 sequences):

 ```bash
 sh infer_kitti_subval.sh  #inference
 python ab3dmot_kitti/evaluate.py moma 1 3D 0.25  #evaluation
 ```

## Acknowlegment

Our codes are mainly based on [QD-3DT](https://github.com/SysCV/qd-3dt), and the evaluation code for KITTI dataset is from [AB3DMOT](https://github.com/xinshuoweng/AB3DMOT). Thanks for their contributions.

## Citation
 ```
@inproceedings{huang2023momam3t,
    author = {Kuan-Chih Huang, Ming-Hsuan Yang and Yi-Hsuan Tsai},
    title = {Delving into Motion-Aware Matching for Monocular 3D Object Tracking},
    booktitle = {ICCV},
    year = {2023}    
}
 ```
