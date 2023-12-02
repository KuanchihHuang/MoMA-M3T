## Installation

This repo is mainly developed with a single V100 GPU on our local environment (python=3.7, cuda=11.1, pytorch=1.7), and we recommend you to use anaconda to create a vitural environment:

```bash
conda create -n momam3t python=3.7
conda activate momam3t
```

Install PyTorch:

```bash
pip3 install torch==1.7.1 torchvision==0.8.2 -f https://download.pytorch.org/whl/cu111/torch_stable.html
```

## Data Preparation

Download data from [KITTI Tracking](https://www.cvlibs.net/datasets/kitti/eval_tracking.php) and [nuScenes](https://www.nuscenes.org/nuscenes#download) datasets.

**data structure**
```
MoMA-M3T
├── checkpoints/
├── tools/
├── scripts/
├── checkpoints/
├── models/
├── configs/
├── data/
│   ├── nuscenes/
│   │   ├── maps/
│   │   ├── samples/
│   │   ├── v1.0-test/
|   |   ├── v1.0-trainval/
│   ├── KITTI/
│   │   ├── tracking/
│   │   │   ├── training/
│   │   │   │   ├── image_02/
│   │   │   │   ├── label_02/
│   │   │   │   ├── calib/
│   │   │   │   ├── oxts/
```

Run the scripts:

```bash
python scripts/convert_nuScenes.py
python scripts/kitti2coco.py
```


