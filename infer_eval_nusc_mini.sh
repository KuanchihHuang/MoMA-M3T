#inference
pip install motmetrics==1.2.0
CUDA_VISIBLE_DEVICES=0 python  inference.py --config ./configs/Nusc/nusc_minival.yaml
#evaluation
pip install motmetrics==1.1.3
python eval_nusc_mot.py --version=v1.0-mini --root=data/nuscenes/ --work_dir=./output/Nusc --gt_anns=data/nuscenes/anns/tracking_val_mini.json
