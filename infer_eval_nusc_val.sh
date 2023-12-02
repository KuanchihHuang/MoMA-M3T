#inference
#pip install motmetrics==1.2.0
#CUDA_VISIBLE_DEVICES=2 python  inference.py --config ./configs/Nusc/nusc_val.yaml
#evaluation
pip install motmetrics==1.1.3
python eval_nusc_mot.py --version=v1.0-trainval --root=data/nuscenes/ --work_dir=./output/Nusc --gt_anns=data/nuscenes/anns/tracking_val.json
