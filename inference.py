import argparse
from collections import defaultdict
import os
import os.path as osp
import torch
from models.tracker_manager import MotionTrackerManager
import tqdm
import copy
import yaml
import json
import pickle
from utils.coco_format import read_file
from utils.general_output import general_output
import utils.tracking_utils as tu
from datasets.video_parser import VID


cat_mapping = {
    'kitti': {"car": 2, "pedestrian": 0, "cyclist": 1},
    'nuscenes': {'bicycle': 0, 'motorcycle': 1, 'pedestrian': 2, 'bus': 3, 'car': 4, 'trailer': 5, 'truck': 6, 'construction_vehicle': 7, 'traffic_cone': 8, 'barrier': 9}
}

def parse_args():
    parser = argparse.ArgumentParser(description="motion tracker")
    parser.add_argument("--config", dest='config', help='settings of tracker in yaml format')
    args = parser.parse_args()
    return args

def single_gpu_test(
    model,
    dataset_name,
    dets,
    out_path: str = None,
    img_infos: list = [],
):
    outputs = defaultdict(list)
    prog_bar = tqdm.tqdm(total=len(img_infos))

    coco_outputs = defaultdict(list)
    pred_id = 0
    modelcats = cat_mapping[dataset_name]

    for i, data in enumerate(img_infos):
        with torch.no_grad():

            img_info = copy.deepcopy(img_infos[i])
            img_info.update({"img_info": img_infos[i]})
            img_info.update({"calib": img_infos[i]["cali"]})
            img_info.update({"frame_id": img_infos[i]["index"]})

            if dataset_name == 'nuscenes':  # for nuscenes
                filename_clean = '/'.join(img_info['filename'].split('/')[2:])
                det = dets[filename_clean]
                
                #transform category id from mmdetection3d to ours
                mmdet_cats = ['car', 'truck', 'trailer', 'bus', 'construction_vehicle', 'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone', 'barrier']

                boxes_3d = det['boxes_3d'][:,:7]
                boxes_2d = det['boxes_2d']
                scores_3d = det['scores_3d']
                labels_3d = det['labels_3d']

                dim = boxes_3d[:,3:6][:,[1,2,0]] #lhw(mmdet3D)->hwl(ours)
                loc = boxes_3d[:,:3]
                #TODO
                loc[:,1] -= (dim[:,0] / 2.0)
                dep = boxes_3d[:,2]
                rot = boxes_3d[:,-1]
                det_yaws = -1 * tu.alpha2yaw_torch(-1 *rot, loc[:, 0:1].view(-1), loc[:, 2:3].view(-1))

                cat = [modelcats[mmdet_cats[i]] for i in labels_3d]
        
                det_labels = torch.tensor(cat).view(-1).cuda()
                det_bboxes = boxes_2d.cuda()
                projection = det_bboxes.new_tensor(img_info['calib'])
                det_depths = dep.cuda().view(-1,1).float()
                det_dims = dim.cuda().float()
                det_2dcs = tu.cameratoimage_torch(loc.cuda().float(), projection)
                det_alphas = det_yaws.view(-1,1).cuda().float()
                
            elif dataset_name == 'kitti':  # for kitti
                log_id = img_info["video_id"]
                fr_id = img_info["frame_id"]
                det = dets[log_id]["frames"][fr_id]["annotations"]
                
                depth_results = []
                bbox_results = [] 
                dim_results = []
                alpha_results = []
                cen_2ds_results = []
                depth_uncertainty_results = []
                tracking_results = {}
                label_results = []

                for i, ann in enumerate(det):
                    depth_results.append(ann['depth'])
                    dim_results.append(ann['dimension'])
                    alpha_results.append(ann['alpha'])
                    cen_2ds_results.append(ann['box_center'])
                    bbox_results.append(ann['box']+[ann['score']])
                    label_results.append(modelcats[ann['obj_type'].lower()])

                det_depths = torch.tensor(depth_results).cuda().view(-1,1)
                det_bboxes = torch.tensor(bbox_results).cuda().view(-1,5)
                det_labels = torch.tensor(label_results).cuda().view(-1)
                det_dims = torch.tensor(dim_results).cuda().view(-1,3)
                det_alphas = torch.tensor(alpha_results).cuda().view(-1,1)
                det_2dcs = torch.tensor(cen_2ds_results).cuda().view(-1,2)
            
            else:
                raise NotImplementedError
            
            det = {'det_labels': det_labels, 'det_bboxes': det_bboxes, 'det_depths': det_depths, 'det_dims': det_dims, 'det_2dcs': det_2dcs, 'det_alphas': det_alphas}
            
            modelcats = cat_mapping[dataset_name]
            result, use_3d_center = model.simple_test(
                det_out=det, img_meta=img_info
            )

        coco_outputs, pred_id = general_output(
            coco_outputs, result, img_info, use_3d_center, pred_id, modelcats, out_path
        )

        prog_bar.update()
    prog_bar.close()

    return coco_outputs


if __name__ == "__main__":
    args = parse_args()

    cfg = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
    out_path = cfg['output_path']
    info_path = cfg['info_path']
    det_path = cfg['detection_path']
    dataset_name = cfg['dataset_name']

    if dataset_name == "kitti":
        cats = [cat.capitalize() for cat in list(cat_mapping[dataset_name].keys())]
        dets = read_file(det_path, category=cats)
    elif dataset_name == "nuscenes":
        with open(det_path, "rb") as f:
            dets = pickle.load(f)['results']
    else:
        raise NotImplementedError
    
    os.makedirs(out_path, exist_ok=True)
    out_json_path = osp.join(out_path, "output.json")

    img_infos = []
    vid = VID(info_path)
    vid_ids = vid.getVidIds()
    
    for vid_id in vid_ids:
        img_ids = vid.getImgIdsFromVidId(vid_id)
        for img_id in img_ids:
            info = vid.loadImgs([img_id])[0]
            info["filename"] = info["file_name"]
            info["type"] = "VID"
            info["first_frame"] = True if info["index"] == 0 else False
            img_infos.append(info)

    model = MotionTrackerManager()

    coco_outputs = single_gpu_test(
        model,
        dataset_name,
        dets=dets,
        out_path=out_path,
        img_infos=img_infos,
    )

    print(f"\nwriting results to {out_path}")

    with open(out_json_path, "w") as f:
        json.dump(coco_outputs, f)

