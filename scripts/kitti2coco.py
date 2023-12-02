import os
import os.path as osp
from collections import defaultdict
import numpy as np
from scipy.spatial.transform import Rotation as R
from PIL import Image
import json
import utils.kitti_utils as ku
import utils.tracking_utils as tu
#import mmcv

def dump_json(filename, data):
    with open(filename, 'w') as f:
        json.dump(data, f)

cats_mapping = {
    'pedestrian': 1,
    'cyclist': 2,
    'car': 3,
    'truck': 4,
    'tram': 5,
    'misc': 6,
    'dontcare': 7
}
kitti_cats = {
    'Pedestrian': 'pedestrian',
    'Cyclist': 'cyclist',
    'Car': 'car',
    'Van': 'car',
    'Truck': 'truck',
    'Tram': 'tram',
    'Person': 'pedestrian',
    'Person_sitting': 'pedestrian',
    'Misc': 'misc',
    'DontCare': 'dontcare'
}

val_sets = ['0001', '0004', '0011', '0012', '0013', '0014', '0015', '0018']

def convert_track(data_dir, mode=None, adjust_center=True):
    kitti = defaultdict(list)

    img_dir = osp.join(data_dir, 'image_02')
    label_dir = osp.join(data_dir, 'label_02')
    cali_dir = osp.join(data_dir, 'calib')
    oxt_dir = osp.join(data_dir, 'oxts')

    if not osp.exists(img_dir):
        print(f"Folder {img_dir} is not found")
        return None

    if not osp.exists(label_dir):
        label_dir = None

    vid_names = sorted(os.listdir(img_dir))
    print(f"{data_dir} with {len(vid_names)} sequences")

    for k, v in cats_mapping.items():
        kitti['categories'].append(dict(id=v, name=k))

    img_id = 0
    global_track_id = 0
    ann_id = 0

    for vid_id, vid_name in enumerate(vid_names):
        if mode == 'train':
            if vid_name in val_sets:
                continue
        elif mode == 'val':
            if vid_name not in val_sets:
                continue
        elif mode == 'mini':
            if vid_name not in mini_sets:
                continue
        print("VID ID: {}".format(vid_id))
        ind2id = dict()
        trackid_maps = dict()
        img_names = sorted([
            f.path for f in os.scandir(osp.join(img_dir, vid_name))
            if f.is_file() and f.name.endswith('png')
        ])
        vid_info = dict(id=vid_id, name=vid_name, n_frames=len(img_names))
        kitti['videos'].append(vid_info)

        projection = ku.read_calib(cali_dir, vid_id)

        for fr, img_name in enumerate(sorted(img_names)):
            img = Image.open(img_name)
            img = np.array(img)
            fields = ku.read_oxts(oxt_dir, vid_id)
            poses = [ku.KittiPoseParser(fields[i]) for i in range(len(fields))]

            rotation = R.from_matrix(poses[fr].rotation).as_euler('xyz')
            position = poses[fr].position - poses[0].position
            pose_dict = dict(rotation=rotation.tolist(),
                             position=position.tolist())

            height, width, _ = img.shape
            index = fr
            img_info = dict(file_name=img_name,
                            cali=projection.tolist(),
                            pose=pose_dict,
                            height=height,
                            width=width,
                            fov=60,
                            near_clip=0.15,
                            id=img_id,
                            video_id=vid_id,
                            index=index)
            kitti['images'].append(img_info)
            ind2id[index] = img_id
            img_id += 1

        if label_dir:
            label_file = osp.join(label_dir, '{}.txt'.format(vid_name))
            #labels = mmcv.list_from_file(label_file)
            with open(label_file, 'r') as f:
                labels = f.readlines()
            for label in labels:
                label = label.split()
                cat = label[2]
                if cat in ['DontCare']:
                    continue
                # if cat not in kitti_cats.keys():
                #     continue
                image_id = ind2id[int(label[0])]
                if label[1] in trackid_maps.keys():
                    track_id = trackid_maps[label[1]]
                else:
                    track_id = global_track_id
                    trackid_maps[label[1]] = track_id
                    global_track_id += 1
                x1, y1, x2, y2 = float(label[6]), float(label[7]), float(
                    label[8]), float(label[9])

                if adjust_center:
                    # KITTI GT uses the bottom of the car as center (x, 0, z).
                    # Prediction uses center of the bbox as center (x, y, z).
                    # So we align them to the bottom center as GT does
                    y_cen_adjust = float(label[10]) / 2.0
                else:
                    y_cen_adjust = 0.0

                center_2d = tu.cameratoimage(
                    np.array([[
                        float(label[13]),
                        float(label[14]) - y_cen_adjust,
                        float(label[15])
                    ]]), projection).flatten().tolist()

                ann = dict(id=ann_id,
                           image_id=image_id,
                           category_id=cats_mapping[kitti_cats[cat]],
                           instance_id=track_id,
                           alpha=float(label[5]),
                           roty=float(label[16]),
                           dimension=[
                               float(label[10]),
                               float(label[11]),
                               float(label[12])
                           ],
                           translation=[
                               float(label[13]),
                               float(label[14]) - y_cen_adjust,
                               float(label[15])
                           ],
                           is_occluded=int(label[4]),
                           is_truncated=float(label[3]),
                           center_2d=center_2d,
                           delta_2d=[
                               center_2d[0] - (x1 + x2) / 2.0,
                               center_2d[1] - (y1 + y2) / 2.0
                           ],
                           bbox=[x1, y1, x2 - x1, y2 - y1],
                           area=(x2 - x1) * (y2 - y1),
                           iscrowd=False,
                           ignore=False,
                           segmentation=[[x1, y1, x1, y2, x2, y2, x2, y1]])
                kitti['annotations'].append(ann)
                ann_id += 1
    return kitti

def main():

    data_dir = 'data/KITTI/'
    out_dir = 'data/KITTI/anns'

    print('Convert KITTI Tracking dataset to COCO style.')
    if not osp.isfile(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    print('tracking validate subset')
    track_dir = osp.join(data_dir, 'tracking', 'training')
    ann = convert_track(track_dir, mode='val')
    dump_json(osp.join(out_dir, 'tracking_subval.json'), ann)

    print("tracking train subset")
    track_dir = osp.join(data_dir, 'tracking', 'training')
    ann = convert_track(track_dir, mode='train')
    dump_json(osp.join(out_dir, 'tracking_subtrain.json'), ann)

if __name__ == "__main__":
    main()
