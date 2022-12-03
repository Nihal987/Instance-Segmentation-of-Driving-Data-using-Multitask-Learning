import numpy as np
import json

from .AutoDriveDataset import AutoDriveDataset
from .convert import convert, id_dict, id_dict_single
from tqdm import tqdm
from os.path import exists
from ..core.general import segments2boxes

single_cls = True       # just detect vehicle

class BddDataset(AutoDriveDataset):
    def __init__(self, cfg, is_train, inputsize, transform=None):
        super().__init__(cfg, is_train, inputsize, transform)
        self.db = self._get_db()
        self.cfg = cfg

    def _get_db(self):
        """
        get database from the annotation file

        Inputs:

        Returns:
        gt_db: (list)database   [a,b,c,...]
                a: (dictionary){'image':, 'information':, ......}
        image: image path
        mask: path of the segmetation label
        label: [cls_id, segmentation2box] 
        in_seg_mask: poly2D seg format
        """
        print('building database...')
        gt_db = []
        height, width = self.shapes
        for lane in tqdm(list(self.lane_list)):
            lane_path = str(lane)
            if lane_path.endswith('.DS_Store'):
                continue
            label_path = lane_path.replace(str(self.lane_root), str(self.label_root)).replace(".png", ".json")
            image_path = lane_path.replace(str(self.lane_root), str(self.img_root)).replace(".png", ".jpg")
            in_mask_path = lane_path.replace(str(self.lane_root), str(self.in_mask_root)).replace(".png", ".txt")

            with open(label_path, 'r') as f:
                label = json.load(f)
            data = label['frames'][0]['objects']
            data = self.filter_data(data)
            gt = np.zeros((len(data), 5))
            for idx, obj in enumerate(data):
                category = obj['category']
                if category == "traffic light":
                    color = obj['attributes']['trafficLightColor']
                    category = "tl_" + color
                if category in id_dict.keys():
                    x1 = float(obj['box2d']['x1'])
                    y1 = float(obj['box2d']['y1'])
                    x2 = float(obj['box2d']['x2'])
                    y2 = float(obj['box2d']['y2'])
                    cls_id = id_dict[category]
                    if single_cls:
                         cls_id=0
                    gt[idx][0] = cls_id
                    box = convert((width, height), (x1, x2, y1, y2))
                    gt[idx][1:] = list(box)

            # Verify Instance Segmentation Labels
            with open(in_mask_path, 'r') as f:
                lb = [x.split() for x in f.read().strip().splitlines() if len(x)]
                if any(len(x) > 6 for x in lb):  # is segment
                    classes = np.array([x[0] for x in lb], dtype=np.float32)
                    segments = [np.array(x[1:], dtype=np.float32).reshape(-1, 2) for x in lb]  # (cls, xy1...)
                    lb = np.concatenate((classes.reshape(-1, 1), segments2boxes(segments)), 1)  # (cls, xywh)
                lb = np.array(lb, dtype=np.float32)
            nl = len(lb)
            if nl:
                assert lb.shape[1] == 5, f'labels require 5 columns, {lb.shape[1]} columns detected'
                assert (lb >= 0).all(), f'negative label values {lb[lb < 0]}'
                assert (lb[:, 1:] <= 1).all(), f'non-normalized or out of bounds coordinates {lb[:, 1:][lb[:, 1:] > 1]}'
                _, i = np.unique(lb, axis=0, return_index=True)
                if len(i) < nl:  # duplicate row check
                    lb = lb[i]  # remove duplicates
                    if segments:
                        #segments = segments[i]
                        segments = [segments[x] for x in i]
                    msg = f'WARNING: {image_path}: {nl - len(i)} duplicate labels removed'
                    print(msg)
                
            rec = [{
                'image': image_path,
                'label': gt,
                'in_label': lb,
                'lane': lane_path,
                'in_segments': segments
            }]

            gt_db += rec
        print('database build finish')
        return gt_db

    def filter_data(self, data):
        remain = []
        for obj in data:
            if 'box2d' in obj.keys():  # obj.has_key('box2d'):
                if single_cls:
                    if obj['category'] in id_dict_single.keys():
                        remain.append(obj)
                else:
                    remain.append(obj)
        return remain

    def evaluate(self, cfg, preds, output_dir, *args, **kwargs):
        """  
        """
        pass
