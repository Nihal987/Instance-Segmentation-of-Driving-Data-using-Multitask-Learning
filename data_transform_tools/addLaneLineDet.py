import os
import sys
import json
import argparse
import shutil
from os import path
from pathlib import Path

def convert(target,images,laneLine,class_list):
    Path(target).mkdir(parents=True, exist_ok=True)
    with open(laneLine,"r") as ann_file:
        data = json.load(ann_file)
        count = 1
        for img_filename in os.listdir(images):
            for entry in data:
                if img_filename == entry["name"] and "labels" in entry.keys():
                    filename = img_filename
                    filename = filename.replace("jpg","json")
                    with open(target+"/"+filename,"w") as json_file:

                        for label in entry['labels']:
                            poly2d = label['poly2d'][0]["vertices"]
                            del label['poly2d']
                            x1,y1 = poly2d[0][0],poly2d[0][1]
                            x2,y2 = poly2d[1][0],poly2d[1][1]
                            box2d = {}
                            box2d["x1"] = x1
                            box2d["y1"] = y1
                            box2d["x2"] = x2
                            box2d["y2"] = y2
                            label['box2d'] = box2d
                            id = int(label['id'])
                            del label['id']
                            label['id'] = id

                        frames = []
                        objects = {}
                        objects["objects"] = entry['labels']
                        frames.append(objects)
                        del entry['labels']
                        entry['frames'] = frames
                        json_object = json.dumps(entry)
                        json_file.write(json_object)
            print("Count:",count)
            count+=1
                        

if __name__ == "__main__":
    target = "bdd/ll_det_annotations/train"
    images = "bdd/images/train"
    laneLine = "ll_det_annotations/lane_train.json"
    class_list = ["crosswalk","double other","double white","double yellow","road curb","single other","single white","single yellow"]
    convert(target,images,laneLine,class_list)
    print("CONVERSION DONE")