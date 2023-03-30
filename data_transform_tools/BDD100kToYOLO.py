import os
import sys
import json
import argparse
import shutil
from os import path
from pathlib import Path

def convert_labels(input_labels, input_images,output_path,class_list):
    Path(output_path+"/images").mkdir(parents=True, exist_ok=True)
    Path(output_path+"/labels").mkdir(parents=True, exist_ok=True)
    with open(input_labels,"r") as ann_file:
        data = json.load(ann_file)
        for img_filename in os.listdir(input_images):
            for entry in data:
                if img_filename == entry["name"] and "labels" in entry.keys():
                    shutil.copy(input_images+"/"+img_filename,output_path+"/images/"+img_filename)
                    filename = img_filename
                    filename = filename.replace("jpg","txt")
                    # os.chdir(output_path)
                    with open(output_path+"/labels/"+filename,"w") as text_file:
                        for label in entry['labels']:
                            class_name = label["category"]
                            if class_name not in class_list:
                                continue
                            class_id = class_list.index(str(class_name))
                            normalized_point_list = []
                            normalized_point_list.append(class_id)
                            vertices = label["poly2d"][0]["vertices"]
                            for vertex in vertices:
                                (px,py) = vertex
                                if(px>1280 and px<1281):
                                    px = 1
                                else:
                                    px = px/1280
                                if(py>720 and py<721):
                                    py = 1
                                else:
                                    py = py/720
                                normalized_point_list.append(px)
                                normalized_point_list.append(py)
                            for i in range (len(normalized_point_list)):
                                text_file.write(str(normalized_point_list[i])+" ")
                            text_file.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_labels',type=str, default=None,
                        help='Path of the poly2d json file')
    parser.add_argument('--input_images',type=str, default=None,
                        help='Path of the images folder')
    parser.add_argument('--output_path',type=str, default=None,
                        help='Path for the output labels and images')

    args = parser.parse_args(sys.argv[1:])

    # input_labels = "old_format/annotations/drivable_val.json"
    # input_images = "old_format/images/val"
    # output_path = "val"

    class_list = ["direct","alternative"]
    print("------BEGINING CONVERSION------")
    convert_labels(args.input_labels,args.input_images,args.output_path,class_list)
    print("--------------DONE-------------")

# python3 BDD100kToYOLO.py --input_labels old_format/annotations/drivable_val.json --input_images old_format/images/val --output_path val
# python3 BDD100kToYOLO.py --input_labels old_format/annotations/drivable_train.json --input_images old_format/images/train --output_path train