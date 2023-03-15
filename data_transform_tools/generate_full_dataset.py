import os
import shutil
from os import path
from pathlib import Path

def move(output_path,sub_path,yolo,yolop):
    Path(output_path+"/images/"+sub_path).mkdir(parents=True, exist_ok=True)
    Path(output_path+"/det_annotations/"+sub_path).mkdir(parents=True, exist_ok=True)
    Path(output_path+"/in_seg_annotations/"+sub_path).mkdir(parents=True, exist_ok=True)
    Path(output_path+"/ll_seg_annotations/"+sub_path).mkdir(parents=True, exist_ok=True)
    count = 0
    for in_label in os.listdir(yolo+sub_path+"labels/"):
        shutil.copy(yolo+sub_path+"labels/"+in_label,output_path+"/in_seg_annotations/"+sub_path)
        img = in_label.replace(".txt",".jpg")
        ll_seg = in_label.replace(".txt",".png")
        det_label = in_label.replace(".txt",".json")
        shutil.copy(yolop+"/images/"+sub_path+img,output_path+"/images/"+sub_path)
        shutil.copy(yolop+"/ll_seg_annotations/"+sub_path+ll_seg,output_path+"/ll_seg_annotations/"+sub_path)
        shutil.copy(yolop+"/det_annotations/"+sub_path+det_label,output_path+"/det_annotations/"+sub_path)
        count += 1
    print("Moved :",count)

output_path = 'bdd'
sub_path = 'val/'
yolo = 'YOLO_format/'
yolop = 'unformatted'
move(output_path,sub_path,yolo,yolop)