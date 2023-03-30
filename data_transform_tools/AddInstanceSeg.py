import os
import shutil
from os import path
from pathlib import Path

def moveAnnotations(output_path,in_seg_folder,input_path,context):
    Path(output_path+in_seg_folder+context).mkdir(parents=True, exist_ok=True)
    true,false = 0,0
    for img in os.listdir(output_path+"images/"+context):
        annot_text = img.replace("jpg","txt")
        # print(input_path+"/"+annot_text)
        if path.exists(input_path+annot_text):
            true+=1
            shutil.copy(input_path+annot_text,output_path+in_seg_folder+"/"+context)
        else:
            false+=1
    print("True: ",true)
    print("False: ",false)

    

context = "val"
in_seg_folder = "in_seg_annotations/"
output_path = "bdd/"
input_path = "YOLO_format/"+context+"/labels/"

moveAnnotations(output_path,in_seg_folder,input_path,context)
print("DONE")