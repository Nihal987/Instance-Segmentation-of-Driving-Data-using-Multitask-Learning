<div align="left">  

# Multitask Learning of Driving Data

## Objective 
- The primary objective of this project is to improve the performance of <b>Instance Segmentation of Drivable Area</b>. using <b>Multitask learning</b><br/>
- Multitask learning allows a model to learn multiple related tasks simultaneously, which can lead to improved performance on each task and better generalization to new tasks.
- Simply put, this model is able to perform different tasks such as vehicle detection, Drivable Area Instance Segmentation and Detection and Segmentation of Lane markings simultaneoulsy.<br/>
- This computer vision project would be useful for autonomous vehicles while navigating on the road and making decisions for switching lanes.
![output](output/Ins+vehicle_det+lane_seg.jpg)

## Introduction
 This project is an extension of the [YOLOP](https://github.com/hustvl/YOLOP) network, wherein the network has been modified to include Instance Segmentation of the Drivable Area instead of Semantic Segmentation.

### Network Architecture 
![network](output/network.png)

### Contributions
- The original YOLOP network has been modified to add an Instance Segmentation branch. One of the heads that originally performed Semantic Segmentation of the Drivable Area now performs Instance Segmentation of Drivable Area
- The Instance Segmentation method was inspiried from the [YOLOACT](arXiv:1904.02689 ) paper
- The original YOLOP detection head was modified to perform lane detection instead
- Detection and Semantic Segmentation of the Lane Line are performed in parallel to capture information about the lane attributes (as in if it is a dashed white line, solid yellow line etc)
- The model is able to simultaneoulsy train on 3 tasks  in autonomous driving: lane detection detection, drivable area instance segmentation and lane semantic segmentation to save computational costs, reduce inference time as well as improve the performance of each task. 

## Experiements
- Various experiments are conducted to determine what combination of features produces the best results for Instance Segmentation of Drivable area
- By modifying a single parameter in the configuration file of this model, we can selectively freeze specific branches of the network. This allows us to experiment with different branch combinations and conduct various tests.
- The various network combinations are :
    1. Instance Segmentation of Drivable Area [Base] (Ins Seg)
    1. Instance Segmentation of Drivable Area + Semantic Segmentation of Lane Markings (Ins Seg + Lane Sem Seg)
    1. Instance Segmentation of Drivable Area + Semantic Segmentation of Lane Markings + Vehicle Detection (Ins Seg + Lane Sem Seg + Veh Dect)
    1. Instance Segmentation of Drivable Area + Semantic Segmentation of Lane Markings + Detection of Lane Markings (Ins Seg + Lane Sem Seg + Lane Dect)

## Results

#### Drivable Area (DA) Instance Segmentation Bouding box Results

| Training_method                      | Recall(%) |  P(%) | mAP(50%)| mAP(95%) | Speed(ms/frame) |
| ---------------                      | --------- | ----- | ------- | ---------| --------------- |
| `Ins Seg Drivable Area(DA)`          | 64.4      | 62.4  | 64.8    | 28.4     | 6.0             |
| **`Ins Seg DA + Lane Marking Sem Seg`**         | **81.1**  |**77.6**|**83.6** |**59.8**| **5.6**     |
| `Ins Seg DA + Lane Marking Sem Seg + Veh Dect`  | 80.7      | 77.0  | 84.6    | 58.4     | 7.0             |
| `Ins Seg DA + Lane Marking Sem Seg + Lane Dect` | -         | -     | -       | -        | -               |

#### Drivable Area Mask Results

| Training_method                      | Recall(%) |  P(%) | mAP(50%)| mAP(95%) | Speed(ms/frame) |
| ---------------                      | --------- | ----- | ------- | ---------| --------------- |
| `Ins Seg Drivable Area(DA)`          | 59.1      | 58.8  | 58.0    | 23.2     | 6.0             |
| **`Ins Seg DA + Lane Marking Sem Seg`**         |**79.2**   |**76.3**|**81.2**|**54.4**  |**5.6**          |
| `Ins Seg DA + Lane Marking Sem Seg + Veh Dect`  | 78.4      | 75.9  | 81.9    | 53.4     | 7.0             |
| `Ins Seg DA + Lane Marking Sem Seg + Lane Dect` | -         | -     | -       | -        | -               |


## Dataset Structure
Download the dataset from this [link]()

```
bdd
├── det_annotations
│   ├── test
│   ├── train
│   └── val
├── images
│   ├── test
│   ├── train
│   └── val
├── in_seg_annotations
│   ├── test
│   ├── train
│   └── val
├── ll_det_annotations
│   ├── test
│   ├── train
│   └── val
└── ll_seg_annotations
    ├── test
    ├── train
    └── val
```

## Running the Project

### Requirements
The codebase has been developed on Python 3.8.1, PyTorch 1.7+ and torchvision 0.8+:

To install the requirements run the below line
```setup
pip install -r requirements.txt
```

### Training

You can set the training configuration in the `./lib/config/default.py`. (Including:  the loading of preliminary model,  loss,  data augmentation, optimizer, warm-up and cosine annealing, auto-anchor, training epochs, batch_size).

If you want try alternating optimization or train model for single task, please modify the corresponding configuration in `./lib/config/default.py` to `True`. (As following, all configurations is `False`, which means training multiple tasks end to end).

```python
# Alternating optimization
_C.TRAIN.SEG_ONLY = False           # Only train two segmentation branchs
_C.TRAIN.DET_ONLY = False           # Only train detection branch
_C.TRAIN.ENC_SEG_ONLY = False       # Only train encoder and two segmentation branchs
_C.TRAIN.ENC_DET_ONLY = False       # Only train encoder and detection branch

# Single task 
_C.TRAIN.DRIVABLE_ONLY = False      # Only train da_segmentation task
_C.TRAIN.LANE_ONLY = False          # Only train ll_segmentation task
_C.TRAIN.DET_ONLY = False          # Only train detection task
```

Start training:

```shell
python tools/train.py
```



### Evaluation

You can set the evaluation configuration in the `./lib/config/default.py`. (Including： batch_size and threshold value for nms).

Start evaluating:

```shell
python tools/test.py --weights weights/End-to-end.pth
```



### Demo Test

You can store the image or video in `--source`, and then save the reasoning result to `--save-dir`

```shell
python tools/demo.py --source inference/images
```
The demo can be applied to a video as well




