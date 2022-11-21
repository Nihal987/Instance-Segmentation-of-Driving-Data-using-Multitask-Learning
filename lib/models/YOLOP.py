import torch
from torch import tensor
import torch.nn as nn
import sys,os
import math
import sys

sys.path.append(os.getcwd())
#sys.path.append("lib/models")
#sys.path.append("lib/utils")
#sys.path.append("/workspace/wh/projects/DaChuang")
from lib.utils import initialize_weights
# from lib.models.common2 import DepthSeperabelConv2d as Conv
# from lib.models.common2 import SPP, Bottleneck, BottleneckCSP, Focus, Concat, Detect
from lib.models.common import Conv, SPP, Bottleneck, BottleneckCSP, Focus, Concat, Detect, SharpenConv, IDetect, ISegment
from torch.nn import Upsample
from lib.utils import check_anchor_order
from lib.core.evaluate import SegmentationMetric
from lib.config import cfg
from lib.utils.utils import time_synchronized


# The lane line and the driving area segment branches without share information with each other and without link
nc_in = 2
names_in = ["direct","alternative"]
detect_anchors = [[3,9,5,11,4,20], [7,18,6,39,12,31], [19,50,38,81,68,157]]
ins_seg_anchors = [[10,13, 16,30, 33,23], [30,61, 62,45, 59,119], [116,90, 156,198, 373,326]]

YOLOP = [
[24, 33, 41],   #Det_out_idx, Da_Segout_idx, LL_Segout_idx
[ -1, Focus, [3, 32, 3]],   #0
[ -1, Conv, [32, 64, 3, 2]],    #1
[ -1, BottleneckCSP, [64, 64, 1]],  #2
[ -1, Conv, [64, 128, 3, 2]],   #3
[ -1, BottleneckCSP, [128, 128, 3]],    #4
[ -1, Conv, [128, 256, 3, 2]],  #5
[ -1, BottleneckCSP, [256, 256, 3]],    #6
[ -1, Conv, [256, 512, 3, 2]],  #7
[ -1, SPP, [512, 512, [5, 9, 13]]],     #8
[ -1, BottleneckCSP, [512, 512, 1, False]],     #9
[ -1, Conv,[512, 256, 1, 1]],   #10
[ -1, Upsample, [None, 2, 'nearest']],  #11
[ [-1, 6], Concat, [1]],    #12
[ -1, BottleneckCSP, [512, 256, 1, False]], #13
[ -1, Conv, [256, 128, 1, 1]],  #14
[ -1, Upsample, [None, 2, 'nearest']],  #15
[ [-1,4], Concat, [1]],     #16         #Encoder

[ -1, BottleneckCSP, [256, 128, 1, False]],     #17
[ -1, Conv, [128, 128, 3, 2]],      #18
[ [-1, 14], Concat, [1]],       #19
[ -1, BottleneckCSP, [256, 256, 1, False]],     #20
[ -1, Conv, [256, 256, 3, 2]],      #21
[ [-1, 10], Concat, [1]],   #22
[ -1, BottleneckCSP, [512, 512, 1, False]],     #23
[ [17, 20, 23], Detect,  [1, detect_anchors, [128, 256, 512]]], #Detection head 24

[ 16, Conv, [256, 128, 3, 1]],   #25
[ -1, Upsample, [None, 2, 'nearest']],  #26
[ -1, BottleneckCSP, [128, 64, 1, False]],  #27
[ -1, Conv, [64, 32, 3, 1]],    #28
[ -1, Upsample, [None, 2, 'nearest']],  #29
[ -1, Conv, [32, 16, 3, 1]],    #30
[ -1, BottleneckCSP, [16, 8, 1, False]],    #31
[ -1, Upsample, [None, 2, 'nearest']],  #32
[ -1, Conv, [8, 2, 3, 1]], #33 Lane line segmentation head

[ 16, BottleneckCSP, [256, 128, 1, False]],     #34
[ -1, Conv, [128, 128, 3, 2]],      #35
[ [-1, 14], Concat, [1]],       #36
[ -1, BottleneckCSP, [256, 256, 1, False]],     #37
[ -1, Conv, [256, 256, 3, 2]],      #38
[ [-1, 10], Concat, [1]],   #39
[ -1, BottleneckCSP, [512, 512, 1, False]],     #40
[ [34, 37, 40], ISegment,  [nc_in, ins_seg_anchors, 32, 128, [128, 256, 512] ]], # Instance Segmentation Head 41
]


class MCnet(nn.Module):
    def __init__(self, block_cfg, **kwargs):
        super(MCnet, self).__init__()
        layers, save= [], []
        self.nc = 1
        self.nc_in = nc_in # Hard codded
        self.detector_index = -1
        self.det_out_idx = block_cfg[0][0]
        self.ll_seg_out_idx = block_cfg[0][1]
        self.in_seg_out_idx = block_cfg[0][2]
       
        # Build model
        print(f"\n{'':>3}{'from':>18} {'module':<40}{'arguments':<30}")
        for i, (from_, block, args) in enumerate(block_cfg[1:]):
            block = eval(block) if isinstance(block, str) else block  # eval strings
            if block is Detect:
                self.detector_index = i
            block_ = block(*args)
            block_.index, block_.from_ = i, from_
            layers.append(block_)
            t = str(block)[8:-2].replace('__main__.', '')  # module type
            print(f'{i:>3}{str(from_):>18}  {t:<40}{str(args):<30}')  # print
            save.extend(x % i for x in ([from_] if isinstance(from_, int) else from_) if x != -1)  # append to savelist
        assert self.detector_index == block_cfg[0][0]

        self.model, self.save = nn.Sequential(*layers), sorted(save)
        self.names = [str(i) for i in range(self.nc)]
        self.names_in = names_in # Hard coded

        # set stride、anchor for detector and segment
        indexes = [self.detector_index,-1]
        for index in indexes:
            m = self.model[index]  # Segment
            if isinstance(m, (Detect)):
                s = 128  # 2x min stride
                with torch.no_grad():
                    model_out = self.forward(torch.zeros(1, 3, s, s))
                    detects, _, _= model_out
                    m.stride = torch.tensor([s / x.shape[-2] for x in detects])  # forward
                print("stride"+str(m.stride ))
                m.anchors /= m.stride.view(-1, 1, 1)  # Set the anchors for the corresponding scale
                type = "Detect"
                check_anchor_order(m,type)
                self.stride = m.stride
                self._initialize_biases(type=type)

            if isinstance(m, (IDetect,ISegment)):
                s = 128 # 2x min stride
                forward = lambda x: self.forward(x)[0] if isinstance(m, ISegment) else self.forward(x)
                m.stride = torch.tensor([s / x.shape[-2] for x in forward(torch.zeros(1, 3, s, s))]) 
                # with torch.no_grad():
                #     model_out = self.forward(torch.zeros(1, 3, s, s))[0]
                #     detects, _, _= model_out
                #     m.stride = torch.tensor([s / x.shape[-2] for x in detects])  # forward
                print("stride"+str(m.stride ))
                m.anchors /= m.stride.view(-1, 1, 1)  # Set the anchors for the corresponding scale
                type = "IDetect" if isinstance(m, (IDetect,ISegment)) else "Detect"
                check_anchor_order(m,type)
                self.stride = m.stride
                self._initialize_biases(type=type)
        
        initialize_weights(self)

    def forward(self, x):
        cache = []
        out = []
        det_out = None
        ins_out = None
        for i, block in enumerate(self.model):
            if block.from_ != -1:
                x = cache[block.from_] if isinstance(block.from_, int) else [x if j == -1 else cache[j] for j in block.from_]       #calculate concat detect
            x = block(x)
            if i == self.ll_seg_out_idx:     #save driving area segment result
                m=nn.Sigmoid()
                out.append(m(x))
            if i == self.detector_index:
                det_out = x
            if i == self.in_seg_out_idx:
                ins_out = x
            cache.append(x if block.index in self.save else None)
        out.insert(0,det_out)
        out.append(ins_out)
        return out
            
    
    def _initialize_biases(self, cf=None,type="Detect"):  # initialize biases into Detect(), cf is class frequency
        # https://arxiv.org/abs/1708.02002 section 3.3
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.
        # m = self.model[-1]  # Detect() module
        index = -1 if type=="IDetect" else self.detector_index
        m = self.model[index]  # Detect() module
        for mi, s in zip(m.m, m.stride):  # from
            b = mi.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
            b.data[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
            b.data[:, 5:] += math.log(0.6 / (m.nc - 0.99)) if cf is None else torch.log(cf / cf.sum())  # cls
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

def get_net(cfg, **kwargs): 
    m_block_cfg = YOLOP
    model = MCnet(m_block_cfg, **kwargs)
    return model


if __name__ == "__main__":
    # from torch.utils.tensorboard import SummaryWriter
    model = get_net(False)
    input_ = torch.randn((1, 3, 256, 256))
    model_out = model(input_)
    detects, lane_line_seg, ins_seg = model_out

    for det in detects:
        print(det.shape)
    print(lane_line_seg.shape)
    print("Instance Segmentation")
    p,proto = ins_seg
    for p1 in p:
        print(p1.shape)
    print(proto.shape)
 
