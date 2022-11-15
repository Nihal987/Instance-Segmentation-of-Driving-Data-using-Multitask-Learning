from .utils import initialize_weights, xyxy2xywh, is_parallel, DataLoaderX, torch_distributed_zero_first, clean_str
from .autoanchor import check_anchor_order, run_anchor, kmean_anchors
from .augmentations import augment_hsv, random_perspective, cutout, letterbox,letterbox_for_img
from .plot import plot_img_and_mask,plot_one_box,show_seg_result

import threading

def threaded(func):
    # Multi-threads a target function and returns thread. Usage: @threaded decorator
    def wrapper(*args, **kwargs):
        thread = threading.Thread(target=func, args=args, kwargs=kwargs, daemon=True)
        thread.start()
        return thread

    return wrapper