from easydict import EasyDict as edict
import numpy as np


def getLabels():
    with open('labels.txt', 'r') as f:
        a = list()
        labs = [l.strip() for l in f.readlines()]
        for lab in labs:
            if lab == '----': break
            a += [lab]
    return a
__C = edict()
# Consumers can get config by:
#   from config import cfg
cfg = __C

__C.anchors = np.array([[10, 13], [16, 30], [33, 23], [30, 61], [62, 45], [59, 119], [116, 90], [156, 198], [373, 326]])
__C.classes = 30  # any change made to this must also change yolo_head.py(yolo_head()) and make sure last layer have 3*(class + 5) filters for each detector ()
__C.num = 9
__C.num_anchors_per_layer = 3

__C.batch_size = 1
__C.sample_size = 1600
__C.total_epoch = 1
__C.batch_per_epoch = 3000

__C.scratch = False  # darknet53.conv.74.npz has issue, turn scrach on or test other weights
__C.names = getLabels()
#
# Training options
#
__C.train = edict()

__C.train.ignore_thresh = .5
__C.train.momentum = 0.9
__C.train.decay = 0.0005
__C.train.learning_rate = 0.001
__C.train.max_batches = 50200
__C.train.lr_steps = [40000, 45000]
__C.train.lr_scales = [.1, .1]
__C.train.max_truth = 30
__C.train.mask = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
__C.train.image_resized = 1600   # { 320, 352, ... , 608} multiples of 32

#
# image process options
#
__C.preprocess = edict()
__C.preprocess.angle = 0
__C.preprocess.saturation = 1.5
__C.preprocess.exposure = 1.5
__C.preprocess.hue = .1
__C.preprocess.jitter = .3
__C.preprocess.random = 1
