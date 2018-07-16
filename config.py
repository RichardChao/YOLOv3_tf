from easydict import EasyDict as edict
import numpy as np

__C = edict()
# Consumers can get config by:
#   from config import cfg
cfg = __C

# train with real camera data
# image_dir = 'C:\\Users\\P900\\Desktop\\myWork\\dmg_inspect_YOLOv3\\CID_project_dataset\\CID_Photo'
# annotation_dir = 'C:\\Users\\P900\\Desktop\\myWork\\dmg_inspect_YOLOv3\\CID_project_dataset\\annotation'
# label_dir = 'labels.txt'
# annotation_dir_deep = False
# __C.image_format = 'jpg'

# train with real scanner data
image_dir = 'C:\\Users\\P900\\Desktop\\myWork\\upload_CID_pic\\图片'
annotation_dir = 'C:\\Users\\P900\\Desktop\\myWork\\upload_CID_pic\\标记'
label_dir = 'labels.txt'
annotation_dir_deep = False
__C.image_format = 'jpg'


# train with ImageNet Dataset
# image_dir = 'D:\\dataset\\ILSVRC 2014 DET train\\images\\ILSVRC_2014'
# annotation_dir = 'C:\\Users\\P900\\Desktop\\myWork\\YOLOv3_tf\\LSVRC2014_annotation'
# image_dir = 'D:\\dataset\\ILSVRC 2014 DET train\\images'
# annotation_dir = 'D:\\ILSVRC2012_bbox_train_v2'
# annotation_dir = 'C:\\Users\\P900\\Desktop\\myWork\\YOLOv3_tf\\LSVRC2013_annotation'
# annotation_dir_deep = True
# label_dir = 'LSVRC2014_label_200.txt'
# __C.image_format = 'JPEG'


ckpt_dir = 'C:\\Users\\P900\\Desktop\\myWork\\YOLOv3_tf\\ckpt\\'

testset = 'C:\\Users\\P900\\Desktop\\myWork\\YOLOv3_tf\\testset'
result_dir = 'C:\\Users\\P900\\Desktop\\myWork\\YOLOv3_tf\\result'
def getLabels():
    with open(label_dir, 'r') as f:
        a = list()
        labs = [l.strip() for l in f.readlines()]
        for lab in labs:
            if lab == '----': break
            a += [lab]
    return a

__C.anchors = np.array([[10, 13], [16, 30], [33, 23], [30, 61], [62, 45], [59, 119], [116, 90], [156, 198], [373, 326]])
# anchors needs scale once sample_size changes
__C.classes = 200  # any change made to this must also change yolo_head.py(yolo_head()) and make sure last layer have 3*(class + 5) or N*(class + 4 + 1) for N=3 in yolo v3 and 4 location cord and 1 class filters for each detector ()
__C.num = 9
__C.num_anchors_per_layer = 3

__C.batch_size = 1
__C.sample_size = 1600
__C.total_epoch = 10
__C.batch_per_epoch = 500000

__C.lr_thresh = [100, 600]
__C.lr_array = [1e-3, 1e-4, 1e-5]

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
