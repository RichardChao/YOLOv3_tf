import tensorflow as tf
import numpy as np
import os
import glob
import xml.etree.ElementTree as ET
from config import getLabels
import Image
from config import cfg
from numpy.random import permutation as perm


classes = getLabels()
def convert(size, box):
    dw = 1./size[0]
    dh = 1./size[1]
    x = (box[0] + box[1])/2.0
    y = (box[2] + box[3])/2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return [x, y, w, h]


def convert_annotation(annotation_dir, image_id):
    in_file = open(annotation_dir + '\\%s.xml'%(image_id))

    tree = ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)
    bboxes = []
    for i, obj in enumerate(root.iter('object')):
        if i > 29:
            break
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult) == 1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
        bb = convert((w, h), b) + [cls_id]
        bboxes.extend(bb)
    if len(bboxes) < 30*5:
        bboxes = bboxes + [0, 0, 0, 0, 0]*(30-int(len(bboxes)/5))

    return np.array(bboxes, dtype=np.float32).flatten().tolist()

def convert_img(image_dir, image_id):
    image = Image.open(image_dir + '\\%s.jpg'%(image_id))
    resized_image = image.resize((cfg.sample_size, cfg.sample_size), Image.BICUBIC)
    image_data = np.array(resized_image, dtype='float32')/255
    # img_raw = image_data.tobytes()
    return image_data
def shuffle(image_dir, annotation_dir):
    # get all image id
    os.chdir(annotation_dir)
    image_ids = os.listdir('.')
    image_ids = glob.glob(str(image_ids) + '*.xml')
    # get batches
    batch_per_epoch = cfg.batch_per_epoch
    batch_size = cfg.batch_size
    for i in range(cfg.total_epoch):
        shuffle_idx = perm(np.arange(len(image_ids)))
        for b in range(batch_per_epoch):
            img_batch = None
            coord_batch = None
            for j in range(b * batch_size, b * batch_size + batch_size):
                train_instance = image_ids[shuffle_idx[j]]
                image_id = train_instance.split('.')[0]
                xywhc = convert_annotation(annotation_dir, image_id)
                coord = np.reshape(xywhc, [30, 5])

                image_data = convert_img(image_dir, image_id)
                img = np.reshape(image_data, [cfg.sample_size, cfg.sample_size, 3])
                # for data Augmentation
                # img = tf.image.resize_images(img, [cfg.train.image_resized, cfg.train.image_resized])

                if coord is None: continue
                try:
                    # coord_batch += [np.expand_dims(coord, 0)]
                    # img_batch += [np.expand_dims(img, 0)]
                    # tensor_a = tf.expand_dims(coord, 0)
                    # tensor_b = tf.expand_dims(img, 0)
                    if img_batch is None:
                        img_batch = np.expand_dims(img, 0)
                    else:
                        img_batch = np.concatenate([img_batch, np.expand_dims(img, 0)], 0)

                    if coord_batch is None:
                        coord_batch = np.expand_dims(coord, 0)
                    else:
                        coord_batch = np.concatenate([coord_batch, np.expand_dims(coord, 0)], 0)
                except:
                    print('err expand_dims')
            # coord_batch = np.concatenate(coord_batch, 0)
            # img_batch = np.concatenate(img_batch, 0)
            yield img_batch, coord_batch