import tensorflow as tf
import numpy as np
import os
import glob
import xml.etree.ElementTree as ET
import random
from config import getLabels
from PIL import Image
from config import cfg, annotation_dir_deep, repo_dir
from numpy.random import permutation as perm
from pathlib import Path

def convert(size, box):
    dw = 1./size[0]
    dh = 1./size[1]
    x = (box[0] + box[1])/2.0
    y = (box[2] + box[3])/2.0
    w = abs(box[1] - box[0])
    h = abs(box[3] - box[2])
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return [x, y, w, h]


def convert_annotation(annotation_dir, image_id, folder):
    classes = getLabels()
    if annotation_dir_deep:
        in_file = open(annotation_dir + '\\%s\\%s.xml'%(folder, image_id))
    else:
        try:
            in_file = open(annotation_dir + '\\%s.xml'%(image_id))
        except Exception as e:
            print(e)
            return False

    tree = ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    try:
        w = int(size.find('width').text)
        h = int(size.find('height').text)
    except Exception as e:
        # print(e)
        return False
    if w == 0 or h == 0:
        print(image_id, '--w,h:', w, h)
        return False
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


def convert_img(image_dir, image_id, folder):
    if annotation_dir_deep:
        image = Image.open(image_dir + '\\%s\\%s'%(folder, folder) + '\\%s.%s'%(image_id, cfg.image_format))
    else:
        image = Image.open(image_dir + '\\%s.%s'%(image_id, cfg.image_format))
    resized_image = image.resize((cfg.sample_size, cfg.sample_size), Image.BICUBIC)
    image_data = np.array(resized_image, dtype='float32')/255
    # img_raw = image_data.tobytes()
    return image_data

def get_all_annotation (image_dir, annotation_dir):
    # get all image id
    os.chdir(annotation_dir)
    if annotation_dir_deep:
        image_ids = []
        image_folder = os.listdir('.')
        for folder in image_folder:
            img_file = Path(image_dir + "\\" + folder)
            if img_file.is_dir():
                os.chdir(annotation_dir + '\\' + folder)
                temp_list = os.listdir('.')
                temp_list = glob.glob(str(temp_list) + '*.xml')
                temp_list = [(item, folder) for item in temp_list]
                image_ids += temp_list
    else:
        image_ids = os.listdir('.')
        image_ids = glob.glob(str(image_ids) + '*.xml')

    image_ids = image_ids[:]
    # random.shuffle(image_ids)
    total = len(image_ids)
    batch_size = cfg.batch_size
    print('total_epoch:', cfg.total_epoch)
    print('batch_size', cfg.batch_size)
    print('image_ids length:', len(image_ids))
    os.chdir(repo_dir)
    return image_ids[:int(total*0.8)], image_ids[int(total*0.8):int(total*0.9)], image_ids[int(total*0.9):]

def shuffle(image_dir, annotation_dir, image_ids, total_epoch=1):

    # get batches
    batch_size = cfg.batch_size
    # batch_per_epoch = int(len(image_ids) / batch_size)
    for i in range(total_epoch):
        shuffle_idx = perm(np.arange(len(image_ids)))
        img_batch = None
        coord_batch = None
        k = 0
        for j in range(len(image_ids)):
            try:
                xywhc = None
                train_instance = image_ids[shuffle_idx[j]]
                if annotation_dir_deep:
                    image_id = train_instance[0].split('.xml')[0]
                    xywhc = convert_annotation(annotation_dir, image_id, train_instance[1])
                    coord = np.reshape(xywhc, [30, 5])
                    # print('imageID:{}, xywhc: {}'.format(image_id, xywhc))
                    image_data = convert_img(image_dir, image_id, train_instance[1])
                else:
                    image_id = train_instance.split('.xml')[0]
                    xywhc = convert_annotation(annotation_dir, image_id, None)
                    coord = np.reshape(xywhc, [30, 5])
                    # print('imageID:{}, xywhc: {}'.format(image_id, xywhc))
                    image_data = convert_img(image_dir, image_id, None)
                if not xywhc:
                    continue
                img = np.reshape(image_data, [cfg.sample_size, cfg.sample_size, 3])

                # data Aug
                # rnd = tf.less(tf.random_uniform(shape=[], minval=0, maxval=2), 1)
                #
                # # rnd is part of data Augmentation
                # def flip_img_coord(_img, _coord):
                #     zeros = tf.constant([[0, 0, 0, 0, 0]] * 30, tf.float32)
                #     img_flipped = tf.image.flip_left_right(_img)
                #     idx_invalid = tf.reduce_all(tf.equal(coord, 0), axis=-1)
                #     coord_temp = tf.concat([tf.minimum(tf.maximum(1 - _coord[:, :1], 0), 1),
                #                             _coord[:, 1:]], axis=-1)
                #     coord_flipped = tf.where(idx_invalid, zeros, coord_temp)
                #     return img_flipped, coord_flipped
                #
                # img, coord = tf.cond(rnd, lambda: (tf.identity(img), tf.identity(coord)),
                #                      lambda: flip_img_coord(img, coord))

                if coord is None: continue
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
                k += 1
            except Exception as e:
                pass
                # print('shuffle error', e)
                # print('image:{} has illegal shape'.format(image_id))
                # continue
            # coord_batch = np.concatenate(coord_batch, 0)
            # img_batch = np.concatenate(img_batch, 0)
            if k == batch_size:
                yield img_batch, coord_batch
                k = 0
                img_batch = None
                coord_batch = None

        # yield None, None