import xml.etree.ElementTree as ET
import numpy as np
import os
import tensorflow as tf
from PIL import Image
from config import getLabels
import glob
# sets = [('2007', 'trainval'), ('2012', 'trainval')]


classes = getLabels()
sample_size = 1600
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


def convert_annotation(image_id):
    in_file = open('C:\\Users\\P900\\Desktop\\myWork\\dmg_inspect_YOLOv3\\CID_project_dataset\\annotation\\%s.xml'%(image_id))

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

def convert_img(image_id):
    image = Image.open('C:\\Users\\P900\\Desktop\\myWork\\dmg_inspect_YOLOv3\\CID_project_dataset\\CID_Photo\\%s.jpg'%(image_id))
    resized_image = image.resize((sample_size, sample_size), Image.BICUBIC)
    image_data = np.array(resized_image, dtype='float32')/255
    img_raw = image_data.tobytes()
    return img_raw

filename = os.path.join('trainval'+'0712_'+str(sample_size)+'.tfrecords')
writer = tf.python_io.TFRecordWriter(filename)

os.chdir('C:\\Users\\P900\\Desktop\\myWork\\dmg_inspect_YOLOv3\\CID_project_dataset\\annotation')
image_ids = os.listdir('.')
image_ids = glob.glob(str(image_ids) + '*.xml')
# print(filename)
for image_id in image_ids:
    image_id = image_id.split('.')[0]
    xywhc = convert_annotation(image_id)
    img_raw = convert_img(image_id)

    example = tf.train.Example(features=tf.train.Features(feature={
        'xywhc':
                tf.train.Feature(float_list=tf.train.FloatList(value=xywhc)),
        'img':
                tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw])),
        }))
    writer.write(example.SerializeToString())
writer.close()
