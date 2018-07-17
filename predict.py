from yolo_top import yolov3
import numpy as np
import tensorflow as tf
from config import cfg, ckpt_dir, testset, result_dir
from PIL import Image, ImageDraw, ImageFont
from draw_boxes import draw_boxes
import cv2
import matplotlib.pyplot as plt
import os
import glob
import xml.etree.cElementTree as ET

# IMG_ID ='008957'
repo_dir = str(os.getcwd())
os.chdir(testset)
image_ids = os.listdir('.')
image_ids = glob.glob(str(image_ids) + '*.' + cfg.image_format)
os.chdir(repo_dir)


def indent(elem, level=0):
    i = "\n" + level * "  "
    j = "\n" + (level - 1) * "  "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "  "
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
        for subelem in elem:
            indent(subelem, level + 1)
        if not elem.tail or not elem.tail.strip():
            elem.tail = j
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = j
    return elem


def draw_xml(image, boxes, box_classes, class_names, scores=None, image_id=''):
    image = Image.fromarray(np.floor(image * 255 + 0.5).astype('uint8'))
    root = ET.Element("annotation")

    folder = ET.SubElement(root, "folder")
    folder.text = ''

    filename = ET.SubElement(root, "filename")
    filename.text = str(image_id)

    source = ET.SubElement(root, "source")
    database = ET.SubElement(source, "database")
    database.text = 'ILSVRC_2014'

    size = ET.SubElement(root, "size")
    width = ET.SubElement(size, "width")
    width.text = str(image.size[0])
    height = ET.SubElement(size, "height")
    height.text = str(image.size[1])
    depth = ET.SubElement(size, "depth")
    depth.text = '3'

    ET.SubElement(root, "segmented").text = '0'
    for box, cls_idx in zip(boxes, box_classes):
        obj = ET.SubElement(root, "object")
        ET.SubElement(obj, "name").text = str(class_names[cls_idx])
        ET.SubElement(obj, "pose").text = 'Unspecified'
        ET.SubElement(obj, "truncated").text = '0'
        ET.SubElement(obj, "difficult").text = '0'

        bndbox = ET.SubElement(obj, "bndbox")

        top, left, bottom, right = box
        top = max(0, np.floor(top + 0.5).astype('int32'))
        left = max(0, np.floor(left + 0.5).astype('int32'))
        bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
        right = min(image.size[0], np.floor(right + 0.5).astype('int32'))

        ET.SubElement(bndbox, "xmax").text = str(right)
        ET.SubElement(bndbox, "xmin").text = str(left)
        ET.SubElement(bndbox, "ymax").text = str(top)
        ET.SubElement(bndbox, "ymin").text = str(bottom)
        print(str(class_names[cls_idx]), (left, top), (right, bottom))

    root = indent(root)
    tree = ET.ElementTree(root)
    return tree

def do_predict(image_ids):
    image_datas = []
    for image_id in image_ids[:]:
        image_test = Image.open(testset + '\\' + image_id)
        resized_image = image_test.resize((cfg.sample_size, cfg.sample_size), Image.BICUBIC)
        image_data = np.array(resized_image, dtype='float32')
        image_datas.append((image_id, image_data, image_test))

    imgs_holder = tf.placeholder(tf.float32, shape=[1, cfg.sample_size, cfg.sample_size, 3])
    istraining = tf.constant(False, tf.bool)
    cfg.batch_size = 1
    cfg.scratch = True

    model = yolov3(imgs_holder, None, istraining)
    img_hw = tf.placeholder(dtype=tf.float32, shape=[2])
    boxes, scores, classes = model.pedict(img_hw, iou_threshold=0.5, score_threshold=0.02)

    saver = tf.train.Saver()

    with tf.Session() as sess:
        ckpt = tf.train.get_checkpoint_state(ckpt_dir)
        saver.restore(sess, ckpt.model_checkpoint_path)
        gs = int(ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1])
        print('Restore batch: ', gs)
        for image_id, image_data, image_test in image_datas:
            boxes_, scores_, classes_ = sess.run([boxes, scores, classes],
                                                feed_dict={
                                                        img_hw: [image_test.size[1], image_test.size[0]],
                                                        imgs_holder: np.reshape(image_data / 255, [1, cfg.sample_size, cfg.sample_size, 3])})
            try:
                image_draw = draw_boxes(np.array(image_test, dtype=np.float32) / 255, boxes_, classes_, cfg.names, scores=scores_)
                # cv2.imshow("prediction.png", cv2.cvtColor(image_draw, cv2.COLOR_RGB2BGR))
                print('predict:', image_id)
                tree = draw_xml(np.array(image_test, dtype=np.float32) / 255, boxes_, classes_, cfg.names, scores=scores_, image_id=image_id)

                tree.write(result_dir + '\\predicted_xmls\\{}.xml'.format(image_id))
                cv2.imwrite(result_dir + '\\' + image_id, cv2.cvtColor(image_draw, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_PNG_COMPRESSION), 9])
                # fig = plt.figure(frameon=False)
                # ax = plt.Axes(fig, [0, 0, 1, 1])
                # ax.set_axis_off()
                # fig.add_axes(ax)
                # plt.imshow(image_draw, interpolation='none')
                # plt.savefig('C:\\Users\\P900\\Desktop\\myWork\\YOLOv3_tf\\prediction.jpg', dpi='figure', interpolation='none')
                # # fig.savefig('prediction.jpg')
                # plt.show()
            except Exception as e:
                print(e)


do_predict(image_ids)

cv2.waitKey(0)
