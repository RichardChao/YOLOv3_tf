from yolo_top import yolov3
import numpy as np
import tensorflow as tf
from config import cfg
from PIL import Image, ImageDraw, ImageFont
from draw_boxes import draw_boxes
import cv2
import matplotlib.pyplot as plt
import os
import glob
# IMG_ID ='008957'
os.chdir('testset')
image_ids = os.listdir('.')
image_ids = glob.glob(str(image_ids) + '*.' + cfg.image_format)
os.chdir('..')
def do_predict(image_ids):
    image_datas = []
    for image_id in image_ids:
        image_test = Image.open('testset/' + image_id)
        resized_image = image_test.resize((cfg.sample_size, cfg.sample_size), Image.BICUBIC)
        image_data = np.array(resized_image, dtype='float32')
        image_datas.append((image_id, image_data, image_test))

    imgs_holder = tf.placeholder(tf.float32, shape=[1, cfg.sample_size, cfg.sample_size, 3])
    istraining = tf.constant(False, tf.bool)
    cfg.batch_size = 1
    cfg.scratch = True

    model = yolov3(imgs_holder, None, istraining)
    img_hw = tf.placeholder(dtype=tf.float32, shape=[2])
    boxes, scores, classes = model.pedict(img_hw, iou_threshold=0.5, score_threshold=0.001)

    saver = tf.train.Saver()
    ckpt_dir = 'ckpt/'

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
                cv2.imwrite("result/" + image_id, cv2.cvtColor(image_draw, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_PNG_COMPRESSION), 9])
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
