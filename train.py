from yolo_top import yolov3
import numpy as np
import tensorflow as tf
from yolov3.data import shuffle
from config import cfg, image_dir, annotation_dir, ckpt_dir
import os
import pickle

# img, truth = shuffle(image_dir, annotation_dir)
# print(img,truth)
img_holder = tf.placeholder(tf.float32, shape=(cfg.batch_size, cfg.train.image_resized, cfg.train.image_resized, 3), name='img_holder')

truth_holder = tf.placeholder(tf.float32, shape=(cfg.batch_size, 30, 5), name='truth_holder')
istraining = tf.placeholder(tf.bool, shape=(), name='istraining')

model = yolov3(img_holder, truth_holder, istraining)

loss = model.compute_loss()

# optimizer
global_step = tf.Variable(0, trainable=False)
# lr = tf.train.exponential_decay(0.0001, global_step=global_step, decay_steps=2e4, decay_rate=0.1)
lr = tf.train.piecewise_constant(global_step, cfg.lr_thresh, cfg.lr_array)
optimizer = tf.train.AdamOptimizer(learning_rate=lr)
# optimizer = tf.train.MomentumOptimizer(learning_rate=lr, momentum=0.9)
update_op = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
vars_det = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="Head")
# for var in vars_det:
#     print(var)
with tf.control_dependencies(update_op):
    train_op = optimizer.minimize(loss, global_step=global_step, var_list=vars_det)
saver = tf.train.Saver()

# data Aug, padding & notation conversion not added, 1600 only
image_resized = [1600]
# , 1440, 1280
# , 1280, 960
cfg.anchors = (1600 / 416) * cfg.anchors
cfg.anchors = cfg.anchors.astype(int)
gs = 0

def train(size):
    cfg.train.image_resized = size
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        ckpt = tf.train.get_checkpoint_state(ckpt_dir)
        if (ckpt and ckpt.model_checkpoint_path):
            saver.restore(sess, ckpt.model_checkpoint_path)
            gs = int(ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1])
            sess.run(tf.assign(global_step, gs))
            print('Restore batch: ', gs)
        else:
            print('no checkpoint found')
            sess.run(tf.global_variables_initializer())
        loss_sum = 0
        for i, (img, truth) in enumerate(shuffle(image_dir, annotation_dir)):
            _, loss_ = sess.run([train_op, loss], feed_dict={img_holder: img, truth_holder: truth, istraining: True})
            loss_sum += loss_
            if i % 10 == 0:
                temp_step = np.maximum(np.minimum(i, 10), 1)
                print('loss_{}:{}, loss_ave:{}'.format(i, loss_, loss_sum / temp_step))
                loss_sum = 0
            if i % 100 == 0 and i != 0:
                # file = '{}-{}{}'
                # profile = file.format('yolov3', global_step, '.profile')
                # profile = os.path.join(ckpt_dir, profile)
                # with open(profile, 'wb') as profile_ckpt:
                #     pickle.dump(loss_profile, profile_ckpt)
                print('save ckpt at step:', i)
                saver.save(sess, ckpt_dir + 'yolov3.ckpt', global_step=global_step, write_meta_graph=False)


for size in image_resized:
    train(size)
