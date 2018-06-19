from yolo_top import yolov3
import numpy as np
import tensorflow as tf
from yolov3.data import shuffle
from config import cfg
import os
import pickle


image_dir = 'C:\\Users\\P900\\Desktop\\myWork\\dmg_inspect_YOLOv3\\CID_project_dataset\\CID_Photo'
annotation_dir = 'C:\\Users\\P900\\Desktop\\myWork\\dmg_inspect_YOLOv3\\CID_project_dataset\\annotation'
# img, truth = shuffle(image_dir, annotation_dir)
# print(img,truth)
img_holder = tf.placeholder(tf.float32, shape=(cfg.batch_size, cfg.sample_size, cfg.sample_size, 3), name='img_holder')

truth_holder = tf.placeholder(tf.float32, shape=(cfg.batch_size, 30, 5), name='truth_holder')
istraining = tf.placeholder(tf.bool, shape=(), name='istraining')

model = yolov3(img_holder, truth_holder, istraining)

loss = model.compute_loss()

# optimizer
global_step = tf.Variable(0, trainable=False)
# lr = tf.train.exponential_decay(0.0001, global_step=global_step, decay_steps=2e4, decay_rate=0.1)
lr = tf.train.piecewise_constant(global_step, [100, 200], [1e-3, 1e-4, 1e-5])
optimizer = tf.train.AdamOptimizer(learning_rate=lr)
# optimizer = tf.train.MomentumOptimizer(learning_rate=lr, momentum=0.9)
update_op = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
vars_det = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="Head")
# for var in vars_det:
#     print(var)
with tf.control_dependencies(update_op):
    train_op = optimizer.minimize(loss, global_step=global_step, var_list=vars_det)
saver = tf.train.Saver()
ckpt_dir = 'C:\\Users\\P900\\Desktop\\myWork\\YOLOv3_tf\\ckpt\\'



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
    for i, (img, truth) in enumerate(shuffle(image_dir, annotation_dir)):
        _, loss_ = sess.run([train_op, loss], feed_dict={img_holder: img, truth_holder: truth, istraining: True})
        print('loss_{}:'.format(i), loss_)
        if (i % 100 == 0 and i != 0):
            # file = '{}-{}{}'
            # profile = file.format('yolov3', global_step, '.profile')
            # profile = os.path.join(ckpt_dir, profile)
            # with open(profile, 'wb') as profile_ckpt:
            #     pickle.dump(loss_profile, profile_ckpt)
            print('save ckpt at step:', i)
            saver.save(sess, ckpt_dir + 'yolov3.ckpt', global_step=global_step, write_meta_graph=False)
