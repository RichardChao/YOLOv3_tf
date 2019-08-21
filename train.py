from yolo_top import yolov3
import numpy as np
import tensorflow as tf
from yolov3.data import shuffle,get_all_annotation
from config import cfg, image_dir, annotation_dir, ckpt_dir
import os
import pickle

# img, truth = shuffle(image_dir, annotation_dir)
# print(img,truth)
# tfconfig = tf.ConfigProto()
# tfconfig.gpu_options.allow_growth = True
# tfconfig.gpu_options.per_process_gpu_memory_fraction = 0.9

img_holder = tf.placeholder(tf.float32, shape=(cfg.batch_size, cfg.train.image_resized, cfg.train.image_resized, 3), name='img_holder')

truth_holder = tf.placeholder(tf.float32, shape=(cfg.batch_size, 30, 5), name='truth_holder')
istraining = tf.placeholder(tf.bool, shape=(), name='istraining')

model = yolov3(img_holder, truth_holder, istraining, trainable_head=True)

loss = model.compute_loss()

# optimizer
global_step = tf.Variable(0, trainable=False)
# lr = tf.train.exponential_decay(0.0001, global_step=global_step, decay_steps=2e4, decay_rate=0.1)
lr1 = tf.train.piecewise_constant(global_step, [2215*3, 2215*5], [5e-5, 1e-4, 1e-5])
# lr2 = tf.train.piecewise_constant(global_step, [100, 1000], [1e-5, 5e-6, 1e-6])

# optimizer1 = tf.train.AdamOptimizer(learning_rate=lr1)
optimizer2 = tf.train.AdamOptimizer(learning_rate=lr1)



# optimizer = tf.train.MomentumOptimizer(learning_rate=lr, momentum=0.9)
update_op = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
vars_det = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="Head")
# for var in vars_det:
#     print(var)
with tf.control_dependencies(update_op):
    # gvs1 = optimizer1.compute_gradients(loss, var_list=vars_det)
    # clip_grad_var1 = [gv if gv[0] is None else [
    #     tf.clip_by_norm(gv[0], 100.), gv[1]] for gv in gvs1]
    # train_op1 = optimizer1.apply_gradients(clip_grad_var1, global_step=global_step)
    train_op1 = None
    gvs2 = optimizer2.compute_gradients(loss)
    clip_grad_var2 = [gv if gv[0] is None else [
        tf.clip_by_norm(gv[0], 100.), gv[1]] for gv in gvs2]
    train_op2 = optimizer2.apply_gradients(clip_grad_var2, global_step=global_step)

    # train_op1 = optimizer1.minimize(loss, global_step=global_step, var_list=vars_det)
    # train_op2 = optimizer2.minimize(loss, global_step=global_step)
    # var_list=var_Feature_Extractor

saver = tf.train.Saver()

# data Aug, padding & notation conversion not added, 1600 only
image_resized = [416]
# , 1440, 1280
# , 1280, 960
# cfg.anchors = (image_resized[-1] / 416) * cfg.anchors
cfg.anchors = cfg.anchors.astype(int)
gs = 0

def main(size, train_op1, train_op2):
    cfg.train.image_resized = size
    cfg.sample_size = size
    # restore_flag = False
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        ckpt = tf.train.get_checkpoint_state(ckpt_dir)
        if (ckpt and ckpt.model_checkpoint_path):
            saver.restore(sess, ckpt.model_checkpoint_path)
            gs = int(ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1])
            sess.run(tf.assign(global_step, gs))
            print('Restore batch: ', sess.run(global_step))
            # restore_flag = True
        else:
            print('no checkpoint found')
            sess.run(tf.global_variables_initializer())

        # conv74, conv66, conv58 = model.show_head_layers()
        train_ids, eval_ids, test_ids = get_all_annotation(image_dir, annotation_dir)
        batch_per_epoch = int(len(train_ids) / cfg.batch_size)  # how many batch in each training epoch
        eval_size = int(len(eval_ids) / cfg.total_epoch)
        break_patience = 2
        print('batch_per_epoch', batch_per_epoch, 'eval_size', eval_size)
        def train (train_op, phase, epoches=cfg.total_epoch):
            loss_sum = 0
            loss_eval = []
            break_num = 0
            # sess.run(tf.assign(global_step, 0))
            for i, (img, truth) in enumerate(shuffle(image_dir, annotation_dir, train_ids, total_epoch=epoches)):
                gs = sess.run(global_step)
                _, loss_ = sess.run([train_op, loss], feed_dict={img_holder: img, truth_holder: truth, istraining: True})
                loss_sum += loss_
                # print(i, loss_)
                if gs % 10 == 0 and gs != 0:
                    temp_step = np.maximum(np.minimum(gs, 10), 1)
                    print('loss_{}:{}, loss_ave:{}'.format(gs, loss_/cfg.batch_size, loss_sum / cfg.batch_size / temp_step))
                    loss_sum = 0
                if gs % batch_per_epoch == 0 and gs != 0 and (gs / batch_per_epoch >= 5 or phase == 'op2'):
                    # file = '{}-{}{}'
                    # profile = file.format('yolov3', global_step, '.profile')
                    # profile = os.path.join(ckpt_dir, profile)
                    # with open(profile, 'wb') as profile_ckpt:
                    #     pickle.dump(loss_profile, profile_ckpt)
                    print('save ckpt at step:', gs)
                    saver.save(sess, ckpt_dir + 'yolov3.ckpt', global_step=global_step, write_meta_graph=False)
    
                if gs % batch_per_epoch == 1 and (gs / batch_per_epoch >= 1):
                    loss_eval.append(0)
                    eval_order = len(loss_eval) - 1
                    eval_batch_num = 0
                    for gs, (img, truth) in enumerate(
                        shuffle(image_dir, annotation_dir, eval_ids[eval_size*eval_order: eval_size*(eval_order+1)])):
                        _eval, loss_eval_sub = sess.run([train_op, loss],
                                            feed_dict={img_holder: img, truth_holder: truth, istraining: True})
                        eval_batch_num += 1
                        loss_eval[eval_order] += loss_eval_sub
                    if eval_batch_num > 0:
                        loss_eval[eval_order] = loss_eval[eval_order] / (eval_batch_num*cfg.batch_size)
                    else:
                        print('eval_batch_num:', eval_batch_num)
                    print('eval at epoch {}, loss_ave:{}'.format(eval_order, loss_eval[eval_order]))
                    if eval_order > 2 and loss_eval[eval_order] > loss_eval[eval_order-1] and loss_eval[eval_order] > loss_eval[eval_order-2] and loss_eval[eval_order] > loss_eval[eval_order-3]:
                        break_num += 1
                        if break_num > break_patience:
                            print('break training', loss_eval)
                            break
                    elif eval_order > 2:
                        break_num = 0
        # if not restore_flag:
        # print('train_op1 started')
        # train(train_op1, 'op1', 20)
        print('train_op2 started')
        train(train_op2, 'op2', cfg.total_epoch)
        test_batch_num = 0
        loss_test = 0
        for gs, (img, truth) in enumerate(
                shuffle(image_dir, annotation_dir, test_ids)):
            _test, loss_test_sub = sess.run([train_op2, loss],
                                            feed_dict={img_holder: img, truth_holder: truth, istraining: True})
            test_batch_num += 1
            loss_test += loss_test_sub
        if test_batch_num > 0:
            print('test  loss_ave:{}'.format(loss_test/(test_batch_num*cfg.batch_size)))
        else:
            print('test_batch_num:', test_batch_num)

    return 'Fin'


for size in image_resized:
    main(size, train_op1, train_op2)
