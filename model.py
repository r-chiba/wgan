from __future__ import print_function
from __future__ import division

import os
import time
import math
import glob
import numpy as np
import tensorflow as tf
import cv2

from utils import *

class WGAN:
    def __init__(self, sess, flags):
        self.sess = sess
        self.batch_size = flags.batch_size
        self.input_height = flags.input_height
        self.input_width = flags.input_width
        self.n_channel = flags.n_channel
        self.output_height = flags.output_height
        self.output_width = flags.output_width
        self.z_dim = flags.z_dim
        self.g_dim = flags.g_dim
        self.d_dim = flags.d_dim
        self.learning_rate = flags.learning_rate
        self.savedir = flags.save_dir
        self.training = flags.train
        self.n_epoch = flags.n_epoch
        self.n_critic = flags.n_critic
        self.clip = flags.clip

    def generator(self, z, reuse=False, training=True):
        h, w = self.output_height, self.output_width
        h2, w2 = pad_out_size_same(h, 2), pad_out_size_same(w, 2)
        h4, w4 = pad_out_size_same(h2, 2), pad_out_size_same(w2, 2)
        h8, w8 = pad_out_size_same(h4, 2), pad_out_size_same(w4, 2)
        h16, w16 = pad_out_size_same(h8, 2), pad_out_size_same(w8, 2)

        with tf.variable_scope('generator') as scope:
            if reuse == True:
                scope.reuse_variables()

            hid0, self.h0_w, self.h0_b = linear(z, self.g_dim*8*h16*w16, 'g_h0', with_w=True)
            hid0 = tf.nn.relu(hid0)
            hid0 = tf.reshape(hid0, [self.batch_size, h16, w16, self.g_dim*8])

            hid1, self.h1_w, self.h1_b = deconv2d(hid0, [self.batch_size, h8, w8, self.g_dim*4], 
                sth=2, stw=2, name='g_h1', with_w=True)
            hid1 = tf.contrib.layers.batch_norm(hid1, decay=0.9,
                updates_collections=None, epsilon=1e-5, scale=True, is_training=training)
            hid1 = tf.nn.relu(hid1)

            hid2, self.h2_w, self.h2_b = deconv2d(hid1, [self.batch_size, h4, w4, self.g_dim*2], 
                sth=2, stw=2, name='g_h2', with_w=True)
            hid2 = tf.contrib.layers.batch_norm(hid2, decay=0.9,
                updates_collections=None, epsilon=1e-5, scale=True, is_training=training)
            hid2 = tf.nn.relu(hid2)

            hid3, self.h3_w, self.h3_b = deconv2d(hid2, [self.batch_size, h2, w2, self.g_dim], 
                sth=2, stw=2, name='g_h3', with_w=True)
            hid3 = tf.contrib.layers.batch_norm(hid3, decay=0.9,
                updates_collections=None, epsilon=1e-5, scale=True, is_training=training)
            hid3 = tf.nn.relu(hid3)

            hid4, self.h4_w, self.h4_b = deconv2d(hid3, [self.batch_size, h, w, self.n_channel], 
                sth=2, stw=2, name='g_h4', with_w=True)

            if reuse == True:
                tf.histogram_summary('g_h0_w', self.h0_w)
                tf.histogram_summary('g_h0_b', self.h0_b)
                tf.histogram_summary('g_h1_w', self.h1_w)
                tf.histogram_summary('g_h1_b', self.h1_b)
                tf.histogram_summary('g_h2_w', self.h2_w)
                tf.histogram_summary('g_h2_b', self.h2_b)
                tf.histogram_summary('g_h3_w', self.h3_w)
                tf.histogram_summary('g_h3_b', self.h3_b)
                tf.histogram_summary('g_h4_w', self.h4_w)
                tf.histogram_summary('g_h4_b', self.h4_b)

            return tf.nn.tanh(hid4)

    def critic(self, x, reuse=False):
        with tf.variable_scope('critic') as scope:
            if reuse == True:
                scope.reuse_variables()

            h0, self.h0_w, self.h0_b= conv2d(x, self.d_dim, sth=2, stw=2, 
                name='c_h0', with_w=True)
            #h0 = tf.contrib.layers.batch_norm(h0, decay=0.9,
            #    updates_collections=None, epsilon=1e-5, scale=True, is_training=self.training)
            h0 = lrelu(h0)

            h1, self.h1_w, self.h1_b= conv2d(h0, self.d_dim*2, sth=2, stw=2, 
                name='c_h1', with_w=True)
            #h1 = tf.contrib.layers.batch_norm(h1, decay=0.9,
            #    updates_collections=None, epsilon=1e-5, scale=True, is_training=self.training)
            h1 = lrelu(h1)

            h2, self.h2_w, self.h2_b= conv2d(h1, self.d_dim*4, sth=2, stw=2,
                name='c_h2', with_w=True)
            #h2 = tf.contrib.layers.batch_norm(h2, decay=0.9,
            #    updates_collections=None, epsilon=1e-5, scale=True, is_training=self.training)
            h2 = lrelu(h2)
            
            h3, self.h3_w, self.h3_b= conv2d(h2, self.d_dim*8, sth=2, stw=2, 
                name='c_h3', with_w=True)
            #h3 = tf.contrib.layers.batch_norm(h3, decay=0.9,
            #    updates_collections=None, epsilon=1e-5, scale=True, is_training=self.training)
            h3 = lrelu(h3)

            h4,self.h4_w, self.h4_b = linear(tf.reshape(h3, [self.batch_size, -1]),
                1, 'c_h4', with_w=True)

            if reuse == False:
                tf.histogram_summary('c_h0_w', self.h0_w)
                tf.histogram_summary('c_h0_b', self.h0_b)
                tf.histogram_summary('c_h1_w', self.h1_w)
                tf.histogram_summary('c_h1_b', self.h1_b)
                tf.histogram_summary('c_h2_w', self.h2_w)
                tf.histogram_summary('c_h2_b', self.h2_b)
                tf.histogram_summary('c_h3_w', self.h3_w)
                tf.histogram_summary('c_h3_b', self.h3_b)
                tf.histogram_summary('c_h4_w', self.h4_w)
                tf.histogram_summary('c_h4_b', self.h4_b)

            return h4

    def build_model(self):
        self.z = tf.placeholder(tf.float32, [None, self.z_dim], name='z')
        self.x_real = tf.placeholder(tf.float32, 
            [self.batch_size, self.input_height, self.input_width, self.n_channel], name='x')
        self.x_fake = self.generator(self.z)
        self.x_sample = self.generator(self.z, reuse=True, training=False)
        self.c_real = self.critic(self.x_real)
        self.c_fake = self.critic(self.x_fake, reuse=True)

        #self.c_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        #    logits=self.d_real, targets=tf.ones_like(self.d_real)))
        #self.c_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        #    logits=self.d_fake, targets=tf.zeros_like(self.d_fake)))
        #self.c_loss = self.d_loss_real + self.d_loss_fake
        #self.g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        #    logits=self.d_fake, targets=tf.ones_like(self.d_fake)))
        self.c_loss_real = tf.reduce_mean(self.c_real)
        self.c_loss_fake = tf.reduce_mean(self.c_fake)
        self.c_loss = self.c_loss_fake - self.c_loss_real
        self.g_loss = -tf.reduce_mean(self.c_fake)

        #self.c_vars = [x for x in tf.trainable_variables() if 'c_' in x.name]
        #self.g_vars = [x for x in tf.trainable_variables() if 'g_' in x.name]

        self.c_optimizer = tf.train.RMSPropOptimizer(self.learning_rate)
        self.g_optimizer = tf.train.RMSPropOptimizer(self.learning_rate)

        c_grads_and_vars = self.c_optimizer.compute_gradients(self.c_loss)
        g_grads_and_vars = self.g_optimizer.compute_gradients(self.g_loss)
        c_grads_and_vars = [[grad, var] for grad, var in c_grads_and_vars
            if grad is not None and var.name.startswith('c')]
        g_grads_and_vars = [[grad, var] for grad, var in g_grads_and_vars
            if grad is not None and var.name.startswith('g')]

        #for _, var in c_grads_and_vars:
        #    print(var.name)
        #print('-----')
        #for _, var in g_grads_and_vars:
        #    print(var.name)
        #print('-----')
        
        self.c_train_op = self.c_optimizer.apply_gradients(c_grads_and_vars)
        self.g_train_op = self.g_optimizer.apply_gradients(g_grads_and_vars)

        #critic_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='critic')
        #for var in critic_variables:
        #    print(var.name)
        self.w_clip = [var.assign(tf.clip_by_value(var, -self.clip, self.clip))
            for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='critic')]

        #tf.scalar_summary('c_loss_real', self.c_loss_real)
        #tf.scalar_summary('c_loss_fake', self.c_loss_fake)
        tf.scalar_summary('c_loss', -self.c_loss)
        tf.scalar_summary('g_loss', self.g_loss)

        self.saver = tf.train.Saver()
        self.summary = tf.merge_all_summaries()
        self.writer = tf.train.SummaryWriter(self.savedir, self.sess.graph)

    def train(self, batch_generator):
        def tile_image(imgs):
            d = int(math.sqrt(imgs.shape[0]-1))+1
            h = imgs[0].shape[0]
            w = imgs[0].shape[1]
            r = np.zeros((h*d, w*d, 3), dtype=np.float32)
            for idx, img in enumerate(imgs):
                idx_y = int(idx/d)
                idx_x = idx-idx_y*d
                r[idx_y*h:(idx_y+1)*h, idx_x*w:(idx_x+1)*w, :] = img
            return r

        self.sess.run(tf.initialize_all_variables())

        step = 0
        epoch = 1
        start = time.time()
        while epoch <= self.n_epoch:


            #_, g_loss = self.sess.run([self.g_optimizer, self.g_loss], 
            #    feed_dict={self.z: batch_z})
            #_, d_loss, x_fake, x_real, summary = self.sess.run(
            #    [self.d_optimizer, self.d_loss, self.x_fake, self.x_real, self.summary],
            #    feed_dict={self.z: batch_z, self.x_real: batch_images})
            for i in xrange(self.n_critic):
                batch_images, _, last_batch = batch_generator()
                batch_z = np.random.uniform(-1., 1., [self.batch_size, self.z_dim]).astype(np.float32)
                _, c_loss, summary = self.sess.run([self.c_train_op, self.c_loss, self.summary],
                    feed_dict={self.z: batch_z, self.x_real: batch_images})
                _ = self.sess.run(self.w_clip)

            batch_z = np.random.uniform(-1., 1., [self.batch_size, self.z_dim]).astype(np.float32)
            _, g_loss = self.sess.run([self.g_train_op, self.g_loss], feed_dict={self.z: batch_z})

            if step > 0 and step % 10 == 0:
                self.writer.add_summary(summary, step)

            if step % 10 == 0:
                elapsed = time.time() - start
                print("epoch %3d(%6d): loss(C)=%.4e, loss(G)=%.4e; time/step=%.2f sec" %
                        (epoch, step, c_loss, g_loss, elapsed if step == 0 else elapsed / 10))
                start = time.time()

            if step % 100 == 0:
                img_real = tile_image(batch_images) * 255. + 127.5
                img_real = cv2.cvtColor(img_real, cv2.COLOR_RGB2BGR)

                z = np.random.uniform(-1, 1, [self.batch_size, self.z_dim])
                img_fake = self.sess.run(self.x_sample, feed_dict={self.z: z})
                img_fake = tile_image(img_fake) * 255. + 127.5
                img_fake = cv2.cvtColor(img_fake, cv2.COLOR_RGB2BGR)
                cv2.imwrite(os.path.join(self.savedir, "images", "img_%d_real.png" % step), 
                    img_real)
                cv2.imwrite(os.path.join(self.savedir, "images", "img_%d_fake1.png" % step), 
                    img_fake)

            step += 1
            if last_batch == True: epoch += 1
