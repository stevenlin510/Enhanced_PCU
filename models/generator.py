# -*- coding: utf-8 -*-
# @Description :
# @Author      : Guocheng Qian
# @Email       : guocheng.qian@kaust.edu.sa
# Modified by Wei-Cheng Lin

import tensorflow as tf
from functions import ops
from tf_ops.sampling.tf_sampling import gather_point, farthest_point_sample
from functions.pointnet_util import pointnet_sa_module, pointnet_fp_module

class PUGCN(object):
    """
    PU-GCN: Point Cloud models using Graph Convolutional Networks
    https://arxiv.org/abs/1912.03264.pdf
    """

    def __init__(self, opts, is_training, name="Generator"):
        self.opts = opts
        self.is_training = is_training
        self.name = name
        self.reuse = False
        self.num_point = self.opts.patch_num_point
        self.up_ratio = self.opts.up_ratio
        self.up_ratio_real = self.up_ratio + self.opts.more_up
        self.out_num_point = int(self.num_point * self.up_ratio)

    def __call__(self, inputs):
        with tf.variable_scope(self.name, reuse=self.reuse):
            features, idx = ops.feature_extractor(inputs,
                                                  self.opts.block, self.opts.n_blocks,
                                                  self.opts.channels, self.opts.k, self.opts.d,
                                                  use_global_pooling=self.opts.use_global_pooling,
                                                  scope='feature_extraction', is_training=self.is_training,
                                                  bn_decay=None)

            H = ops.up_unit(features, self.up_ratio_real,
                            self.opts.upsampler,
                            k=self.opts.k,
                            idx=idx,
                            scope="up_block",
                            use_att=self.opts.use_att,
                            is_training=self.is_training, bn_decay=None)

            coord = ops.conv2d(H, 32, [1, 1],
                               padding='VALID', stride=[1, 1],
                               bn=False, is_training=self.is_training,
                               scope='fc_layer1', bn_decay=None,
                               activation_fn=tf.nn.leaky_relu
                               )

            coord = ops.conv2d(coord, 3, [1, 1],
                               padding='VALID', stride=[1, 1],
                               bn=False, is_training=self.is_training,
                               scope='fc_layer2', bn_decay=None,
                               activation_fn=None, weight_decay=0.0)
            outputs = tf.squeeze(coord, [2])

            if self.up_ratio_real > self.up_ratio:
                outputs = gather_point(outputs, farthest_point_sample(self.out_num_point, outputs))
            outputs += tf.reshape(tf.tile(tf.expand_dims(inputs, 2), [1, 1, self.up_ratio, 1]),
                                  [inputs.shape[0], self.num_point * self.up_ratio, -1])  # B, N, 4, 3

        self.reuse = True
        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.name)
        return outputs

class Enhanced_PCU(object):

    def __init__(self, opts, is_training, name="Generator"):
        self.opts = opts
        self.is_training = is_training
        self.name = name
        self.reuse = False
        self.num_point = self.opts.patch_num_point
        self.up_ratio = self.opts.up_ratio
        self.up_ratio_real = self.up_ratio + self.opts.more_up
        self.out_num_point = int(self.num_point * self.up_ratio)

    def __call__(self, inputs):
        with tf.variable_scope(self.name, reuse=self.reuse):

            features, idx = ops.feature_extractor(inputs,
                                                  self.opts.block, self.opts.n_blocks,
                                                  self.opts.channels, self.opts.k, self.opts.d,
                                                  use_global_pooling=self.opts.use_global_pooling,
                                                  scope='feature_extraction', is_training=self.is_training,
                                                  bn_decay=None)

            global_feature = ops.global_feature_extractor(inputs, self.up_ratio, scope='global_feature_extractor'
                                    )

            upsampled_feature = ops.multi_branch_upsampling(features, self.up_ratio_real,
                            scope="mbu",
                            use_att=self.opts.use_att,
                            is_training=self.is_training, bn_decay=None)

            final_feature = ops.attention_fusion(upsampled_feature, global_feature, scope='attention_fusion')

            coord = ops.conv2d(final_feature, 32, [1, 1],
                               padding='VALID', stride=[1, 1],
                               bn=False, is_training=self.is_training,
                               scope='fc_layer1', bn_decay=None,
                               activation_fn=tf.nn.leaky_relu
                               )

            coord = ops.conv2d(coord, 3, [1, 1],
                               padding='VALID', stride=[1, 1],
                               bn=False, is_training=self.is_training,
                               scope='fc_layer2', bn_decay=None,
                               activation_fn=None, weight_decay=0.0)
            outputs = tf.squeeze(coord, [2])

            if self.up_ratio_real > self.up_ratio:
                outputs = gather_point(outputs, farthest_point_sample(self.out_num_point, outputs))
            outputs += tf.reshape(tf.tile(tf.expand_dims(inputs, 2), [1, 1, self.up_ratio, 1]),
                                  [inputs.shape[0], self.num_point * self.up_ratio, -1])  # B, N, 4, 3

        self.reuse = True
        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.name)
        return outputs

