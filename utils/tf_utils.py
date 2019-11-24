# -*- coding: utf-8 -*-
"""
 @File    : tf_utils.py
 @Time    : 2019/11/14 下午5:09
 @Author  : yizuotian
 @Description    :
"""
import tensorflow as tf


def remove_pad(input_tensor):
    """

    :param input_tensor:
    :return:
    """
    pad_tag = input_tensor[..., -1]
    real_size = tf.cast(tf.reduce_sum(pad_tag), tf.int32)
    return input_tensor[:real_size, :-1]


def pad_to_fixed_size(input_tensor, fixed_size):
    """
    增加padding到固定尺寸,在第二维增加一个标志位,0-padding,1-非padding
    :param input_tensor: 二维张量
    :param fixed_size:
    :return:
    """
    input_size = tf.shape(input_tensor)[0]
    x = tf.pad(input_tensor, [[0, 0], [0, 1]], mode='CONSTANT', constant_values=1)
    # padding
    padding_size = tf.maximum(0, fixed_size - input_size)
    x = tf.pad(x, [[0, padding_size], [0, 0]], mode='CONSTANT', constant_values=0)
    return x
