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
