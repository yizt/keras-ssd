# -*- coding: utf-8 -*-
"""
 @File    : losses.py
 @Time    : 2019/11/14 下午5:00
 @Author  : yizuotian
 @Description    :
"""
import tensorflow as tf
from tensorflow.python.keras import layers


def cls_loss(predict_cls_logits, true_cls_ids, anchors_tag):
    """
    分类损失
    :param predict_cls_logits: 预测的anchors类别，[batch_size,num_anchors,num_classes]
    :param true_cls_ids:实际的anchors类别，[batch_size,num_anchors]
    :param anchors_tag: [batch_size,num_anchors]  1：正样本，-1：负样本，0: ignore
    :return:
    """
    indices = tf.where(tf.not_equal(anchors_tag, 0.))
    predict_cls_logits = tf.gather_nd(predict_cls_logits, indices)  # [N,num_classes]
    true_cls_ids = tf.gather_nd(true_cls_ids, indices)  # [N]
    # 转为onehot编码
    num_classes = tf.shape(predict_cls_logits)[-1]
    true_cls_ids = tf.one_hot(tf.cast(true_cls_ids, tf.int32), depth=num_classes)  # [N,num_classes]

    # 交叉熵损失函数
    losses = tf.nn.softmax_cross_entropy_with_logits_v2(
        labels=true_cls_ids, logits=predict_cls_logits)
    losses = tf.reduce_mean(losses)
    return losses


def smooth_l1_loss(y_true, y_predict, sigma=1.):
    """
    smooth L1损失函数；   0.5*sigma^2*x^2 if |x| <1/sigma^2 else |x|-0.5/sigma^2; x是 diff
    :param y_true:[N,4]
    :param y_predict:[N,4]
    :param sigma
    :return:
    """
    sigma_2 = sigma ** 2
    abs_diff = tf.abs(y_true - y_predict, name='abs_diff')
    loss = tf.where(tf.less(abs_diff, 1. / sigma_2), 0.5 * sigma_2 * tf.pow(abs_diff, 2), abs_diff - 0.5 / sigma_2)
    return loss


def regress_loss(predict_deltas, deltas, anchors_tag):
    """
    边框回归损失
    :param predict_deltas: 预测的回归目标，[batch_size, num_anchors, 4]
    :param deltas: 真实的回归目标，[batch_size, num_anchors, 4]
    :param anchors_tag: [batch_size,num_anchors]  1：正样本，-1：负样本，0: ignore
    :return:
    """
    # 去除padding和负样本
    indices = tf.where(tf.equal(anchors_tag, 1.))  # 正样本才计算回归目标
    predict_deltas = tf.gather_nd(predict_deltas, indices)  # [N,4]
    deltas = tf.gather_nd(deltas, indices)  # [N,4]

    # 考虑正样本为零的情况
    loss = tf.cond(tf.size(deltas) > 0,
                   true_fn=lambda: smooth_l1_loss(deltas, predict_deltas),
                   false_fn=lambda: tf.constant([0.0]))
    loss = tf.reduce_mean(loss)
    return loss


class MultiClsLoss(layers.Layer):
    def __init__(self, **kwargs):
        super(MultiClsLoss, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        """

        :param inputs:
        :param kwargs:
        :return:
        """
        predict_cls_logits_list, true_cls_ids_list, anchors_tag_list = inputs
        loss_list = []
        for predict_cls_logits, true_cls_ids, anchors_tag in zip(predict_cls_logits_list,
                                                                 true_cls_ids_list, anchors_tag_list):
            loss_list.append(cls_loss(predict_cls_logits, true_cls_ids, anchors_tag))
        loss = tf.concat(loss_list, axis=0)
        return tf.reduce_mean(loss)

    def compute_output_shape(self, input_shape):
        return ()


class MultiRegressLoss(layers.Layer):
    def __init__(self, **kwargs):
        super(MultiRegressLoss, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        predict_deltas_list, deltas_list, anchors_tag_list = inputs
        loss_list = []
        for predict_deltas, deltas, anchors_tag in zip(predict_deltas_list,
                                                       deltas_list, anchors_tag_list):
            loss_list.append(regress_loss(predict_deltas, deltas, anchors_tag))

        loss = tf.concat(loss_list, axis=0)
        return tf.reduce_mean(loss)

    def compute_output_shape(self, input_shape):
        return ()
