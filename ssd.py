# -*- coding: utf-8 -*-
"""
 @File    : ssd.py
 @Time    : 2019/11/14 下午4:30
 @Author  : yizuotian
 @Description    :
"""
from tensorflow.python.keras import backend, layers, Model
from utils.anchor import generate_anchors, FeatureSpec
from layers.target import SSDTarget
from layers.losses import MultiRegressLoss, MultiClsLoss
from utils.anchor import FeatureSpec
from typing import List


def seperable_conv2d(x, filters, name, kernel_size=1, stride=1, padding='same'):
    """

    :param x:
    :param filters:
    :param name:
    :param kernel_size:
    :param stride:
    :param padding:
    :return:
    """
    prefix = name
    channel_axis = 1 if backend.image_data_format() == 'channels_first' else -1
    x = layers.DepthwiseConv2D(kernel_size=kernel_size,
                               strides=stride,
                               activation=None,
                               use_bias=False,
                               padding=padding,
                               name=prefix + 'depthwise')(x)
    x = layers.BatchNormalization(axis=channel_axis,
                                  epsilon=1e-3,
                                  momentum=0.999,
                                  name=prefix + 'depthwise_BN')(x)

    x = layers.ReLU(6., name=prefix + 'depthwise_relu')(x)

    # Project
    x = layers.Conv2D(filters,
                      kernel_size=1,
                      padding='same',
                      use_bias=False,
                      activation=None,
                      name=prefix + 'project')(x)
    return x


def cls_headers(feature_list, num_anchors_list, num_classes):
    headers = []
    for i, (feature, num_anchors) in enumerate(zip(feature_list, num_anchors_list)):
        header = seperable_conv2d(feature, num_anchors * num_classes,
                                  'cls_header_{}'.format(i), kernel_size=3)
        # 打平
        header = layers.Reshape(target_shape=(-1, num_classes),
                                name='cls_header_flatten_{}'.format(i))(header)
        headers.append(header)

    # 拼接所有header
    # headers = layers.Concatenate(axis=0, name='cls_header_concat')(headers)
    return headers


def rgr_headers(feature_list, num_anchors_list):
    headers = []
    for i, (feature, num_anchors) in enumerate(zip(feature_list, num_anchors_list)):
        header = seperable_conv2d(feature, num_anchors * 4, 'rgr_header_{}'.format(i), kernel_size=3)
        # 打平
        header = layers.Reshape(target_shape=(-1, 4),
                                name='rgr_header_flatten_{}'.format(i))(header)
        headers.append(header)
    # 拼接所有header
    # headers = layers.Concatenate(axis=0, name='rgr_header_concat')(headers)
    return headers


def ssd_model(feature_fn, input_shape, num_classes, specs: List[FeatureSpec],
              max_gt_num, positive_iou_threshold, negative_iou_threshold, stage='train'):
    image_input = layers.Input(shape=input_shape)

    anchors_list = generate_anchors(specs)
    feature_list = feature_fn(image_input)

    num_anchors_list = [len(spec.aspect_ratios) + 2 for spec in specs]
    predict_logits_list = cls_headers(feature_list, num_anchors_list, num_classes)
    predict_deltas_list = rgr_headers(feature_list, num_anchors_list)

    if stage == 'train':
        gt_boxes = layers.Input(shape=(max_gt_num, 5), dtype='float32')
        gt_class_ids = layers.Input(shape=(max_gt_num, 2), dtype='int32')
        # 分类和回归目标
        deltas_list = [''] * len(anchors_list)
        labels_list = [''] * len(anchors_list)
        anchors_tag_list = [''] * len(anchors_list)
        for i, anchors in enumerate(anchors_list):
            target = SSDTarget(anchors, positive_iou_threshold, negative_iou_threshold,
                               name='target_{}'.format(i))
            deltas_list[i], labels_list[i], anchors_tag_list[i] = target([gt_boxes, gt_class_ids])

        # 计算loss
        cls_loss = MultiClsLoss(name='multi_cls_loss')([predict_logits_list,
                                                        labels_list, anchors_tag_list])
        rgr_loss = MultiRegressLoss(name='multi_rgr_loss')([predict_deltas_list,
                                                            deltas_list, anchors_tag_list])
        m = Model([image_input, gt_boxes, gt_class_ids], [cls_loss, rgr_loss])

    return m


if __name__ == '__main__':
    from base_net.mobilenet import mobilenet_v2_features
    from config import cfg

    model = ssd_model(mobilenet_v2_features, (300, 300, 3), 3,
                      cfg.specs, cfg.max_gt_num, cfg.positive_iou_threshold,
                      cfg.negative_iou_threshold)
    model.summary()
