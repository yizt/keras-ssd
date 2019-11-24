# -*- coding: utf-8 -*-
"""
 @File    : ssd.py
 @Time    : 2019/11/14 下午4:30
 @Author  : yizuotian
 @Description    :
"""
import tensorflow as tf
from tensorflow.python.keras import backend, layers, Model
from utils.anchor import generate_anchors, FeatureSpec
from layers.target import SSDTarget
from layers.losses import regress_loss, cls_loss
from layers.detect_boxes import DetectBox
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
    headers = layers.Concatenate(axis=1, name='cls_header_concat')(headers)  # [B,num_anchors,num_classes]
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
    headers = layers.Concatenate(axis=1, name='rgr_header_concat')(headers)  # [B,num_anchors,4]
    return headers


def ssd_model(feature_fn, input_shape, num_classes, specs: List[FeatureSpec],
              max_gt_num, positive_iou_threshold, negative_iou_threshold,
              score_threshold=0.5, iou_threshold=0.3, max_detections_per_class=100,
              max_total_detections=100, stage='train'):
    image_input = layers.Input(shape=input_shape)

    anchors = generate_anchors(specs)
    feature_list = feature_fn(image_input)

    num_anchors_list = [len(spec.aspect_ratios) + 2 for spec in specs]
    predict_logits = cls_headers(feature_list, num_anchors_list, num_classes)
    predict_deltas = rgr_headers(feature_list, num_anchors_list)

    if stage == 'train':
        gt_boxes = layers.Input(shape=(max_gt_num, 5), dtype='float32')
        gt_class_ids = layers.Input(shape=(max_gt_num, 2), dtype='int32')
        # 分类和回归目标
        # for i, anchors in enumerate(anchors_list):
        #     target = SSDTarget(anchors, positive_iou_threshold, negative_iou_threshold,
        #                        name='target_{}'.format(i))
        #     deltas_list[i], labels_list[i], anchors_tag_list[i] = target([gt_boxes, gt_class_ids])
        #
        # # 计算loss
        # cls_loss = MultiClsLoss(name='multi_cls_loss')([predict_logits_list,
        #                                                 labels_list, anchors_tag_list])
        # rgr_loss = MultiRegressLoss(name='multi_rgr_loss')([predict_deltas_list,
        #                                                     deltas_list, anchors_tag_list])
        deltas, cls_ids, anchors_tag = SSDTarget(anchors, positive_iou_threshold, negative_iou_threshold,
                                                 name='ssd_target')([gt_boxes, gt_class_ids])
        cls_losses = layers.Lambda(lambda x: cls_loss(*x),
                                   name='cls_losses')([predict_logits, cls_ids, anchors_tag])
        rgr_losses = layers.Lambda(lambda x: regress_loss(*x),
                                   name='rgr_loss')([predict_deltas, deltas, anchors_tag])

        m = Model([image_input, gt_boxes, gt_class_ids], [cls_losses, rgr_losses])
    else:
        boxes, class_ids, scores = DetectBox(anchors,
                                             score_threshold=score_threshold,
                                             iou_threshold=iou_threshold,
                                             max_detections_per_class=max_detections_per_class,
                                             max_total_detections=max_total_detections)(
            [predict_deltas, predict_logits])

        m = Model(image_input, [boxes, class_ids, scores])

    return m


def main():
    from config import cfg

    model = ssd_model(cfg.feature_fn, (300, 300, 3), cfg.num_classes,
                      cfg.specs, cfg.max_gt_num, cfg.positive_iou_threshold,
                      cfg.negative_iou_threshold, stage='train')
    model.summary()


def test():
    x = layers.Input(shape=(4, 100, 100, 3))
    y = layers.Concatenate(axis=0)([x, x])
    m = Model(x, y)
    m.summary()


if __name__ == '__main__':
    main()
    # test()
