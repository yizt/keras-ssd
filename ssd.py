# -*- coding: utf-8 -*-
"""
 @File    : ssd.py
 @Time    : 2019/11/14 下午4:30
 @Author  : yizuotian
 @Description    :
"""
from typing import List

from tensorflow.python.keras import layers, Model

from layers.detect_boxes import DetectBox
from layers.losses import regress_loss, cls_loss
from layers.target import SSDTarget
from utils.anchor import generate_anchors, FeatureSpec


def ssd_model(feature_fn, cls_head_fn, rgr_head_fn, input_shape, num_classes, specs: List[FeatureSpec],
              max_gt_num=100, positive_iou_threshold=0.5, negative_iou_threshold=0.4,
              negatives_per_positive=3, min_negatives_per_image=5,
              score_threshold=0.5, iou_threshold=0.3, max_detections_per_class=100,
              max_total_detections=100, stage='train'):
    image_input = layers.Input(shape=input_shape, name='input_image')

    anchors = generate_anchors(specs, input_shape[0])
    feature_list = feature_fn(image_input)

    num_anchors_list = [len(spec.aspect_ratios) + 2 for spec in specs]
    predict_logits = cls_head_fn(feature_list, num_anchors_list, num_classes)
    predict_deltas = rgr_head_fn(feature_list, num_anchors_list)

    if stage == 'train':
        gt_boxes = layers.Input(shape=(max_gt_num, 5), dtype='float32', name='input_gt_boxes')
        gt_class_ids = layers.Input(shape=(max_gt_num, 2), dtype='int32', name='input_gt_class_ids')
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
        cls_losses = layers.Lambda(lambda x: cls_loss(*x,
                                                      negatives_per_positive=negatives_per_positive,
                                                      min_negatives_per_image=min_negatives_per_image),
                                   name='class_loss')([predict_logits, cls_ids, anchors_tag])
        rgr_losses = layers.Lambda(lambda x: regress_loss(*x),
                                   name='bbox_loss')([predict_deltas, deltas, anchors_tag])

        m = Model([image_input, gt_boxes, gt_class_ids], [cls_losses, rgr_losses])
    elif stage == 'debug':
        gt_boxes = layers.Input(shape=(max_gt_num, 5), dtype='float32', name='input_gt_boxes')
        gt_class_ids = layers.Input(shape=(max_gt_num, 2), dtype='int32', name='input_gt_class_ids')
        deltas, cls_ids, anchors_tag = SSDTarget(anchors, positive_iou_threshold, negative_iou_threshold,
                                                 name='ssd_target')([gt_boxes, gt_class_ids])
        m = Model([image_input, gt_boxes, gt_class_ids], [cls_ids, anchors_tag, predict_logits])
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

    # model = ssd_model(cfg.feature_fn, cfg.input_shape, cfg.num_classes,
    #                   cfg.specs, stage='train')
    # model.summary()
    # model = ssd_model(cfg.feature_fn, cfg.input_shape, cfg.num_classes,
    #                   cfg.specs, stage='test')
    # model.summary()
    model = ssd_model(cfg.feature_fn, cfg.cls_head_fn, cfg.rgr_head_fn,
                      cfg.input_shape, cfg.num_classes,
                      cfg.specs, stage='debug')
    model.summary()


def test():
    x = layers.Input(shape=(4, 100, 100, 3))
    y = layers.Concatenate(axis=0)([x, x])
    m = Model(x, y)
    m.summary()


if __name__ == '__main__':
    main()
    # test()
