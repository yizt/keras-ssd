# -*- coding: utf-8 -*-
"""
 @File    : target.py
 @Time    : 2019/11/14 下午5:03
 @Author  : yizuotian
 @Description    :
"""
import tensorflow as tf
from tensorflow.python.keras import layers
from utils.tf_utils import remove_pad


def target_graph(gt_boxes, gt_class_ids, anchors, num_anchors, positive_iou_threshold, negative_iou_threshold):
    """

    :param gt_boxes: [max_gt_num,(y1,x1,y2,x2,tag)]
    :param gt_class_ids: [max_gt_num,(class_id,tag)]
    :param anchors: [num_anchors,(y1,x1,y2,x2)]
    :param num_anchors:
    :param positive_iou_threshold:
    :param negative_iou_threshold:
    :return deltas: [num_anchors,(dy,dx,dh,dw)
    :return class_ids: [num_anchors]
    :return anchors_tag: [num_anchors] 1-positive,-1:negative,0-ignore
    """
    gt_boxes = remove_pad(gt_boxes)
    gt_class_ids = remove_pad(gt_class_ids)[:, 0]
    # gt boxes为0时，增加一个虚拟的背景boxes;防止后续计算ious时为空
    gt_boxes, gt_class_ids = tf.cond(tf.size(gt_boxes) > 0,
                                     true_fn=lambda: [gt_boxes, gt_class_ids],
                                     false_fn=lambda: [tf.constant([[0., 0., 1., 1.]], dtype=tf.float32),
                                                       tf.constant([[0]], dtype=tf.int32)])

    ious = compute_iou(gt_boxes, anchors)
    anchors_iou_max = tf.reduce_max(ious, axis=0)  # [num_anchors]
    anchors_match_gt_indices = tf.argmax(ious, axis=0)  # [num_anchors]

    match_gt_boxes = tf.gather(gt_boxes, anchors_match_gt_indices)
    match_gt_class_ids = tf.gather(gt_class_ids, anchors_match_gt_indices)

    # 正负样本标志;1-positive,-1:negative,0-ignore
    anchors_tag = tf.where(anchors_iou_max >= positive_iou_threshold,
                           tf.ones_like(anchors_iou_max),
                           tf.where(anchors_iou_max < negative_iou_threshold,
                                    -1 * tf.ones_like(anchors_iou_max),
                                    tf.zeros_like(anchors_iou_max)))
    # 回归目标
    deltas = regress_target(anchors, match_gt_boxes)
    # 分类目标
    class_ids = match_gt_class_ids
    class_ids.set_shape([num_anchors])

    return [deltas, class_ids, anchors_tag]


def regress_target(anchors, gt_boxes):
    """
    计算回归目标
    :param anchors: [N,(y1,x1,y2,x2)]
    :param gt_boxes: [N,(y1,x1,y2,x2)]
    :return: [N,(y1,x1,y2,x2)]
    """
    # 高度和宽度
    h = anchors[:, 2] - anchors[:, 0]
    w = anchors[:, 3] - anchors[:, 1]

    gt_h = gt_boxes[:, 2] - gt_boxes[:, 0]
    gt_w = gt_boxes[:, 3] - gt_boxes[:, 1]
    # 中心点
    center_y = (anchors[:, 2] + anchors[:, 0]) * 0.5
    center_x = (anchors[:, 3] + anchors[:, 1]) * 0.5
    gt_center_y = (gt_boxes[:, 2] + gt_boxes[:, 0]) * 0.5
    gt_center_x = (gt_boxes[:, 3] + gt_boxes[:, 1]) * 0.5

    # 回归目标
    dy = (gt_center_y - center_y) / h
    dx = (gt_center_x - center_x) / w
    dh = tf.log(gt_h / h)
    dw = tf.log(gt_w / w)

    target = tf.stack([dy, dx, dh, dw], axis=1)
    target /= tf.constant([0.1, 0.1, 0.2, 0.2])
    # target = tf.where(tf.greater(target, 100.0), 100.0, target)
    return target


def compute_iou(gt_boxes, anchors):
    """
    计算iou
    :param gt_boxes: [N,(y1,x1,y2,x2)]
    :param anchors: [M,(y1,x1,y2,x2)]
    :return: IoU [N,M]
    """
    gt_boxes = tf.expand_dims(gt_boxes, axis=1)  # [N,1,4]
    anchors = tf.expand_dims(anchors, axis=0)  # [1,M,4]
    # 交集
    intersect_w = tf.maximum(0.0,
                             tf.minimum(gt_boxes[:, :, 3], anchors[:, :, 3]) -
                             tf.maximum(gt_boxes[:, :, 1], anchors[:, :, 1]))
    intersect_h = tf.maximum(0.0,
                             tf.minimum(gt_boxes[:, :, 2], anchors[:, :, 2]) -
                             tf.maximum(gt_boxes[:, :, 0], anchors[:, :, 0]))
    intersect = intersect_h * intersect_w

    # 计算面积
    area_gt = (gt_boxes[:, :, 3] - gt_boxes[:, :, 1]) * \
              (gt_boxes[:, :, 2] - gt_boxes[:, :, 0])
    area_anchor = (anchors[:, :, 3] - anchors[:, :, 1]) * \
                  (anchors[:, :, 2] - anchors[:, :, 0])

    # 计算并集
    union = area_gt + area_anchor - intersect
    # 交并比
    iou = tf.divide(intersect, union, name='regress_target_iou')
    return iou


class SSDTarget(layers.Layer):
    def __init__(self, anchors, positive_iou_threshold, negative_iou_threshold, **kwargs):
        self.anchors = anchors
        self.num_anchors = len(anchors)
        self.positive_iou_threshold = positive_iou_threshold
        self.negative_iou_threshold = negative_iou_threshold
        super(SSDTarget, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        """

        :param inputs:
        inputs[0] gt_boxes: [B,max_gt_num,(y1,x1,y2,x2,tag)]
        inputs[1] gt_class_ids: [B,max_gt_num,(class_id,tag)],
        :param kwargs:
        :return:
        """
        gt_boxes, gt_class_ids = inputs
        anchors = tf.constant(self.anchors, dtype=tf.float32)
        options = {"anchors": anchors,
                   "num_anchors": self.num_anchors,
                   "positive_iou_threshold": self.positive_iou_threshold,
                   "negative_iou_threshold": self.negative_iou_threshold}
        outputs = tf.map_fn(lambda x: target_graph(*x, **options),
                            elems=[gt_boxes, gt_class_ids],
                            dtype=[tf.float32, tf.int32, tf.float32])
        return outputs

    def compute_output_shape(self, input_shape):
        batch_size, anchors_num = input_shape[0][:2]
        return [(batch_size, anchors_num, 4),  # deltas
                (batch_size, anchors_num),  # class_ids
                (batch_size, anchors_num)  # anchors_tag
                ]


def test():
    sess = tf.Session()
    anchors_iou_max = tf.constant([0.5, 0.4, 0.3, 0.6, 0.1])
    anchors_tag = tf.where(anchors_iou_max >= 0.5, tf.ones_like(anchors_iou_max),
                           tf.where(anchors_iou_max < 0.4, tf.ones_like(anchors_iou_max) * -1,
                                    tf.zeros_like(anchors_iou_max)))
    print(sess.run(anchors_tag))


if __name__ == '__main__':
    test()
    # a = tf.constant([1, 1, 1, 1])
    # x = tf.ones_like(a)
    # idx = tf.constant([[1], [3]])
    # v = tf.constant([1, 1])
    # y = tf.tensor_scatter_update(a, idx, [0, 0])
    # # z = tf.scatter_update(x, idx, v)
    # sess.run(tf.global_variables_initializer())
    # print(sess.run(tf.scatter_nd(idx, v, [8])))
    # print(sess.run(y))
    # print(sess.run(z))
    # g = tf.Graph()
    # with g.as_default():
    #     a = tf.Variable(initial_value=[[0, 0, 0, 0], [0, 0, 0, 0]])
    #     b = tf.scatter_update(a, [0, 1], [[1, 1, 0, 0], [1, 0, 4, 0]])
    #
    # with tf.Session(graph=g) as sess:
    #     sess.run(tf.global_variables_initializer())
    #     print(sess.run(a))
    #     print(sess.run(b))
