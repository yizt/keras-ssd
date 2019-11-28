# -*- coding: utf-8 -*-
"""
 @File    : detect_boxes.py
 @Time    : 2019/11/24 下午3:37
 @Author  : yizuotian
 @Description    :
"""
from tensorflow.python.keras import layers
import tensorflow as tf
from utils.tf_utils import pad_to_fixed_size


def apply_regress(deltas, anchors):
    """
    应用回归目标到边框
    :param deltas: 回归目标[N,(dy, dx, dh, dw)]
    :param anchors: anchor boxes[N,(y1,x1,y2,x2)]
    :return:
    """
    # 高度和宽度
    h = anchors[:, 2] - anchors[:, 0]
    w = anchors[:, 3] - anchors[:, 1]

    # 中心点坐标
    cy = (anchors[:, 2] + anchors[:, 0]) * 0.5
    cx = (anchors[:, 3] + anchors[:, 1]) * 0.5

    # 回归系数
    deltas *= tf.constant([0.1, 0.1, 0.2, 0.2])
    dy, dx, dh, dw = deltas[:, 0], deltas[:, 1], deltas[:, 2], deltas[:, 3]

    # 中心坐标回归
    cy += dy * h
    cx += dx * w
    # 高度和宽度回归
    h *= tf.exp(dh)
    w *= tf.exp(dw)

    # 转为y1,x1,y2,x2
    y1 = cy - h * 0.5
    x1 = cx - w * 0.5
    y2 = cy + h * 0.5
    x2 = cx + w * 0.5

    return tf.stack([y1, x1, y2, x2], axis=1)


def detect_boxes(boxes, class_logits, score_threshold, iou_threshold, max_detections_per_class,
                 max_total_detections):
    """
    使用类别相关的非极大抑制nms生成最终检测边框
    :param boxes: 形状为[num_boxes, 4]的二维浮点型Tensor.
    :param class_logits: 形状为[num_boxes,num_classes] 原始的预测类别
    :param score_threshold:  浮点数, 过滤低于阈值的边框
    :param iou_threshold: 浮点数,IOU 阈值
    :param max_detections_per_class: 每一个类别最多输出边框数
    :param max_total_detections: 最多输出边框数
    :return: 检测边框、边框得分、边框类别
    """
    # 类别得分和预测类别
    class_scores = tf.reduce_max(tf.nn.softmax(class_logits, axis=-1), axis=-1)  # [num_boxes]
    class_ids = tf.argmax(class_logits, axis=-1)  # [num_boxes]
    # 过滤背景类别class_id=0和低于阈值的
    keep = tf.where(tf.logical_and(class_ids > 0,
                                   class_scores >= score_threshold))
    keep_class_scores = tf.gather_nd(class_scores, keep)
    keep_class_ids = tf.gather_nd(class_ids, keep)
    keep_boxes = tf.gather_nd(boxes, keep)

    # 按类别nms
    unique_class_ids = tf.unique(class_ids)[0]

    def per_class_nms(class_id):
        # 当前类别的索引
        idx = tf.where(tf.equal(keep_class_ids, class_id))  # [n,1]
        cur_class_scores = tf.gather_nd(keep_class_scores, idx)
        cur_class_boxes = tf.gather_nd(keep_boxes, idx)

        indices = tf.image.non_max_suppression(cur_class_boxes,
                                               cur_class_scores,
                                               max_detections_per_class,
                                               iou_threshold,
                                               score_threshold)  # 一维索引
        # 映射索引
        idx = tf.gather(idx, indices)  # [m,1]
        # padding值为 -1
        pad_num = tf.maximum(0, max_detections_per_class - tf.shape(idx)[0])
        return tf.pad(idx, paddings=[[0, pad_num], [0, 0]], mode='constant', constant_values=-1)

    # 经过类别nms后保留的class_id 索引
    nms_keep = tf.map_fn(fn=per_class_nms, elems=unique_class_ids)  # [s,max_detections_per_class,1]
    # 打平
    nms_keep = tf.reshape(nms_keep, shape=[-1])  # [s]
    # 去除padding
    nms_keep = tf.gather_nd(nms_keep, tf.where(nms_keep > -1))  # [s]

    # 获取类别nms的边框,评分,类别以及logits
    output_boxes = tf.gather(keep_boxes, nms_keep)
    output_scores = tf.gather(keep_class_scores, nms_keep)
    output_class_ids = tf.gather(keep_class_ids, nms_keep)

    # 保留评分最高的top N
    top_num = tf.minimum(max_total_detections, tf.shape(output_scores)[0])
    top_idx = tf.nn.top_k(output_scores, k=top_num)[1]  # top_k返回tuple(values,indices)
    output_boxes = tf.gather(output_boxes, top_idx)
    output_scores = tf.gather(output_scores, top_idx)
    output_class_ids = tf.gather(output_class_ids, top_idx)

    # 增加padding,返回最终结果
    return [pad_to_fixed_size(output_boxes, max_total_detections),
            pad_to_fixed_size(tf.expand_dims(output_scores, axis=1), max_total_detections),
            pad_to_fixed_size(tf.expand_dims(output_class_ids, axis=1), max_total_detections)]


class DetectBox(layers.Layer):
    """
    根据候选框生成最终的检测框
    """

    def __init__(self, anchors, score_threshold=0.5, iou_threshold=0.3, max_detections_per_class=100,
                 max_total_detections=100, **kwargs):
        """
        :param anchors: numpy array [num_anchors,4]
        :param score_threshold: 分数阈值
        :param iou_threshold: nms iou阈值; 由于是类别相关的iou值较低
        :param max_detections_per_class: 每一个类别最多输出边框数
        :param max_total_detections: 最多输出边框数
        """
        self.anchors = anchors
        self.score_threshold = score_threshold
        self.iou_threshold = iou_threshold
        self.max_detections_per_class = max_detections_per_class
        self.max_total_detections = max_total_detections
        super(DetectBox, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        """
        应用边框回归，并使用nms生成最后的边框
        :param inputs:
        inputs[0]: deltas, [batch_size,num_anchors,(dy,dx,dh,dw)]
        inputs[1]: class logits [batch_size,num_anchors,num_classes]
        inputs[2]: anchors [batch_size,num_anchors,(y1,x1,y2,x2)]
        :param kwargs:
        :return:
        """
        deltas, class_logits = inputs
        # 应用边框回归
        anchors = tf.constant(self.anchors, dtype=tf.float32)
        options = {"anchors": anchors}
        boxes = tf.map_fn(lambda x: apply_regress(*x, **options),
                          elems=[deltas],
                          dtype=tf.float32)

        # # 非极大抑制
        options = {"score_threshold": self.score_threshold,
                   "iou_threshold": self.iou_threshold,
                   "max_detections_per_class": self.max_detections_per_class,
                   "max_total_detections": self.max_total_detections
                   }

        outputs = tf.map_fn(lambda x: detect_boxes(*x, **options),
                            elems=[boxes, class_logits],
                            dtype=[tf.float32] * 2 + [tf.int64])
        return outputs

    def compute_output_shape(self, input_shape):
        """
        注意多输出，call返回值必须是列表
        :param input_shape:
        :return: [boxes,scores,class_ids]
        """
        return [(input_shape[0][0], self.max_total_detections, 4 + 1),
                (input_shape[0][0], self.max_total_detections, 1 + 1),
                (input_shape[0][0], self.max_total_detections, 1 + 1)]
