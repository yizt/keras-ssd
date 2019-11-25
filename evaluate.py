# -*- coding: utf-8 -*-
"""
 @File    : evaluate.py
 @Time    : 2019/11/25 上午9:36
 @Author  : yizuotian
 @Description    :
"""

import argparse
import sys
import time
import numpy as np
from datasets.dataset import VocDataset, ImageInfo
from config import cfg
from utils import np_utils, eval_utils
from ssd import ssd_model
from utils.generator import TestGenerator
from utils.preprocess import PredictionTransform
from typing import List


def main(args):
    from tensorflow.python.keras import backend as K
    import tensorflow as tf

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)
    K.set_session(session)

    # 加载数据集
    dataset = VocDataset(cfg.voc_path, class_mapping=cfg.class_mapping)
    dataset.prepare()
    print("len:{}".format(len(dataset.get_image_info_list())))
    test_image_info_list = [info for info in dataset.get_image_info_list()
                            if info.type == args.data_set][:args.eval_num]
    print("len:{}".format(len(test_image_info_list)))
    # generator
    transform = PredictionTransform(cfg.image_size, cfg.mean_pixel, cfg.std)
    gen = TestGenerator(test_image_info_list, transform, cfg.input_shape, args.batch_size)
    # 加载模型
    m = ssd_model(cfg.feature_fn, cfg.input_shape, cfg.num_classes, cfg.specs,
                  score_threshold=0.01,
                  iou_threshold=cfg.iou_threshold,
                  max_detections_per_class=cfg.max_detections_per_class,
                  max_total_detections=cfg.max_total_detections,
                  stage='test')
    m.load_weights(args.weight_path, by_name=True)
    # m.summary()
    # 预测边框、得分、类别
    s_time = time.time()
    boxes, scores, class_ids = m.predict_generator(
        gen,
        verbose=1,
        workers=4,
        use_multiprocessing=False)
    print("预测 {} 张图像,耗时：{} 秒".format(len(test_image_info_list), time.time() - s_time))
    # 去除padding
    predict_scores = [np_utils.remove_pad(score)[:, 0] for score in scores]
    predict_labels = [np_utils.remove_pad(label)[:, 0] for label in class_ids]
    # 还原检测边框到
    predict_boxes = recover_detect_boxes(test_image_info_list, boxes)
    # 以下是评估过程
    annotations = eval_utils.get_annotations(test_image_info_list, cfg.num_classes)
    detections = eval_utils.get_detections(predict_boxes, predict_scores, predict_labels, cfg.num_classes)
    average_precisions = eval_utils.voc_eval(annotations, detections, iou_threshold=0.5, use_07_metric=True)
    print("ap:{}".format(average_precisions))
    # 求mean ap 去除背景类
    mAP = np.mean(np.array(list(average_precisions.values()))[1:])
    print("mAP:{}".format(mAP))
    print("整个评估过程耗时：{} 秒".format(time.time() - s_time))


def recover_detect_boxes(image_info_list: List[ImageInfo], boxes):
    """
    检测边框坐标还原到原始图像
    :param image_info_list:
    :param boxes: [B,max_detections,(y1,x1,y2,x2,tag)]
    :return:
    """
    boxes_list = []
    boxes = np.clip(boxes, 0, cfg.image_size)  # 边框裁剪到图像内
    for image_info, box in zip(image_info_list, boxes):
        box = np_utils.remove_pad(box)
        if len(box) == 0:
            boxes_list.append(box)
        else:
            h, w = image_info.height, image_info.width
            box[:, [0, 2]] *= h / cfg.image_size
            box[:, [1, 3]] *= w / cfg.image_size
            boxes_list.append(box)
    return boxes_list


if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument("--weight_path", type=str, default=None, help="weight path")
    parse.add_argument("--data_set", type=str, default='test', help="dataset to evaluate")
    parse.add_argument("--batch_size", type=int, default=1, help="batch size")
    parse.add_argument("--eval_num", type=int, default=1000000, help="number of images to evaluate")
    arguments = parse.parse_args(sys.argv[1:])
    main(arguments)
