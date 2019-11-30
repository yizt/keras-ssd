# -*- coding: utf-8 -*-
"""
 @File    : demo.py
 @Time    : 2019/11/26 上午11:47
 @Author  : yizuotian
 @Description    :
"""

import argparse
import sys

import cv2
import numpy as np
import time

from config import cfg
from ssd import ssd_model
from utils import np_utils
from utils.preprocess import PredictionTransform


def main(args):
    from tensorflow.python.keras import backend as K
    import tensorflow as tf

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)
    K.set_session(session)
    # 加载图像
    im = cv2.imread(args.image_path)
    h, w = im.shape[:2]
    trans = PredictionTransform(cfg.image_size, cfg.mean_pixel, cfg.std)
    image = trans(im[:, :, ::-1])

    # 加载模型
    m = ssd_model(cfg.feature_fn, cfg.cls_head_fn, cfg.rgr_head_fn,
                  cfg.input_shape, cfg.num_classes, cfg.specs,
                  score_threshold=cfg.score_threshold,
                  iou_threshold=cfg.iou_threshold,
                  max_detections_per_class=cfg.max_detections_per_class,
                  max_total_detections=cfg.max_total_detections,
                  stage='test')
    m.load_weights(args.weight_path, by_name=True)
    # m.summary()
    # 预测边框、得分、类别
    s_time = time.time()
    boxes, scores, class_ids = m.predict(image[np.newaxis, :, :, :])
    print("预测耗时：{} 秒".format(time.time() - s_time))
    # 去除padding
    scores = np_utils.remove_pad(scores[0])[:, 0]
    class_ids = np_utils.remove_pad(class_ids[0])[:, 0]
    boxes = np_utils.remove_pad(boxes[0])
    # 还原检测边框到
    boxes = np.clip(boxes, 0, 1)
    boxes[:, [0, 2]] *= h
    boxes[:, [1, 3]] *= w

    # 画边框
    class_names = list(cfg.class_mapping.keys())
    for box, score, class_id in zip(boxes.astype(np.int32), scores, class_ids):
        y1, x1, y2, x2 = box
        text = '{}:{:03f}'.format(class_names[class_id], score)
        cv2.putText(im, text, (x1 + 8, y2 + 20), cv2.FONT_HERSHEY_PLAIN,
                    0.75, (0, 255, 0), thickness=1)
        cv2.rectangle(im, (x1, y1), (x2, y2), (0, 0, 255), 2)

    cv2.imwrite(args.result_path, im)


if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument("--weight_path", type=str, default=None, help="weight path")
    parse.add_argument("--image_path", type=str, default=None, help="image path")
    parse.add_argument("--result_path", type=str, default=None, help="output image path")
    arguments = parse.parse_args(sys.argv[1:])
    main(arguments)
