# -*- coding: utf-8 -*-
"""
 @File    : generator.py
 @Time    : 2019/11/24 下午6:26
 @Author  : yizuotian
 @Description    :
"""
from tensorflow.python.keras import utils
import numpy as np
from typing import List
import cv2
from datasets.dataset import ImageInfo
from utils import np_utils


class Generator(utils.data_utils.Sequence):
    def __init__(self, image_info_list: List[ImageInfo], transforms, input_shape, batch_size=1,
                 max_gt_num=50, **kwargs):
        """

        :param image_info_list:
        :param transforms:
        :param input_shape:
        :param batch_size:
        :param max_gt_num:
        :param kwargs:
        """
        self.input_shape = input_shape
        self.image_info_list = image_info_list
        self.transforms = transforms
        self.batch_size = batch_size
        self.max_gt_num = max_gt_num
        self.size = len(image_info_list)
        super(Generator, self).__init__(**kwargs)

    def on_epoch_end(self):
        # 一个epoch重新打乱
        np.random.shuffle(self.image_info_list)

    def __len__(self):
        return self.size // self.batch_size

    def __getitem__(self, index):
        indices = np.arange(index * self.batch_size, (index + 1) * self.batch_size)
        images = np.zeros((self.batch_size,) + self.input_shape, dtype=np.float32)
        batch_gt_boxes = np.zeros((self.batch_size, self.max_gt_num, 5), dtype=np.float32)
        batch_gt_class_ids = np.ones((self.batch_size, self.max_gt_num, 2), dtype=np.int32)
        for i, index in enumerate(indices):
            # 加载图像
            image = cv2.imread(self.image_info_list[index].image_path)[:, :, ::-1]
            gt_boxes = self.image_info_list[index].boxes.copy()  # 不改变原来的
            gt_class_ids = self.image_info_list[index].labels

            # resize图像
            images[i], batch_gt_boxes[i], batch_gt_class_ids[i] = self.transforms(image, gt_boxes, gt_class_ids)
            # pad gt到固定个数
            batch_gt_boxes[i] = np_utils.pad_to_fixed_size(gt_boxes, self.max_gt_num)
            batch_gt_class_ids[i] = np_utils.pad_to_fixed_size(
                np.expand_dims(self.annotation_list[index]['labels'], axis=1),
                self.max_gt_num)

        return {"input_image": images,
                "input_gt_boxes": batch_gt_boxes,
                "input_gt_class_ids": batch_gt_class_ids}, None
