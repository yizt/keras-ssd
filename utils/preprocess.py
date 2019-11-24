# -*- coding: utf-8 -*-
"""
 @File    : preprocess.py
 @Time    : 2019/11/3 下午7:41
 @Author  : yizuotian
 @Description    : 预处理
"""
from utils.augment import *
from utils.photometric import *


class PhotometricDistort(object):
    """
    光度扭曲
    """

    def __init__(self):
        self.pd = [
            RandomContrast(),  # RGB
            ConvertColor(current="RGB", transform='HSV'),  # HSV
            RandomSaturation(),  # HSV
            RandomHue(),  # HSV
            ConvertColor(current='HSV', transform='RGB'),  # RGB
            RandomContrast()  # RGB
        ]
        self.rand_brightness = RandomBrightness()
        self.rand_light_noise = RandomChannelSwap()

    def __call__(self, image, boxes, labels):
        im = image.copy()
        im, boxes, labels = self.rand_brightness(im, boxes, labels)
        if random.random() < 0.5:
            distort = Compose(self.pd[:-1])
        else:
            distort = Compose(self.pd[1:])
        im, boxes, labels = distort(im, boxes, labels)
        return self.rand_light_noise(im, boxes, labels)


class TrainAugmentation:
    def __init__(self, size, mean=0, std=1.0):
        """

        :param size:
        :param mean:
        :param std:
        """
        self.mean = mean
        self.size = size
        self.augment = Compose([
            PhotometricDistort(),
            RandomSampleCrop(),
            RandomHorizontalFlip(),
            ToPercentCoordinates(),
            Resize(self.size, self.size),
            SubtractMeans(self.mean),
            lambda img, boxes=None, labels=None: (img / std, boxes, labels),
            # ToTensor(),
        ])

    def __call__(self, img, boxes, labels):
        """

        Args:
            img: [H,W,3]
            boxes: [N,(y1,x1,y2,x2)]
            labels: [N]
        """
        return self.augment(img, boxes, labels)


class EvalTransform:
    def __init__(self, size, mean=0.0, std=1.0):
        self.transform = Compose([
            ToPercentCoordinates(),
            Resize(size, size),
            SubtractMeans(mean),
            lambda img, boxes=None, labels=None: (img / std, boxes, labels),
            # ToTensor(),
        ])

    def __call__(self, image, boxes, labels):
        return self.transform(image, boxes, labels)


class PredictionTransform:
    def __init__(self, size, mean=0.0, std=1.0):
        self.transform = Compose([
            Resize(size, size),
            SubtractMeans(mean),
            lambda img, boxes=None, labels=None: (img / std, boxes, labels),
            # ToTensor()
        ])

    def __call__(self, image):
        image, _, _ = self.transform(image)
        return image


def main():
    im = np.random.rand(224, 224, 3) * 255
    im = im.astype(np.uint8)
    trans = PhotometricDistort()
    img, _, _ = trans(im, [], [])


if __name__ == '__main__':
    main()
