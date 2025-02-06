#encoding: utf-8

import cv2
import torch
import numpy as np


def tensor2image(image):
    image = 255. * image.cpu().numpy()
    image = np.clip(image, 0, 255).astype(np.uint8)
    return image


def imagelist2tensor(image_list, size):
    '''
    list of image, numpy format
    '''
    image_list = [cv2.resize(image, size) for image in image_list]
    image_list = np.array(image_list, dtype=np.float32)
    image_list = torch.from_numpy(image_list).div_(255)
    return image_list
