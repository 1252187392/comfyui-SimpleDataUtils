#encoding: utf-8

import os
import random

import cv2
import torch
import numpy as np
from .lib.model import init_pipe, generate_image


def cropImage(images):
    new_images = []
    for image in images:
        h, w, c = image.shape
        s = min(h, w)
        crop = (w - s, h - s)
        start_x, start_y = crop[0] // 2, crop[1] // 2
        new_images.append(image[start_y: start_y + s, start_x: start_x + s, :])
    return new_images

class ReadImageDir:
    '''
    指定目录，读取该目录下所有的图片
    '''
    FUNCTION = "read_image_dir"
    RETURN_NAMES = ("images",)
    RETURN_TYPES = ("IMAGE",)
    CATEGORY = 'SimpleDataset'
    #OUTPUT_NODE = True  # 声明这个节点是叶子节点

    @classmethod
    def INPUT_TYPES(self):
        return {'required': {
            'data_dir': ("STRING", {"default": "path to read"}),
        }}


    def read_image_dir(self, data_dir):
        images = []
        for root, dirs, fs in os.walk(data_dir):
            for f in fs:
                tail = f.split('.')[-1]
                print('filename', f, 'tail', tail)
                if tail not in ['jpg', 'jpeg', 'png']:
                    continue
                im = cv2.imread(os.path.join(root, f))
                if im.shape[-1] == 3:
                    print(im.shape)
                    im = im[:, :, ::-1]
                im = np.array(im, dtype=np.float32)
                images.append(torch.from_numpy(im).div_(255))
        #images = np.array(images, dtype=np.float32)
        #images = torch.from_numpy(images).div_(255)
        return (images, )


class GenFileNames:
    '''
    输入是一个list的数据，输出的是同长度的文件名([image_name_0,..],[text_name_0,...],)
    '''

    FUNCTION = "gen_names"
    RETURN_NAMES = ("image_names", "text_names", "save_dir")
    RETURN_TYPES = ("STRING", "STRING", "STRING")
    CATEGORY = 'SimpleDataset'
    OUTPUT_NODE = True # 声明这个节点是叶子节点

    # 定义输入
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {
            'list_image': ("IMAGE",),
            'texts': ('STRING', {'forceInput': True}), # forceInput对应节点输入
            'prefix': ('STRING', ),
            'save_dir': ('STRING', {'default': 'I://datasets//output1'})
        }}

    def gen_names(self, list_image, texts, prefix, save_dir):
        '''
        需要的参数需要是INPUT_TYPE中的key
        list_image: normalize后的image
        texts: len(texts) == len(list_image), texts[i]是list_image[i]的caption
        prefix: 输出文件名的prefix
        save_dir: 输出目录
        '''
        image_names = [prefix + str(i) + '.jpg' for i in range(len(list_image))]
        text_names = [prefix + str(i) + '.txt' for i in range(len(list_image))]
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        for text, text_name in zip(texts, text_names):
            file = open(os.path.join(save_dir, text_name), 'w')
            file.write(text)
            file.close()
        for image, image_name in zip(list_image, image_names):
            image = 255. * image.cpu().numpy()
            image = np.clip(image, 0, 255).astype(np.uint8)[:,:,::-1]
            cv2.imwrite(os.path.join(save_dir, image_name), image)
        for i, text in enumerate(texts):
            print(i, text)

        return (image_names, text_names, save_dir, )


class GenImages:
    FUNCTION = "genImages"
    RETURN_NAMES = ("save_dir", )
    RETURN_TYPES = ("STRING", )
    CATEGORY = 'SimpleDataset'
    OUTPUT_NODE = True # 声明这个节点是叶子节点


    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {
            'data_dir': ("STRING", {"forceInput": True, "default": "I://datasets//output"}),
            'num': ("INT", {"default": 1}),
            'steps': ("INT", {"default": 25})
        }}


    def genImages(self, data_dir, num, steps):
        '''
        每个txt文件生成num张图
        data_dir: 读取该目录下所有.txt文件
        '''
        pipe = init_pipe()
        assert data_dir[-1] not in ('/', '\\')
        save_dir = data_dir + '_gen'
        print('save_dir', save_dir)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        for root, dirs, fs in os.walk(data_dir):
            for f in fs:
                if f[-3:] != 'txt':
                    continue
                content = open(os.path.join(root, f)).read()
                print(content)
                seed = random.randint(0, 100000000)
                images = generate_image(pipe, content, steps, seed, num, 'cuda')# 用sd生成num张image
                for idx, image in enumerate(images):
                    #cv2.imwrite(os.path.join(save_dir, f.replace('.txt', f'_{idx}.jpg')), image)
                    image.save(os.path.join(save_dir, f.replace('.txt', f'_{idx}.jpg')))
                    file = open(os.path.join(save_dir, f.replace('.txt', f'_{idx}.txt')), 'w')
                    file.write(content)
                    file.close()
        return (save_dir, )


class CenterCrop:
    '''
    处理成正方形
    '''
    FUNCTION = "cropImage"
    RETURN_NAMES = ("images",)
    RETURN_TYPES = ("IMAGE",)
    CATEGORY = 'SimpleDataset'
    OUTPUT_NODE = True  # 声明这个节点是叶子节点

    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {
            'images': ("IMAGE", {"forceInput": True}),
            'target_size': ("INT", {"default": 640})
        }}


    def cropImage(self, images, target_size):
        new_images = []
        for image in images:
            image = 255. * image.cpu().numpy()
            image = np.clip(image, 0, 255).astype(np.uint8)
            h, w, c = image.shape
            s = min(h, w)
            crop = (w - s, h - s)
            start_x, start_y = crop[0] // 2, crop[1] // 2
            image = image[start_y: start_y + s,start_x: start_x+s, :]
            image = cv2.resize(image, (target_size, target_size))
            new_images.append(image)
        new_images = np.array(new_images, dtype=np.float32)
        new_images = torch.from_numpy(new_images).div_(255)
        return (new_images, )


NODE_CLASS_MAPPINGS = {
    "GenFileNames": GenFileNames,
    "ReadImageDir": ReadImageDir,
    "GenImages": GenImages,
    "CenterCrop": CenterCrop,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GenFileNames": "GenFileNames",
    "ReadImageDir": "ReadImageDir",
    "GenImages": "GenImages",
    "CenterCrop": "CenterCrop"
}
