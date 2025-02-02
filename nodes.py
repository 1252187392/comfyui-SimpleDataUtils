#encoding: utf-8

import cv2
import os
import numpy as np
import torch
import folder_paths


class ReadImageDir:
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
                images.append(im)
        images = np.array(images, dtype=np.float32)
        images = torch.from_numpy(images).div_(255)
        return (images, )


class GenFileNames:
    '''
    输入是一个list的数据，输出的是通长度的文件名
    '''

    FUNCTION = "gen_names"
    RETURN_NAMES = ("image_names", "text_names")
    RETURN_TYPES = ("STRING", "STRING")
    CATEGORY = 'SimpleDataset'
    OUTPUT_NODE = True # 声明这个节点是叶子节点

    # 定义输入
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {
            'list_image': ("IMAGE",),
            'texts': ('STRING',{'forceInput': True}), # forceInput对应节点输入
            'prefix': ('STRING', ),
            'save_dir': ('STRING', {'default': 'I://datasets//output1'})
        }}

    def gen_names(self, list_image, texts, prefix, save_dir):
        '''
        需要的参数需要是INPUT_TYPE中的key
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
            #image = image.cpu().numpy()
            image = 255. * image.cpu().numpy()
            #image = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            image = np.clip(image, 0, 255).astype(np.uint8)[:,:,::-1]
            print(type(image))
            cv2.imwrite(os.path.join(save_dir, image_name), image)
        for i, text in enumerate(texts):
            print(i, text)

        return (image_names, text_names, )

NODE_CLASS_MAPPINGS = {
    "GenFileNames": GenFileNames,
    "ReadImageDir": ReadImageDir
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GenFileNames": "GenFileNames",
    "ReadImageDir": "ReadImageDir"
}
