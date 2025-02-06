#encoding: utf-8
import torch
import random
from PIL import Image
import numpy as np
from ultralytics import YOLO

from .lib.model import init_controlnet_pipe, generate_image_control
from .lib.utils import imagelist2tensor, tensor2image

class GenImages:
    FUNCTION = "genImages"
    RETURN_NAMES = ("images", "controls")
    RETURN_TYPES = ("IMAGE", "IMAGE")
    CATEGORY = 'SimpleDataset'
    OUTPUT_NODE = True  # 声明这个节点是叶子节点


    @classmethod
    def INPUT_TYPES(cls):
         return {'required': {
            'images': ('IMAGE', ),
            'num': ("INT", {"default": 1}),
            'steps': ("INT", {"default": 25}),
            'prompt': ("STRING", {"forceInput": True}),
            'prefix': ("STRING", {"default": "prompt = prefix + prompt"}),
            'seed' : ("INT", {"default": 32131}),
            'lora_weights': ("STRING", {"default": "load lora"})
        }}


    def genImages(self, images, num, steps, prompt, prefix, seed, lora_weights):
        '''
        '''
        print(type(images))
        print(images.shape)
        res, controls = [], []
        pipe = init_controlnet_pipe(lora_weights)
        for image in images:
            image = 255. * image.cpu().numpy()
            image = np.clip(image, 0, 255).astype(np.uint8)
            image = Image.fromarray(image)
            prompt = prefix + ',' + prompt
            #prompt = prefix
            print("prompt:", prompt)
            images, control_image = generate_image_control(image, pipe, prompt, steps, seed, num, 'cuda')
            res += images
            controls.append(np.array(control_image))
        res = np.array(res, dtype=np.float32)
        res = torch.from_numpy(res).div_(255)
        controls = np.array(controls, dtype=np.float32)
        controls = torch.from_numpy(controls).div_(255)
        return (res, controls,)


def rand_crop(image, box, size=512):
    x1, y1, x2, y2 = box
    H, W, _ = image.shape
    w = x2 - x1
    h = y2 - y1
    if max(w, h) >= size:
        size = max(w, h)
    x_min = max(x2 - size, 0)
    x_max = min(x1, W - size)
    y_min = max(y2 - size, 0)
    y_max = min(y1, H - size)
    x, y = random.randint(x_min, x_max), random.randint(y_min, y_max)
    return image[y:y+size, x:x+size, :]


class CropFace:
    FUNCTION = "cropFace"
    RETURN_NAMES = ("images", )
    RETURN_TYPES = ("IMAGE", )
    CATEGORY = 'SimpleDataset'
    OUTPUT_NODE = False  # 声明这个节点是叶子节点

    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {
            'images': ("IMAGE", {"forceInput": True}),
            'target_size': ("INT", {"default": 512})
        }}


    def cropFace(self, images, target_size):
        #imagelist2tensor
        model_path = "I:\hugginface\huggingface\hub\models--AdamCodd--YOLOv11n-face-detection\snapshots\895c5d8452685fab8ff9e33b28098d926adc90c4\model.pt"
        model = YOLO(model_path)
        res = []
        c_size = [target_size + _*64 for _ in range(3)]
        for image in images:
            image = tensor2image(image)
            results = model.predict(image)
            for r in results:
                box = r.boxes.data.cpu().numpy()
                if len(box) == 0:
                    continue
                box = [int(_) for _ in box[0][:4]]
                for _size in c_size:
                    crop_image = rand_crop(image, box, _size)
                    res.append(crop_image)
        res = imagelist2tensor(res, (target_size, target_size))
        return (res, )


