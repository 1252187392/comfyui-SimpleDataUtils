#from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
from .nodes import MergeImages, GenFileNames, ReadImageDir, GenImagesFromDir, CenterCrop, Video2Frames
from .model_nodes import GenImages, CropFace

NODE_CLASS_MAPPINGS = {
    "GenFileNames": GenFileNames,
    "ReadImageDir": ReadImageDir,
    "GenImagesFromDir": GenImagesFromDir,
    "CenterCrop": CenterCrop,
    "GenImages": GenImages,
    "Video2Frames": Video2Frames,
    "CropFace": CropFace,
    "MergeImages": MergeImages
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GenFileNames": "GenFileNames",
    "ReadImageDir": "ReadImageDir",
    "GenImagesFromDir": "GenImagesFromDir",
    "CenterCrop": "CenterCrop",
    "GenImages": "GenImages",
    "Video2Frames": "Video2Frames",
    "CropFace": "CropFace",
    "MergeImages": "MergeImages"
}


__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
