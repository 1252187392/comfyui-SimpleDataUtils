# encoding: utf-8
import cv2
import torch
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from diffusers import DiffusionPipeline, DDIMScheduler, DDPMScheduler

def init_pipe():
    #repo_id = "runwayml/stable-diffusion-v1-5"
    repo_id = "I:\\models\\huggingface\\hub\\models--runwayml--stable-diffusion-v1-5\\snapshots\\aa9ba505e1973ae5cd05f5aedd345178f52f8e6a"

    scheduler = {
        'ddpm': DDPMScheduler.from_pretrained(repo_id, subfolder="scheduler"),
        'ddim': DDIMScheduler.from_pretrained(repo_id, subfolder="scheduler"),
    }

    base_pipeline = DiffusionPipeline.from_pretrained(
            repo_id,
            scheduler=scheduler['ddpm'],
            revision=None,
            torch_dtype=torch.float16,
            safety_checker=None
    ).to('cuda')
    return base_pipeline


default_neg = "worst quality, low quality, medium quality, deleted, lowres, comic, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, jpeg artifacts, signature, watermark, username, blurry"


def generate_image(pipeline, prompt, steps, seed, nums, device='cuda',
                   negative_prompt=default_neg, size=(512, 512), guidance_scale=7.5):
    generator = torch.Generator(device=device).manual_seed(seed)
    images = []
    for _ in range(nums):
        image = pipeline(prompt, negative_prompt=negative_prompt, num_inference_steps=steps,
                         generator=generator,
                         #cross_attention_kwargs={"scale": scale},
                         guidance_scale=guidance_scale,
                         height=size[1], width=size[0]).images[0]
        images.append(image)
    return images

def debug():
    base_pipeline = init_pipe()
    prompt = "1girl"
    steps = 25
    seed = 231
    nums = 2
    images = generate_image(base_pipeline, prompt, steps, seed, nums)
    for im in images:
        print(im.size)
#debug()
