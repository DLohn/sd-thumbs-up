import os
import json
import torch
import torch.nn.functional as F
import torchvision
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, StableDiffusionPipeline, UniPCMultistepScheduler

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def pad_or_crop_to_target(img, target_height=100, target_width=100):
    # Get original dimensions
    _, height, width = img.shape
    
    # If either dimension of the image is larger than the target size, crop to the target size
    if height > target_height or width > target_width:
        start_height = (height - target_height) // 2
        start_width = (width - target_width) // 2
        img = img[:, start_height:start_height + target_height, start_width:start_width + target_width]
    # If the dimensions are smaller, pad to the target size
    else:
        padding_left = (target_width - width) // 2
        padding_right = target_width - width - padding_left
        padding_top = (target_height - height) // 2
        padding_bottom = target_height - height - padding_top

        # Apply padding
        img = F.pad(img, (padding_left, padding_right, padding_top, padding_bottom), value=0)

    return img

def rand_example_only_hands(dataset_json, hw, n=-1, seed=None):
    generator = torch.Generator()
    if seed is None:
        generator.seed()
    else:
        generator.manual_seed(seed)
    rands = torch.rand(3, generator=generator, device=generator.device).tolist()
    rands_n = torch.randn(1, generator=generator, device=generator.device).tolist()
    if n < 0:
        n = int(rands[0] * (len(dataset_json)-1))
    selected = dataset_json[n]
    img = torch.tensor(np.array(Image.open(os.path.join(dataset_json[-1], selected['name'])))).to(dtype=torch.float32, device=device) / 255.0
    img = img.permute(2, 0, 1)
    img = pad_or_crop_to_target(img, hw[0], hw[1])
    rotangle = (rands_n[0] - 0.5) * 5.0
    offseth = (selected['offset'][0] * hw[0]) + ((rands[1] - 0.5) * (16.0 * (1.0 / 256.0) * hw[0]))
    offsetw = (selected['offset'][1] * hw[1]) + ((rands[2] - 0.5) * (16.0 * (1.0 / 256.0) * hw[1]))
    img = torchvision.transforms.functional.affine(img, rotangle, (offsetw, offseth), 1.0, 0.0, torchvision.transforms.InterpolationMode.NEAREST)
    #plt.imshow((img.cpu().numpy() * 255.0).astype(np.uint8))
    #plt.title(selected['orig_name'])
    #plt.show()
    return img

def rand_example(dataset_json, hw, n=-1, seed=None):
    generator = torch.Generator()
    if seed is None:
        generator.seed()
    else:
        generator.manual_seed(seed)
    rands = torch.rand(3, generator=generator, device=generator.device).tolist()
    rands_n = torch.randn(1, generator=generator, device=generator.device).tolist()
    if n < 0:
        n = int(rands[0] * (len(dataset_json)-1))
    selected = dataset_json[n]
    img = torch.tensor(np.array(Image.open(os.path.join(dataset_json[-1], selected['name'])))).to(dtype=torch.float32, device=device) / 255.0
    img_pose = torch.tensor(np.array(Image.open(os.path.join(dataset_json[-1], selected['name_pose'])))).to(dtype=torch.float32, device=device) / 255.0
    img = torch.cat([img, img_pose], dim=-1)

    img = img.permute(2, 0, 1)
    img = pad_or_crop_to_target(img, hw[0], hw[1])
    rotangle = (rands_n[0] - 0.5) * 5.0
    offseth = (selected['offset'][0] * hw[0]) + ((rands[1] - 0.5) * (16.0 * (1.0 / 256.0) * hw[0]))
    offsetw = (selected['offset'][1] * hw[1]) + ((rands[2] - 0.5) * (16.0 * (1.0 / 256.0) * hw[1]))
    img = torchvision.transforms.functional.affine(img, rotangle, (offsetw, offseth), 1.0, 0.0, torchvision.transforms.InterpolationMode.NEAREST)
    return img

def decompose_to_rows_cols(n, ratio=1.778):
    rows = round(np.sqrt(n / ratio))
    cols = np.ceil(n / rows)
    return int(rows), int(cols)

def display_images(images, n=None, m=None):

    if n is None or m is None:
        n, m = decompose_to_rows_cols(len(images))
    _, axes = plt.subplots(n, m, figsize=(6, 6))
    if n == 1 and m == 1:
        axes = np.array([axes])
    axes = axes.ravel()  # flatting axes for easy iterating
    for i in np.arange(0, n * m):
        if i < len(images):
            axes[i].imshow(images[i])
        axes[i].axis('off')  # to hide axis labels
    plt.subplots_adjust(wspace=0.1)
    plt.show()

def run_model(data_path, sd_prompt, sd_prompt_negative):
    json_path = os.path.join(data_path, 'meta.json')
    if not os.path.exists(json_path):
        raise ValueError('Could not find dataset!')
    with open(json_path) as f:
        dataset = json.load(f)
    dataset.append(data_path)

    use_ext = False
    activate_ext = [
        'thumbs up',
        'thumb up'
    ]
    for key in activate_ext:
        if key in sd_prompt.lower():
            use_ext = True

    enable_extention = True
    use_ext = use_ext and enable_extention

    sd_pipe = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16
    ).to(device)
    sd_pipe.scheduler = UniPCMultistepScheduler.from_config(sd_pipe.scheduler.config)

    if not use_ext:
        pipe = sd_pipe
        sd_out = pipe(sd_prompt, negative_prompt=sd_prompt_negative, num_inference_steps=20)
    else:
        cnet_id = "MakiPan/controlnet-encoded-hands-130k"
        controlnet = ControlNetModel.from_pretrained(cnet_id, torch_dtype=torch.float16).to(device)
        controlnet_pose = ControlNetModel.from_pretrained(
            "lllyasviel/sd-controlnet-openpose", torch_dtype=torch.float16
        )
        pipe = StableDiffusionControlNetPipeline(
            sd_pipe.vae,
            sd_pipe.text_encoder,
            sd_pipe.tokenizer,
            sd_pipe.unet,
            [controlnet, controlnet_pose],
            sd_pipe.scheduler,
            sd_pipe.safety_checker,
            sd_pipe.feature_extractor,
            False
        ).to(device)
        hw = [pipe.unet.config.sample_size * pipe.vae_scale_factor] * 2
        cond_image = rand_example(dataset, hw)
        cond_hands = cond_image[:3]
        cond_pose = cond_image[3:]

        if sd_prompt_negative is None or sd_prompt_negative == '':
            sd_prompt_negative = "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, grainy, blurry"

        sd_out = pipe(sd_prompt, image=[cond_hands[None, ...], cond_pose[None, ...]], negative_prompt=sd_prompt_negative, num_inference_steps=20)
        sd_out.images.append(Image.fromarray((cond_hands.permute(1,2,0) * 255.0).detach().cpu().numpy().astype(np.uint8)))
        sd_out.images.append(Image.fromarray((cond_pose.permute(1,2,0) * 255.0).detach().cpu().numpy().astype(np.uint8)))
    
    display_images(sd_out.images)





if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('data_in', type=str)
    parser.add_argument('--sd_prompt', type=str, default='A young man smiling and giving a thumbs up')
    parser.add_argument('--sd_nprompt', type=str, default=None)
    args = parser.parse_args()

    run_model(args.data_in, args.sd_prompt, args.sd_nprompt)

