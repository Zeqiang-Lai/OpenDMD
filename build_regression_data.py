import os
import fire

import torch
from diffusers import DiffusionPipeline


def main(
    global_rank,
    caption_path,
    model_id='runwayml/stable-diffusion-v1-5',
    save_dir='data/regression',
):
    local_rank = global_rank % 8
    pipe = DiffusionPipeline.from_pretrained(model_id)
    pipe.to(device=f'cuda:{local_rank}', dtype=torch.float16)
    pipe.set_progress_bar_config(disable=True)

    image_dir = os.path.join(save_dir, 'images')
    os.makedirs(image_dir, exist_ok=True)
    latents_dir = os.path.join(save_dir, 'latents')
    os.makedirs(latents_dir, exist_ok=True)

    # load captions part
    with open(caption_path, 'r') as f:
        captions = f.readlines()
    if os.path.exists(save_dir, 'meta.json'):
        # filter out generated caption
        pass

    meta = []
    for i in range(1000):

        batch_size = 1
        num_images_per_prompt = 1
        height = 512
        width = 512
        num_inference_steps = 50
        num_channels_latents = pipe.unet.config.in_channels

        latents = pipe.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents, height, width,
            dtype=torch.float16, device=pipe.device,
            generator=torch.Generator('mlu').manual_seed(123)
        )

        image = pipe(prompt=prompt, latents=latents, num_inference_steps=num_inference_steps).images[0]

        image_path = os.path.join('images', f'{i}.jpg')
        latent_path = os.path.join('latents', f'{i}.pt')
        image.save(os.path.join(save_dir, image_path))
        torch.save(latents, os.path.join(save_dir, latent_path))


if __name__ == '__main__':
    fire.Fire(main)
