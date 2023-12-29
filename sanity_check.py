import torch
from diffusers import StableDiffusionPipeline
from PIL import Image

from dmd.model import encode_prompt, generate_cfg, prepare_latents


def test_gen():
    model_id = 'runwayml/stable-diffusion-v1-5'
    pipe = StableDiffusionPipeline.from_pretrained(model_id)
    pipe.to(device='cuda', dtype=torch.float16)

    with torch.no_grad():
        latents = prepare_latents(
            pipe.unet, pipe.vae,
            batch_size=1,
            device=torch.device('mlu'),
            dtype=torch.float16, generator=torch.Generator().manual_seed(123)
        )
        prompt_embeds = encode_prompt(['a blue dog'], pipe.text_encoder, pipe.tokenizer)
        negative_prompt_embeds = encode_prompt([''], pipe.text_encoder, pipe.tokenizer)
        latents = generate_cfg(pipe.unet, pipe.scheduler, latents, prompt_embeds, negative_prompt_embeds, num_inference_steps=50)
        images = pipe.vae.decode(latents / pipe.vae.config.scaling_factor, return_dict=False)[0]

    images = (images / 2 + 0.5).clamp(0, 1)
    images = images[0].permute(1, 2, 0).detach().cpu().numpy()
    images = (images*255).astype('uint8')
    Image.fromarray(images).save('output.jpg')


if __name__ == '__main__':
    test_gen()
