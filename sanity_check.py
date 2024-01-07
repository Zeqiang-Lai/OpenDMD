import torch
from diffusers import PixArtAlphaPipeline, StableDiffusionPipeline, AutoencoderTiny
from PIL import Image

from dmd.model import encode_prompt, generate_cfg, prepare_latents, generate


def test_sd():
    model_id = "runwayml/stable-diffusion-v1-5"
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipe.to(device="cuda")

    with torch.no_grad():
        latents = prepare_latents(
            pipe.unet,
            pipe.vae,
            batch_size=1,
            device=torch.device("cuda"),
            dtype=torch.float16,
            generator=torch.Generator().manual_seed(123),
        )
        prompt_embeds = encode_prompt(["a dog"], pipe.text_encoder, pipe.tokenizer)
        negative_prompt_embeds = encode_prompt([""], pipe.text_encoder, pipe.tokenizer)
        latents = generate_cfg(
            pipe.unet,
            pipe.scheduler,
            latents,
            prompt_embeds,
            negative_prompt_embeds,
            num_inference_steps=50,
        )
        images = pipe.vae.decode(latents / pipe.vae.config.scaling_factor, return_dict=False)[0]

    images = (images / 2 + 0.5).clamp(0, 1)
    images = images[0].permute(1, 2, 0).detach().cpu().numpy()
    images = (images * 255).astype("uint8")
    Image.fromarray(images).save("output.jpg")


def test_pixart():
    weight_dtype = torch.float32
    model_id = "PixArt-alpha/PixArt-XL-2-512x512"
    pipe = PixArtAlphaPipeline.from_pretrained(model_id, torch_dtype=weight_dtype)
    pipe.vae = AutoencoderTiny.from_pretrained("madebyollin/taesd", torch_dtype=weight_dtype)
    pipe.to(device="cuda")

    with torch.no_grad():
        latents = prepare_latents(
            pipe.transformer,
            pipe.vae,
            batch_size=1,
            device=torch.device("cuda"),
            dtype=weight_dtype,
            generator=torch.Generator().manual_seed(123),
        )
        prompt_embeds, attention_mask = encode_prompt(["a dog"], pipe.text_encoder, pipe.tokenizer)

        latents = generate(
            pipe.transformer,
            pipe.scheduler,
            latents,
            prompt_embeds,
            attention_mask,
        )
        images = pipe.vae.decode(latents / pipe.vae.config.scaling_factor, return_dict=False)[0]

    images = (images / 2 + 0.5).clamp(0, 1)
    images = images[0].permute(1, 2, 0).detach().cpu().numpy()
    images = (images * 255).astype("uint8")
    Image.fromarray(images).save("output.jpg")


def ref_sd():
    model_id = "lykon/dreamshaper-8"
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipe.to(device="cuda")
    images = pipe(
        "a dog",
        num_inference_steps=10,
        guidance_scale=0,
        generator=torch.Generator().manual_seed(123),
    ).images
    images[0].save("output.jpg")


def ref_pixart():
    model_id = "PixArt-alpha/PixArt-XL-2-512x512"
    pipe = PixArtAlphaPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipe.to(device="cuda")
    images = pipe(
        "a dog",
        num_inference_steps=10,
        guidance_scale=0,
        generator=torch.Generator().manual_seed(123),
    ).images
    images[0].save("output.jpg")


def log_validation(vae, model, text_encoder, tokenizer, noise_scheduler):
    device = torch.device("cuda")
    weight_dtype = torch.float16
    seed = 123
    validation_prompt = "a blue dog"

    generator = torch.Generator(device=device).manual_seed(seed)

    with torch.cuda.amp.autocast():
        latents = prepare_latents(model, vae, batch_size=1, device=device, dtype=weight_dtype, generator=generator)
        prompt_embeds = encode_prompt(validation_prompt, text_encoder, tokenizer)
        latents_pred = generate(model, noise_scheduler, latents, prompt_embeds)
        images = vae.decode(latents_pred / vae.config.scaling_factor).sample
        images = (images / 2 + 0.5).clamp(0, 1)

        images = images[0].permute(1, 2, 0).detach().cpu().numpy()
        images = (images * 255).astype("uint8")
        Image.fromarray(images).save("log.jpg")


def test_log_sd():
    model_id = "runwayml/stable-diffusion-v1-5"
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipe.to(device="cuda")

    log_validation(pipe.vae, pipe.unet, pipe.text_encoder, pipe.tokenizer, pipe.scheduler)


def test_log_pixart():
    model_id = "PixArt-alpha/PixArt-XL-2-512x512"
    pipe = PixArtAlphaPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipe.to(device="cuda")

    log_validation(pipe.vae, pipe.transformer, pipe.text_encoder, pipe.tokenizer, pipe.scheduler)


if __name__ == "__main__":
    # test_sd()
    test_pixart()
    # ref_sd()
    # ref_pixart()
    # test_log_sd()
    # test_log_pixart()
