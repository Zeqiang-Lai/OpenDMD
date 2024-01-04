import torch
import torch.nn.functional as F
from diffusers.utils.torch_utils import randn_tensor
from diffusers import Transformer2DModel
from accelerate.utils.other import extract_model_from_parallel
from transformers import T5EncoderModel


def prepare_latents(model, vae, batch_size, device, dtype, generator=None):
    num_channels_latents = model.config.in_channels
    # autoencodertiny only has encoder_block_out_channels attribute.
    # [1,1,1,1] is dummy block_out_channels, it corresponds to default scale factor 4
    block_out_channels = vae.config.get("block_out_channels", vae.config.get("encoder_block_out_channels", [1, 1, 1, 1]))
    vae_scale_factor = 2 ** (len(block_out_channels) - 1)
    height = model.config.sample_size * vae_scale_factor
    width = model.config.sample_size * vae_scale_factor
    shape = (batch_size, num_channels_latents, height // vae_scale_factor, width // vae_scale_factor)
    latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
    return latents


def mask_text_embeddings(emb, mask):
    if emb.shape[0] == 1:
        keep_index = mask.sum().item()
        return emb[:, :, :keep_index, :], keep_index
    else:
        masked_feature = emb * mask[:, None, :, None]
        return masked_feature, emb.shape[2]


def encode_prompt(captions, text_encoder, tokenizer):
    max_length = tokenizer.model_max_length

    is_pixart = isinstance(text_encoder, T5EncoderModel)
    if is_pixart:
        max_length = 120

    with torch.no_grad():
        text_inputs = tokenizer(captions, padding="max_length", max_length=max_length, truncation=True, return_tensors="pt")
        text_input_ids = text_inputs.input_ids
        prompt_embeds = text_encoder(text_input_ids.to(text_encoder.device))[0]
        attention_mask = text_inputs.attention_mask.to(text_encoder.device)

    if is_pixart:
        prompt_embeds = prompt_embeds.unsqueeze(1)
        masked_prompt_embeds, _ = mask_text_embeddings(prompt_embeds, attention_mask)
        masked_prompt_embeds = masked_prompt_embeds.squeeze(1)
        return masked_prompt_embeds

    return prompt_embeds


def encode_prompt_all(captions, text_encoder, tokenizer):
    max_length = tokenizer.model_max_length

    is_pixart = isinstance(text_encoder, T5EncoderModel)
    if is_pixart:
        max_length = 120

    with torch.no_grad():
        text_inputs = tokenizer(captions, padding="max_length", max_length=max_length, truncation=True, return_tensors="pt")
        text_input_ids = text_inputs.input_ids
        prompt_embeds = text_encoder(text_input_ids.to(text_encoder.device))[0]
        attention_mask = text_inputs.attention_mask.to(text_encoder.device)

    with torch.no_grad():
        text_inputs = tokenizer([""] * len(captions), padding="max_length", max_length=max_length, truncation=True, return_tensors="pt")
        text_input_ids = text_inputs.input_ids
        negative_prompt_embeds = text_encoder(text_input_ids.to(text_encoder.device))[0]

    if is_pixart:
        prompt_embeds = prompt_embeds.unsqueeze(1)
        masked_prompt_embeds, keep_indices = mask_text_embeddings(prompt_embeds, attention_mask)
        masked_prompt_embeds = masked_prompt_embeds.squeeze(1)
        masked_negative_prompt_embeds = negative_prompt_embeds[:, :keep_indices, :] if negative_prompt_embeds is not None else None
        return masked_prompt_embeds, masked_negative_prompt_embeds

    return prompt_embeds, negative_prompt_embeds


def generate_cfg(model, scheduler, latents, prompt_embeds, negative_prompt_embeds=None, num_inference_steps=1, guidance_scale=7.5):
    scheduler.set_timesteps(num_inference_steps, device=model.device)
    timesteps = scheduler.timesteps

    if negative_prompt_embeds is not None:
        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)

    for i, t in enumerate(timesteps):
        # expand the latents if we are doing classifier free guidance
        if negative_prompt_embeds is not None:
            latent_model_input = torch.cat([latents] * 2)
        else:
            latent_model_input = latents

        # predict the noise residual
        noise_pred = forward_model(model, latents=latent_model_input, timestep=t, prompt_embeds=prompt_embeds)

        # perform guidance
        if negative_prompt_embeds is not None:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        # compute the previous noisy sample x_t -> x_t-1
        latents = scheduler.step(noise_pred, t, latents, return_dict=False)[0]

    return latents


def isinstance_ddp(model, cls):
    model = extract_model_from_parallel(model)
    return isinstance(model, cls)


def forward_model(model, latents, timestep, prompt_embeds):
    added_cond_kwargs = None

    if isinstance_ddp(model, Transformer2DModel):
        added_cond_kwargs = {"resolution": None, "aspect_ratio": None}
        if extract_model_from_parallel(model).config.sample_size == 128:
            batch_size, _, height, width = latents.shape
            resolution = torch.tensor([height, width]).repeat(batch_size, 1)
            aspect_ratio = torch.tensor([float(height / width)]).repeat(batch_size, 1)
            resolution = resolution.to(dtype=prompt_embeds.dtype, device=prompt_embeds.device)
            aspect_ratio = aspect_ratio.to(dtype=prompt_embeds.dtype, device=prompt_embeds.device)
            added_cond_kwargs = {"resolution": resolution, "aspect_ratio": aspect_ratio}

    if isinstance_ddp(model, Transformer2DModel):
        timestep = timestep.expand(latents.shape[0])

    noise_pred = model(
        latents,
        timestep=timestep,
        encoder_hidden_states=prompt_embeds,
        added_cond_kwargs=added_cond_kwargs,
    ).sample

    if isinstance_ddp(model, Transformer2DModel):
        if extract_model_from_parallel(model).config.out_channels // 2 == latents.shape[1]:
            noise_pred = noise_pred.chunk(2, dim=1)[0]

    return noise_pred


def generate(model, scheduler, latents, prompt_embeds):
    t = torch.full((1,), scheduler.config.num_train_timesteps - 1, device=latents.device).long()
    noise_pred = forward_model(model, latents=latents, timestep=t, prompt_embeds=prompt_embeds)
    latents = eps_to_mu(scheduler, noise_pred, latents, t)
    return latents


def eps_to_mu(scheduler, model_output, sample, timesteps):
    alphas_cumprod = scheduler.alphas_cumprod.to(device=sample.device, dtype=sample.dtype)
    alpha_prod_t = alphas_cumprod[timesteps]
    while len(alpha_prod_t.shape) < len(sample.shape):
        alpha_prod_t = alpha_prod_t.unsqueeze(-1)
    beta_prod_t = 1 - alpha_prod_t
    pred_original_sample = (sample - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)
    return pred_original_sample


def distribution_matching_loss(real_model, fake_model, noise_scheduler, latents, prompt_embeds, negative_prompt_embeds, args):
    bsz = latents.shape[0]
    min_dm_step = int(noise_scheduler.config.num_train_timesteps * args.min_dm_step_ratio)
    max_dm_step = int(noise_scheduler.config.num_train_timesteps * args.max_dm_step_ratio)

    timestep = torch.randint(min_dm_step, max_dm_step, (bsz,), device=latents.device).long()
    noise = torch.randn_like(latents)
    noisy_latents = noise_scheduler.add_noise(latents, noise, timestep)

    with torch.no_grad():
        noise_pred = forward_model(fake_model, latents=noisy_latents, timestep=timestep, prompt_embeds=prompt_embeds.float())
        pred_fake_latents = eps_to_mu(noise_scheduler, noise_pred, noisy_latents, timestep)

        noisy_latents_input = torch.cat([noisy_latents] * 2)
        timestep_input = torch.cat([timestep] * 2)
        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
        noise_pred = forward_model(real_model, latents=noisy_latents_input, timestep=timestep_input, prompt_embeds=prompt_embeds.float())
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + args.guidance_scale * (noise_pred_text - noise_pred_uncond)

        pred_real_latents = eps_to_mu(noise_scheduler, noise_pred, noisy_latents, timestep)

    weighting_factor = torch.abs(latents - pred_real_latents).mean(dim=[1, 2, 3], keepdim=True)

    grad = (pred_fake_latents - pred_real_latents) / weighting_factor
    loss = F.mse_loss(latents, stopgrad(latents - grad))
    return loss


def stopgrad(x):
    return x.detach()
