import logging
import os
import shutil
from pathlib import Path

import accelerate
import diffusers
import piq
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import torch.utils.data
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers import (
    DDPMScheduler,
    UNet2DConditionModel,
    AutoencoderTiny,
)
from diffusers.optimization import get_scheduler
from diffusers.training_utils import compute_snr
from packaging import version
from transformers import AutoTokenizer, CLIPTextModel, BertModel
from PIL import Image
from dmd.args import parse_args
from dmd.data import cycle, TextDataset, RegressionDataset
from dmd.model import (
    distribution_matching_loss,
    encode_prompt,
    generate,
    prepare_latents,
    stopgrad,
)

logger = get_logger(__name__)


class MetricTracker:
    def __init__(self, n):
        self.n = n
        self.data = {}

    def update(self, kwargs):
        for key, value in kwargs.items():
            if key not in self.data:
                self.data[key] = []
            self.data[key].append(value)
            if len(self.data[key]) > self.n:
                self.data[key].pop(0)

    def mean(self, key=None):
        if key is None:
            means = {}
            for key, values in self.data.items():
                means[key] = sum(values) / len(values)
            return means
        values = self.data.get(key, [])
        if not values:
            return None
        return sum(values) / len(values)


def setup_training(args):
    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        # split_batches=True,  # It's important to set this to True when using webdataset to get the right number of steps for lr scheduling. If set to False, the number of steps will be devide by the number of processes assuming batches are multiplied by the number of processes
    )

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_config = dict(vars(args))
        accelerator.init_trackers(args.tracker_project_name, config=tracker_config)

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    return accelerator, logging_dir


def setup_model(args, accelerator, weight_dtype):
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_teacher_model, subfolder="scheduler")
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_teacher_model, subfolder="tokenizer", use_fast=False)
    text_encoder = BertModel.from_pretrained(args.pretrained_teacher_model, subfolder="text_encoder")
    vae = AutoencoderTiny.from_pretrained(args.pretrained_vae_model_name_or_path, subfolder="vae")
    real_unet = UNet2DConditionModel.from_pretrained(args.pretrained_teacher_model, subfolder="unet")

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    real_unet.requires_grad_(False)

    fake_unet = UNet2DConditionModel(**real_unet.config)
    fake_unet.load_state_dict(real_unet.state_dict(), strict=False)
    fake_unet.train()

    student_unet = UNet2DConditionModel(**real_unet.config)
    student_unet.load_state_dict(real_unet.state_dict(), strict=False)
    student_unet.train()

    # Move unet, vae and text_encoder to device and cast to weight_dtype
    # The VAE is in float32 to avoid NaN losses.
    vae.to(accelerator.device)
    text_encoder.to(accelerator.device, dtype=weight_dtype)

    # Move teacher_unet to device, optionally cast to weight_dtype
    real_unet.to(accelerator.device)
    fake_unet.to(accelerator.device)
    student_unet.to(accelerator.device)
    if args.cast_teacher_unet:
        real_unet.to(dtype=weight_dtype)

    if args.gradient_checkpointing:
        student_unet.enable_gradient_checkpointing()
        fake_unet.enable_gradient_checkpointing()

    return real_unet, fake_unet, student_unet, noise_scheduler, tokenizer, text_encoder, vae


def setup_model_saving(accelerator, student_unet):
    # 11. Handle saving and loading of checkpoints
    # `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process:
                student_unet.save_pretrained(os.path.join(output_dir, "student_unet"))

                for i, model in enumerate(models):
                    model.save_pretrained(os.path.join(output_dir, "unet"))

                    # make sure to pop weight so that corresponding model is not saved again
                    weights.pop()

        def load_model_hook(models, input_dir):
            load_model = UNet2DConditionModel.from_pretrained(os.path.join(input_dir, "student_unet"))
            student_unet.load_state_dict(load_model.state_dict())
            student_unet.to(accelerator.device)
            del load_model

            for i in range(len(models)):
                # pop models so that they are not loaded again
                model = models.pop()

                # load diffusers style into model
                load_model = UNet2DConditionModel.from_pretrained(input_dir, subfolder="unet")
                model.register_to_config(**load_model.config)

                model.load_state_dict(load_model.state_dict())
                del load_model

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)


def setup_optimizer_scheduler(args, fake_unet, student_unet):
    fake_optimizer = torch.optim.AdamW(
        fake_unet.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    student_optimizer = torch.optim.AdamW(
        student_unet.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    fake_lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=fake_optimizer,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=args.max_train_steps,
    )
    student_lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=student_optimizer,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=args.max_train_steps,
    )
    return fake_optimizer, student_optimizer, fake_lr_scheduler, student_lr_scheduler


def setup_dataloader(args):
    dataset = TextDataset()
    dm_dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=args.dm_batch_size, num_workers=args.dataloader_num_workers, pin_memory=True
    )
    dataset = RegressionDataset()
    reg_dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=args.reg_batch_size, num_workers=args.dataloader_num_workers, pin_memory=True
    )
    dm_dataloader = cycle(dm_dataloader)
    reg_dataloader = cycle(reg_dataloader)
    return dm_dataloader, reg_dataloader


def main(args):
    accelerator, logging_dir = setup_training(args)
    os.makedirs(logging_dir, exist_ok=True)
    file_handler = logging.FileHandler(os.path.join(logging_dir, "dmd.log"))
    logger.logger.addHandler(file_handler)

    # 10. Handle mixed precision and device placement
    # For mixed precision training we cast all non-trainable weigths to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    (real_unet, fake_unet, student_unet,
     noise_scheduler, tokenizer, text_encoder, vae) = setup_model(args, accelerator, weight_dtype)

    setup_model_saving(accelerator, student_unet)

    dm_dataloader, reg_dataloader = setup_dataloader(args)

    fake_optimizer, student_optimizer, fake_lr_scheduler, student_lr_scheduler = setup_optimizer_scheduler(args, fake_unet, student_unet)

    # Prepare everything with our `accelerator`.
    (fake_unet, student_unet,
     fake_optimizer, student_optimizer,
     fake_lr_scheduler, student_lr_scheduler,
     dm_dataloader, reg_dataloader) = accelerator.prepare(
         fake_unet, student_unet, fake_optimizer, student_optimizer,
         fake_lr_scheduler, student_lr_scheduler, dm_dataloader, reg_dataloader
    )

    # Train!
    total_reg_batch_size = args.reg_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    total_dm_batch_size = args.dm_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Instantaneous batch size per device (reg) = {args.reg_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) (reg) = {total_reg_batch_size}")
    logger.info(f"  Instantaneous batch size per device (reg) = {args.dm_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) (reg) = {total_dm_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run.")
            args.resume_from_checkpoint = None
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

    lpips = piq.LPIPS()
    tracker = MetricTracker(50)

    for step in range(args.max_train_steps):
        prompts = next(dm_dataloader)
        latents_ref, images_ref, prompts_ref = next(reg_dataloader)
        latents_ref = latents_ref.to(accelerator.device)
        images_ref = images_ref.to(accelerator.device)

        if args.gradient_checkpointing:
            accelerator.unwrap_model(fake_unet).disable_gradient_checkpointing()

        logs = {}
        # ------------ train student unet ------------- #

        loss_g = 0

        if args.reg_loss_weight > 0:
            prompt_ref_embeds = encode_prompt(prompts_ref, text_encoder, tokenizer)
            latents_ref_pred = generate(student_unet, noise_scheduler, latents_ref, prompt_ref_embeds)
            images_ref_pred = vae.decode(latents_ref_pred.to(vae.dtype) / vae.config.scaling_factor).sample
            images_ref_pred = (images_ref_pred / 2 + 0.5).clamp(0, 1)
            images_ref_pred = images_ref_pred.to(dtype=images_ref.dtype)
            loss_reg = lpips(images_ref, images_ref_pred)
            loss_g += loss_reg * args.reg_loss_weight

        if args.kl_loss_weight > 0:
            prompt_embeds = encode_prompt(prompts, text_encoder, tokenizer)
            latents = prepare_latents(accelerator.unwrap_model(student_unet), vae, batch_size=len(prompts), device=accelerator.device, dtype=weight_dtype)
            latents_pred = generate(student_unet, noise_scheduler, latents, prompt_embeds)

            if args.reg_loss_weight > 0:
                latents_pred = torch.cat([latents_pred, latents_ref_pred], dim=0)
                prompt_embeds = torch.cat([prompt_embeds, prompt_ref_embeds], dim=0)
                prompts = prompts_ref + prompts
            negative_prompt_embeds = encode_prompt([""]*len(prompts), text_encoder, tokenizer)
            loss_kl = distribution_matching_loss(real_unet, fake_unet, noise_scheduler,
                                                 latents_pred, prompt_embeds, negative_prompt_embeds, args)

            loss_g += loss_kl*args.kl_loss_weight

        accelerator.backward(loss_g)
        if accelerator.sync_gradients:
            accelerator.clip_grad_norm_(student_unet.parameters(), args.max_grad_norm)
        student_optimizer.step()
        student_lr_scheduler.step()
        student_optimizer.zero_grad(set_to_none=True)

        logs.update({'loss_g': loss_g.detach().item()})
        tracker.update({'loss_g': loss_g.detach().item()})

        # ------------ train fake unet ------------- #
        if args.train_fake_unet:
            if args.gradient_checkpointing:
                accelerator.unwrap_model(fake_unet).disable_gradient_checkpointing()

            latents = stopgrad(latents_pred)
            # Get the text embedding for conditioning
            encoder_hidden_states = stopgrad(prompt_embeds)

            # Sample noise that we'll add to the latents
            noise = torch.randn_like(latents)
            bsz = latents.shape[0]
            # Sample a random timestep for each image
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
            timesteps = timesteps.long()

            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            if noise_scheduler.config.prediction_type == "epsilon":
                target = noise
            elif noise_scheduler.config.prediction_type == "v_prediction":
                target = noise_scheduler.get_velocity(latents, noise, timesteps)
            else:
                raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

            # Predict the noise residual and compute loss
            model_pred = fake_unet(noisy_latents, timesteps, encoder_hidden_states).sample

            if args.snr_gamma is None:
                loss_d = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
            else:
                # Compute loss-weights as per Section 3.4 of https://arxiv.org/abs/2303.09556.
                # Since we predict the noise instead of x_0, the original formulation is slightly changed.
                # This is discussed in Section 4.2 of the same paper.
                snr = compute_snr(noise_scheduler, timesteps)
                if noise_scheduler.config.prediction_type == "v_prediction":
                    # Velocity objective requires that we add one to SNR values before we divide by them.
                    snr = snr + 1
                mse_loss_weights = (
                    torch.stack([snr, args.snr_gamma * torch.ones_like(timesteps)], dim=1).min(dim=1)[0] / snr
                )

                loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
                loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
                loss_d = loss.mean()

            accelerator.backward(loss_d)
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(fake_unet.parameters(), args.max_grad_norm)
            fake_optimizer.step()
            fake_lr_scheduler.step()
            fake_optimizer.zero_grad()

            logs.update({'loss_d': loss_d.detach().item()})
            tracker.update({'loss_d': loss_d.detach().item()})

        # Checks if the accelerator has performed an optimization step behind the scenes
        if accelerator.sync_gradients:
            global_step += 1

            if accelerator.is_main_process:
                if global_step % args.checkpointing_steps == 0:
                    # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                    if args.checkpoints_total_limit is not None:
                        checkpoints = os.listdir(args.output_dir)
                        checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                        checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                        # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                        if len(checkpoints) >= args.checkpoints_total_limit:
                            num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                            removing_checkpoints = checkpoints[0:num_to_remove]

                            logger.info(f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints")
                            logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                            for removing_checkpoint in removing_checkpoints:
                                removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                shutil.rmtree(removing_checkpoint)

                    save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    accelerator.save_state(save_path)
                    logger.info(f"Saved state to {save_path}")

                if global_step % args.validation_steps == 0:
                    log_validation(vae, student_unet, text_encoder, tokenizer, noise_scheduler, args, accelerator, weight_dtype, global_step, logging_dir)

        accelerator.log(logs, step=global_step)

        print_logs = {'gstep': global_step}
        print_logs.update(tracker.mean())
        logger.info(print_logs)

        if global_step >= args.max_train_steps:
            break

    # Create the pipeline using using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        student_unet = accelerator.unwrap_model(student_unet)
        student_unet.save_pretrained(os.path.join(args.output_dir, "student_unet"))
    accelerator.end_training()


def log_validation(vae, unet, text_encoder, tokenizer, noise_scheduler, args, accelerator, weight_dtype, step, logging_dir):
    logger.info("Running validation... ")

    unet = accelerator.unwrap_model(unet)

    if args.seed is None:
        generator = None
    else:
        generator = torch.Generator(device=accelerator.device).manual_seed(args.seed)

    image_dir = os.path.join(logging_dir, 'images')
    os.makedirs(image_dir, exist_ok=True)

    with torch.cuda.amp.autocast():
        latents = prepare_latents(accelerator.unwrap_model(unet), vae, batch_size=1, device=accelerator.device, dtype=weight_dtype, generator=generator)
        prompt_embeds = encode_prompt(args.validation_prompt, text_encoder, tokenizer)
        latents_pred = generate(unet, noise_scheduler, latents, prompt_embeds)
        images = vae.decode(latents_pred / vae.config.scaling_factor).sample
        images = (images / 2 + 0.5).clamp(0, 1)

        images = images[0].permute(1, 2, 0).detach().cpu().numpy()
        images = (images*255).astype('uint8')
        Image.fromarray(images).save(os.path.join(image_dir, f'{step}_{args.validation_prompt}.jpg'))


if __name__ == "__main__":
    args = parse_args()
    main(args)
