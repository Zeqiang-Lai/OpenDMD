# python -m torch.distributed.launch --use_env --nodes=1 --nproc_per_node=8 
python train_dmd.py \
    --pretrained_teacher_model runwayml/stable-diffusion-v1-5 \
    --pretrained_vae_model_name_or_path madebyollin/taesd \
    --dm_data_path diffusion_db_prompts.txt \
    --reg_data_path data/diffusion_db_lykon_dreamshaper_8 \
    --mixed_precision=fp16 \
    --dm_batch_size=24 \
    --reg_batch_size=12 \
    --max_train_steps=100000 \
    --validation_steps=100 \
    --gradient_accumulation_steps=1 \
    --kl_loss_weight=1.0 \
    --reg_loss_weight=0.25 \
    --train_fake_unet \
    --learning_rate=1e-5 \
    --gradient_checkpointing \
    --output_dir="saved/dmd" \
    --dataloader_num_workers=8 \
    --checkpointing_steps=1000 --checkpoints_total_limit=3 \
    --resume_from_checkpoint=latest \
    --report_to=tensorboard \
    --seed=453645634 \
    --validation_prompt="a blue dog"
