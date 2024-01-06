# python -m torch.distributed.launch --use_env --nodes=1 --nproc_per_node=8 

MODEL_ID=PixArt-alpha/PixArt-XL-2-512x512
DATA_PATH=data/diffusion_db_pixart_xl_2_512x512_2
SAVE_PATH=saved/dmd_pixart

accelerate launch train_dmd.py \
    --pretrained_teacher_model $MODEL_ID \
    --pretrained_vae_model_name_or_path madebyollin/taesd \
    --model_class transformer \
    --text_encoder_class t5 \
    --vae_class tiny \
    --dm_data_path data/diffusion_db_prompts.txt \
    --reg_data_path $DATA_PATH \
    --mixed_precision=no \
    --dm_batch_size=5 \
    --reg_batch_size=3 \
    --guidance_scale=4.5 \
    --max_train_steps=100000 \
    --validation_steps=10 \
    --gradient_accumulation_steps=4 \
    --kl_loss_weight=1.0 \
    --reg_loss_weight=0.25 \
    --train_fake_model \
    --learning_rate=1e-6 \
    --gradient_checkpointing \
    --output_dir=$SAVE_PATH \
    --dataloader_num_workers=8 \
    --checkpointing_steps=1000 --checkpoints_total_limit=3 \
    --resume_from_checkpoint=latest \
    --report_to=tensorboard \
    --seed=453645634 \
    --validation_prompt="a blue dog"
