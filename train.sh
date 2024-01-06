# python -m torch.distributed.launch --use_env --nodes=1 --nproc_per_node=8 

MODEL_ID=runwayml/stable-diffusion-v1-5
DATA_PATH=data/diffusion_db_runwayml_stable-diffusion-v1-5
SAVE_PATH=saved/dmd_sd15

# MODEL_ID=lykon/dreamshaper-8
# DATA_PATH=data/diffusion_db_lykon_dreamshaper_8
# SAVE_PATH=saved/dmd_dreamshaper-8

accelerate launch train_dmd.py \
    --pretrained_teacher_model $MODEL_ID  \
    --pretrained_vae_model_name_or_path madebyollin/taesd \
    --dm_data_path data/diffusion_db_prompts.txt \
    --reg_data_path $DATA_PATH \
    --mixed_precision=fp16 \
    --dm_batch_size=16 \
    --reg_batch_size=8 \
    --max_train_steps=100000 \
    --validation_steps=10 \
    --gradient_accumulation_steps=1 \
    --kl_loss_weight=1.0 \
    --reg_loss_weight=0.0 \
    --train_fake_model \
    --learning_rate=4e-5 \
    --gradient_checkpointing \
    --output_dir=$SAVE_PATH \
    --dataloader_num_workers=8 \
    --checkpointing_steps=1000 --checkpoints_total_limit=3 \
    --resume_from_checkpoint=latest \
    --report_to=tensorboard \
    --seed=453645634 \
    --validation_prompt="a blue dog"
