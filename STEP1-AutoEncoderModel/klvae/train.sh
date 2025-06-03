export TRAIN_DATA_DIR="/path/to/dataseth5"

accelerate launch train_klvae.py \
  --train_data_dir=$TRAIN_DATA_DIR \
  --validation_images /path/to/dataseth5/BDMAP_A0000001/ct.h5 /path/to/dataseth5/BDMAP_V0000001/ct.h5 \
  --resume_from_checkpoint="latest" \
  --resolution=512 \
  --train_batch_size=2 \
  --gradient_accumulation_steps=1 \
  --dataloader_num_workers=2 \
  --report_to="wandb" \
  --pretrained_model_name_or_path="stable-diffusion-v1-5/stable-diffusion-v1-5" \
  --max_train_steps=1_000_000 \
  --vae_loss="l1" \
  --learning_rate=1e-4 \
  --validation_steps=1000 \
  --checkpointing_steps=1000 \
  --checkpoints_total_limit=5 \
  --kl_weight=1e-6 \
  --output_dir="klvae"