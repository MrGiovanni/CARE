
accelerate launch train_klvae.py \
  --train_data_dir="/mnt/data/tlin67/Dataset_raw/FELIXtemp/FELIXh5" \
  --validation_images ../../../../Dataset_raw/FELIXtemp/FELIXh5/BDMAP_A0000001/ct.h5 ../../../../Dataset_raw/FELIXtemp/FELIXh5/BDMAP_V0000001/ct.h5 \
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
  --kl_weight=0 \
  --output_dir="vae_kl0_lr4_std"