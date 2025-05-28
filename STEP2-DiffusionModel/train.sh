export SD_MODEL_NAME="stable-diffusion-v1-5/stable-diffusion-v1-5"
export FT_VAE_NAME="../STEP1-AutoEncoderModel/klvae/logs/vae_kl6_lr4_std/checkpoint-150000"
export TRAIN_DATA_DIR="/projects/bodymaps/Tianyu/nnunet/Dataset_preprocessed/Dataset807_BDMAP-AV/imagesTr" # Temporary FELIX path!!!!


accelerate launch --mixed_precision="no" train_text_to_image.py \
  --pretrained_model_name_or_path=$SD_MODEL_NAME \
  --finetuned_vae_name_or_path=$FT_VAE_NAME \
  --train_data_dir=$TRAIN_DATA_DIR \
  --resolution=512 \
  --train_batch_size=4 \
  --gradient_accumulation_steps=2 \
  --dataloader_num_workers=2 \
  --max_train_steps=1_000_000 \
  --learning_rate=1e-05 \
  --max_grad_norm=1 \
  --lr_scheduler="constant" \
  --report_to=wandb \
  --validation_steps=1000 \
  --checkpointing_steps=1000 \
  --checkpoints_total_limit=5 \
  --validation_images ../../../Dataset_raw/FELIXtemp/FELIXh5/BDMAP_A0000001/ct.h5 ../../../Dataset_raw/FELIXtemp/FELIXh5/BDMAP_V0000001/ct.h5 \
  --validation_prompt 'An arterial phase CT slice.' 'A portal-venous phase CT slice.' \
  --output_dir="logs/l2_cat_claheCanny"
