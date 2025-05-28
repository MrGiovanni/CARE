export SD_MODEL_NAME="stable-diffusion-v1-5/stable-diffusion-v1-5"
export FT_VAE_NAME="../STEP1-AutoEncoderModel/klvae/logs/vae_kl6_lr4_std/checkpoint-150000"
export TRAINED_UNET_NAME="../STEP2-DiffusionModel/logs/l2_cat_df4_noblur/checkpoint-50000"
export SEG_MODEL_NAME="../../../nnUNet/nnUNet_results/Dataset808_AbdomenAtlasF/nnUNetTrainer__nnUNetResEncUNetLPlans__2d"
DATASET_NAME=$1

EXP_NAME="AS3_$DATASET_NAME" # best if the same as shown in wandb
export TRAIN_DATA_DIR="../ReconstructionPipeline/BDMAP_O_$DATASET_NAME/" # Temporary FELIX path!!!!


accelerate launch --mixed_precision="no" train_text_to_image.py \
  --sd_model_name_or_path=$SD_MODEL_NAME \
  --finetuned_vae_name_or_path=$FT_VAE_NAME \
  --pretrained_unet_name_or_path=$TRAINED_UNET_NAME \
  --seg_model_path=$SEG_MODEL_NAME \
  --train_data_dir=$TRAIN_DATA_DIR \
  --resume_from_checkpoint="latest" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=2 \
  --dataloader_num_workers=1 \
  --max_train_steps=100_000 \
  --learning_rate=1e-05 \
  --max_grad_norm=1 \
  --lr_scheduler="constant" \
  --report_to=wandb \
  --validation_steps=1000 \
  --checkpointing_steps=1000 \
  --checkpoints_total_limit=1 \
  --validation_images ../ReconstructionPipeline/BDMAP_O_$DATASET_NAME/BDMAP_O0000001/ct.h5 ../ReconstructionPipeline/BDMAP_O_$DATASET_NAME/BDMAP_O0000002/ct.h5 \
  --validation_prompt 'An Arterial CT slice.' 'An Portal-venous CT slice.' \
  --output_dir="logs/$EXP_NAME"
