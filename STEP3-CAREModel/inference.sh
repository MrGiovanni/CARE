export SD_MODEL_NAME="stable-diffusion-v1-5/stable-diffusion-v1-5"
export FT_VAE_NAME="../STEP1-AutoEncoderModel/klvae/logs/vae_kl6_lr4_std/checkpoint-150000"

# NOTE: define which model to inference
  # 0 training not finished
  # 1 training finished, not yet inference
  # 2 inference finished, not yet segmented
  # 3 segmentation finished
# Options: [
#       nerf_50         nerf_200,         33
#       FDK_50          FDK_200,          33
#       tensorf_50      tensorf_200,      33
#       naf_50          naf_200,          33
#       Lineformer_50   Lineformer_200    33
#       ASD_POCS_50     ASD_POCS_200      33
#       intratomo_50    intratomo_200     33
#       SART_50         SART_200          33  
#       r2_gaussian_50  r2_gaussian_200   33
# ]
DATASET_NAME=$1 #nerf_50
EXP_NAME="$DATASET_NAME" 
export TRAIN_DATA_DIR="../ReconstructionPipeline/BDMAP_O_$DATASET_NAME/" # Temporary FELIX path!!!!


# NOTE: change this into trained models w.r.t to a specific dataset settings
export TRAINED_UNET_NAME="logs/$EXP_NAME"
export CKPT_EPOCH="50000"
# export SEG_MODEL_NAME="/projects/bodymaps/Tianyu/nnunet/Dataset_results/Dataset101_AbdomenAtlas1-1/nnUNetTrainer__nnUNetResEncUNetPlans_80G__2d"



python -W ignore testEnhanceCTPipeline.py \
  --input_path "../ReconstructionPipeline/BDMAP_O_$DATASET_NAME" \
  --output_path "../ReconstructionPipeline/BDMAP_O_$DATASET_NAME" \
  --finetuned_vae_name_or_path=$FT_VAE_NAME \
  --finetuned_unet_name_or_path="$TRAINED_UNET_NAME/checkpoint-$CKPT_EPOCH" \
  --sd_model_name_or_path=$SD_MODEL_NAME 

