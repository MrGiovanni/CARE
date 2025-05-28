export CKPT_PATH=/projects/bodymaps/Tianyu/AnatomyAwareRecon/ReconstructionPipeline/FlagshipModelCKPTv2/nnUNetTrainer__nnUNetPlans__3d_fullres

python step3_nnUNetPredict.py --CARE --pth ../ReconstructionPipeline/BDMAP_O_Lineformer_50 --checkpoint $CKPT_PATH
python step3_nnUNetPredict.py --CARE --pth ../ReconstructionPipeline/BDMAP_O_Lineformer_200 --checkpoint $CKPT_PATH

python step3_nnUNetPredict.py --CARE --pth ../ReconstructionPipeline/BDMAP_O_nerf_50 --checkpoint $CKPT_PATH
python step3_nnUNetPredict.py --CARE --pth ../ReconstructionPipeline/BDMAP_O_nerf_200 --checkpoint $CKPT_PATH

python step3_nnUNetPredict.py --CARE --pth ../ReconstructionPipeline/BDMAP_O_FDK_50 --checkpoint $CKPT_PATH
python step3_nnUNetPredict.py --CARE --pth ../ReconstructionPipeline/BDMAP_O_FDK_200 --checkpoint $CKPT_PATH

python step3_nnUNetPredict.py --CARE --pth ../ReconstructionPipeline/BDMAP_O_naf_50 --checkpoint $CKPT_PATH
python step3_nnUNetPredict.py --CARE --pth ../ReconstructionPipeline/BDMAP_O_naf_200 --checkpoint $CKPT_PATH

python step3_nnUNetPredict.py --CARE --pth ../ReconstructionPipeline/BDMAP_O_tensorf_50 --checkpoint $CKPT_PATH
python step3_nnUNetPredict.py --CARE --pth ../ReconstructionPipeline/BDMAP_O_tensorf_200 --checkpoint $CKPT_PATH

python step3_nnUNetPredict.py --CARE --pth ../ReconstructionPipeline/BDMAP_O_ASD_POCS_50 --checkpoint $CKPT_PATH
python step3_nnUNetPredict.py --CARE --pth ../ReconstructionPipeline/BDMAP_O_ASD_POCS_200 --checkpoint $CKPT_PATH

python step3_nnUNetPredict.py --CARE --pth ../ReconstructionPipeline/BDMAP_O_intratomo_50 --checkpoint $CKPT_PATH
python step3_nnUNetPredict.py --CARE --pth ../ReconstructionPipeline/BDMAP_O_intratomo_200 --checkpoint $CKPT_PATH

python step3_nnUNetPredict.py --CARE --pth ../ReconstructionPipeline/BDMAP_O_SART_50 --checkpoint $CKPT_PATH
python step3_nnUNetPredict.py --CARE --pth ../ReconstructionPipeline/BDMAP_O_SART_200 --checkpoint $CKPT_PATH

python step3_nnUNetPredict.py --CARE --pth ../ReconstructionPipeline/BDMAP_O_r2_gaussian_50 --checkpoint $CKPT_PATH
python step3_nnUNetPredict.py --CARE --pth ../ReconstructionPipeline/BDMAP_O_r2_gaussian_200 --checkpoint $CKPT_PATH
