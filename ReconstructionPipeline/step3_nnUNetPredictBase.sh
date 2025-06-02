export CKPT_PATH=/projects/bodymaps/Tianyu/AnatomyAwareRecon/ReconstructionPipeline/AnatomySegmentator

python step3_nnUNetPredict.py --pth ./BDMAP_O --checkpoint $CKPT_PATH
python step3_nnUNetPredict.py --pth ./BDMAP_O_Lineformer_50 --checkpoint $CKPT_PATH
python step3_nnUNetPredict.py --pth ./BDMAP_O_nerf_50 --checkpoint $CKPT_PATH
python step3_nnUNetPredict.py --pth ./BDMAP_O_FDK_50 --checkpoint $CKPT_PATH
python step3_nnUNetPredict.py --pth ./BDMAP_O_naf_50 --checkpoint $CKPT_PATH
python step3_nnUNetPredict.py --pth ./BDMAP_O_tensorf_50 --checkpoint $CKPT_PATH
python step3_nnUNetPredict.py --pth ./BDMAP_O_ASD_POCS_50 --checkpoint $CKPT_PATH
python step3_nnUNetPredict.py --pth ./BDMAP_O_intratomo_50 --checkpoint $CKPT_PATH
python step3_nnUNetPredict.py --pth ./BDMAP_O_SART_50 --checkpoint $CKPT_PATH
python step3_nnUNetPredict.py --pth ./BDMAP_O_r2_gaussian_50 --checkpoint $CKPT_PATH
