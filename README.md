<div align="center">
    <img src="assets/fig_logo.png" alt="CARE Logo" width=200/>
<h2 align="center"><span style="color: blue;"><ins>C</ins></span>ompleteness-<span style="color: blue;"><ins>A</ins></span>ware <span style="color: blue;"><ins>R</ins></span>econstruction <span style="color: blue;"><ins>E</ins></span>nhancement</h2>
</div>
<div align="center">


<!-- ![visitors](https://visitor-badge.laobi.icu/badge?page_id=MrGiovanni/CARE) -->
[![GitHub Repo stars](https://img.shields.io/github/stars/MrGiovanni/CARE?style=social)](https://github.com/MrGiovanni/CARE/stargazers)
<a href="https://twitter.com/bodymaps317">
        <img src="https://img.shields.io/twitter/follow/BodyMaps?style=social" alt="Follow on Twitter" />
</a><br/>
**Subscribe us: https://groups.google.com/u/2/g/bodymaps**  

</div>

The Completeness-Aware Reconstruction Enhancement (CARE) framework addresses a critical gap in sparse-view Computed Tomography (CT) reconstruction by shifting the evaluation from traditional pixel-wise metrics to anatomy-aware metrics derived from automated structural segmentation. By incorporating segmentation-informed losses into latent diffusion models, CARE significantly improves the reconstruction fidelity of clinically relevant anatomical structures, ensuring that critical diagnostic features are preserved even under highly limited view conditions.

## Paper
<b>Are Pixel-Wise Metrics Reliable for Sparse-View Computed Tomography Reconstruction?</b> <br/>
[Tianyu Lin](https://lin-tianyu.github.io/)<sup>1</sup>, 
[Xinran Li](https://openreview.net/profile?id=~Xinran_Li7)<sup>1</sup>, 
[Chuntung Zhuang](https://openreview.net/profile?id=~Chuntung_Zhuang1)<sup>1</sup>, 
[Qi Chen](https://scholar.google.com/citations?user=4Q5gs2MAAAAJ&hl=en)<sup>1</sup>, 
[Yuanhao Cai](https://caiyuanhao1998.github.io/)<sup>1</sup>, 
[Kai Ding](https://scholar.google.com/citations?user=OvpsAYgAAAAJ&hl=en&oi=ao)<sup>2</sup>, 
[Alan L. Yuille](https://www.cs.jhu.edu/~ayuille/)<sup>1</sup> and
[Zongwei Zhou](https://www.zongweiz.com/)<sup>1,*</sup> <br/>
<sup>1 </sup>Johns Hopkins University,  <br/>
<sup>2 </sup>Johns Hopkins Medicine  <br/>
<!-- CVPR, 2024 <br/> -->
<a href=''><img src='https://img.shields.io/badge/Paper-PDF-purple'></a>
<a href='https://github.com/MrGiovanni/CARE'><img src='https://img.shields.io/badge/GitHub-Code-blue'></a>

**We have documented common questions for the paper in [Frequently Asked Questions (FAQ)](documents/FAQ.md).**


## Installation
Create an conda environement via:
```bash
conda create -n care python=3.11 -y
conda activate care
```
Then install all requirements using:
```bash
pip install -r requirements.txt
```
We have documented detailed steps to help [prepare for downloading model checkpoints](documents/DOWNLOAD.md).

## Anatmoy-Aware CT Reconstruction Metrics
> [!NOTE]
> The following script is designed for the nine reconstruction methods mentioned in the paper: three traditional reconstruction methods (FDK, SART, ASD-POCS), five NeRF-based reconstruction methods (InTomo, NeRF, TensoRF, NAF, SAX-NeRF) using the [SAX-NeRF Repo](https://github.com/caiyuanhao1998/SAX-NeRF), and a Gaussian-Spaltting-based method R2-GS based on its own [R2-GS Repo](https://github.com/Ruyi-Zha/r2_gaussian). 

Firstly, Based on the CT reconstruction results from [SAX-NeRF Repo](https://github.com/caiyuanhao1998/SAX-NeRF) and [GitHub Repo](https://github.com/Ruyi-Zha/r2_gaussian), please use the provided script to format the dataset
```bash
cd ReconstructionPipeline/  # working directory
python step1_softlink_BDMAP_O.py    
```
The resulting dataset format is:
```bash
└── BDMAP_O/                      # ground truth folder
    └── case_0
        └── ct.nii.gz   # the ground truth CT of this case
└── BDMAP_O_methodName_numViews/  # reconstruction results folder
    └── case_0
        └── ct.nii.gz   # the reconstructed CT from `methodName` method with `numViews` X-rays
```
Next, download the Anatomy Segmentator checkpoint:
```bash
huggingface-cli download TianyuLin/CARE --allow-patterns="segmentator/segmentator3D/*" --to-local-dir="./AnatomySegmentator/"
export CKPT_PATH="./AnatomySegmentator"
```
Then, calculate the proposed anatomy-aware CT reconstruction metrics:
```bash
python -W ignore step2_extractAndpixelMetric.py # calculate pixel-wise metrics (SSIM and PSNR)
bash step3_nnUNetPredictBase.sh                 # inference anatomy segmentator
bash step5_calculateMetricsBase.sh              # calculate segmentation metrics
python -W ignore step6_read_result_csv_for_table1st2nd.py # print metrics in latex table format
```
Also, all calculated metrics would be under `resultsCSV` folder.

## 0. Train Anatomy Segmentator
The anatomy segmentator is built upon nnU-Net, please prepare the dataset first based on [nnU-Net's dataset format](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/dataset_format.md).
Then, train the model (`datasetNum` is your actual nnU-Net dataset number):
```bash
export nnUNet_raw="/path/to/Dataset_raw"
export nnUNet_preprocessed="/path/to/Dataset_preprocessed"
export nnUNet_results="/path/to/nnUNet/nnUNet_results"
DATASET_NUM=# your actual 3-digit nnU-Net dataset number
nnUNetv2_plan_and_preprocess -d $DATASET_NUM -pl nnUNetPlannerResEncL --verify_dataset_integrity
nnUNetv2_train $DATASET_NUM 3d_fullres all -p nnUNetResEncUNetLPlans
```

## 1. Train Autoencoder Model
```bash
cd ./STEP1-AutoEncoderModel/klvae
bash train.sh
```
## 2. Train Diffusion Model
```bash
cd ./STEP2-DiffusionModel
bash train.sh
```
## 3. Train CARE Model
```bash
cd ./STEP3-CAREModel
bash train.sh
```

## 4. Inference and Evaluation
Run the inference of CARE model via:
```bash
cd ./STEP3-CAREModel
bash inference.sh
```
Then, calculate the pixel-wise and anatomy-aware metrics:
```bash
python -W ignore step2_extractAndpixelMetric.py --CARE # calculate pixel-wise metrics (SSIM and PSNR)
bash step3_nnUNetPredictCARE.sh                 # inference anatomy segmentator
bash step5_calculateMetricsCARE.sh              # calculate segmentation metrics
python -W ignore step6_read_result_csv_for_table1st2nd.py # print metrics in latex table format
```


<!-- ## Citation
```
@inproceedings{lin2025are,
  title={Are Pixel-Wise Metrics Reliable for Sparse-View Computed Tomography Reconstruction?},
  author={Lin, Tianyu and Li, Xinran and Zhuang, Chuntung and Chen, Qi and Cai, Yuanhao and Ding, Kai and Yuille, Alan and Zhou, Zongwei},
  booktitle={},
  pages={},
  year={2025}
}
``` -->

## Acknowledgement

This work was supported by the Lustgarten Foundation for Pancreatic Cancer Research and the Patrick J. McGovern Foundation Award. We would like to thank the Johns Hopkins Research IT team in [IT@JH](https://researchit.jhu.edu/) for their support and infrastructure resources where some of these analyses were conducted; especially DISCOVERY HPC [DISCOVERY HPC](https://researchit.jhu.edu/research-hpc/). We thank Hamed Hooshangnejad, Heng Li, Wenxuan Li, and Guofeng Zhang for their helpful suggestions throughout the project.