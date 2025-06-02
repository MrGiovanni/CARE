<div align="center">
    <img src="assets/fig_logo.png" alt="CARE Logo" width=200/>
<h2 align="center"><span style="color: blue;"><ins>C</ins></span>ompleteness-<span style="color: blue;"><ins>A</ins></span>ware <span style="color: blue;"><ins>R</ins></span>econstruction <span style="color: blue;"><ins>E</ins></span>nhancement</h2>
</div>
<div align="center">


![visitors](https://visitor-badge.laobi.icu/badge?page_id=MrGiovanni/CARE)
[![GitHub Repo stars](https://img.shields.io/github/stars/MrGiovanni/CARE?style=social)](https://github.com/MrGiovanni/CARE/stargazers)
<a href="https://twitter.com/bodymaps317">
        <img src="https://img.shields.io/twitter/follow/BodyMaps?style=social" alt="Follow on Twitter" />
</a><br/>
**Subscribe us: https://groups.google.com/u/2/g/bodymaps**  

</div>

CARE (Completeness-Aware Reconstruction Enhancement) is an open-source framework that plugs anatomy-aware intelligence into sparse-view CT reconstruction. It couples a suite of segmentation-derived metrics with a lightweight diffusion-based refinement module, enabling existing analytical, neural-rendering, and low-dose reconstruction methods to recover small organs and vessels that pixel-wise metrics routinely miss. By supervising models with clinically meaningful structural signals, CARE delivers large, consistent gains in diagnostic fidelity while remaining model-agnostic and easy to integrate into any reconstruction pipeline.

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

## Anatmoy-Aware CT Reconstruction Metrics
## 0. Train Anatomy Segmentator
## 1. Train Autoencoder Model
## 2. Train Diffusion Model
## 3. Train CARE Model


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