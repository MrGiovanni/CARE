"""Soft linking BDMAP_O cases"""
import glob
import os, sys, errno
from tqdm import tqdm
import pandas as pd
# import SimpleITK as sitk    # for IO validation
import multiprocessing as mp
import shutil

# mask_folder = "/projects/bodymaps/Data/mask_only/AbdomenAtlas1.1/AbdomenAtlas1.1"
# mask_folder = "/projects/bodymaps/Tianyu/dataset/AbdomenAtlas1.1"
ct_folder = "/mnt/bodymaps/image_only/AbdomenAtlasPro/AbdomenAtlasPro"
gt_folder = "/mnt/ccvl15/tlin67/Dataset_raw/reconFELIX/aarecon_combined_labels"

# tgt_path_mask = 'labelsTr'
tgt_path_ct = 'BDMAP_O'
# os.makedirs(tgt_path_mask, exist_ok=True)
os.makedirs(tgt_path_ct, exist_ok=True)

print(os.getcwd())

def symlink_ct_mask(mask_path):
    bdmap_id = mask_path.split("/")[-2]

    # mask_filename = os.path.join(mask_path, "combined_labels.nii.gz")
    # tgt_mask_filename = os.path.join(tgt_path_mask, f"{bdmap_id}.nii.gz")

    os.makedirs(os.path.join(tgt_path_ct, bdmap_id), exist_ok=True)

    ct_filename = os.path.join(ct_folder, bdmap_id, "ct.nii.gz")
    gt_filename = os.path.join(gt_folder, f"{bdmap_id}.nii.gz")
    tgt_ct_filename = os.path.join(tgt_path_ct, bdmap_id, "ct.nii.gz")
    tgt_gt_filename = os.path.join(tgt_path_ct, bdmap_id, "gt.nii.gz")

    # print(bdmap_id, mask_filename, tgt_mask_filename)
    print(bdmap_id, ct_filename, tgt_ct_filename, gt_filename, tgt_gt_filename)
    # os.symlink(mask_filename, tgt_mask_filename)
    try:
        shutil.copy(ct_filename, tgt_ct_filename)
    except:
        os.remove(tgt_ct_filename)
        shutil.copy(ct_filename, tgt_ct_filename)

    try:
        shutil.copy(gt_filename, tgt_gt_filename)
    except:
        os.remove(tgt_gt_filename)
        shutil.copy(gt_filename, tgt_gt_filename)

if __name__ == "__main__":
    num_processes = int(8)
    bdmap_list = []
    # NOTE: should be verified that all methods are the same in CCVL server first!!!
    sax_cases = sorted(list(set(map(lambda x:"_".join(x.split("/")[-1].split("-")[1].split("_")[0:2]), glob.glob(os.path.join("logs", "Lineformer", "FELIX*"))))))
    data = list(map(lambda x: os.path.join(ct_folder, x)+"/", sax_cases))
    print(sax_cases, len(sax_cases))
    print(data, len(data))

    with mp.Pool(processes=num_processes) as pool:
        results = list(tqdm(pool.imap(symlink_ct_mask, data), total=len(data)))

    print("process done.")
