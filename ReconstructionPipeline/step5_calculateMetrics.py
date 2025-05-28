import os
import numpy as np
import nibabel as nib
import glob
import csv
from tqdm import tqdm
from multiprocessing import Pool
import argparse

from metric_utils import clDice
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from surface_distance import compute_surface_distances, compute_surface_dice_at_tolerance

"""
"labels": {
    0, "background": "背景",
    1, "aorta": "主动脉",
    2, "adrenal_gland_left": "左肾上腺",
    3, "adrenal_gland_right": "右肾上腺",
        4, "common_bile_duct": "胆总管",
    5, "celiac_aa": "腹腔动脉",
        6, "colon": "结肠",NOTE
    7, "duodenum": "十二指肠",NOTE
    8, "gall_bladder": "胆囊",
    9, "postcava": "下腔静脉",
    10, "kidney_left": "左肾",
    11, "kidney_right": "右肾",
    12, "liver": "肝脏",
    13, "pancreas": "胰腺",
        14, "pancreatic_duct": "胰管",
    15, "superior_mesenteric_artery": "肠系膜上动脉",
        16, "intestine": "肠道",NOTE
    17, "spleen": "脾脏",
    18, "stomach": "胃",
        19, "veins": "静脉",
    20, "renal_vein_left": "左肾静脉",
    21, "renal_vein_right": "右肾静脉",
        22, "cbd_stent": "胆总管支架",
    23, "pancreatic_pdac": "胰腺导管腺癌",
    24, "pancreatic_cyst": "胰腺囊肿",
    25, "pancreatic_pnet": "胰腺神经内分泌肿瘤"
}
"""


LARGE_LABEL = [12, 10, 11, 13, 17]   # DSC:      liver, kidney, pancreas, spleen
SMALL_LABEL = [8, 2, 3, 5, 7]        # NSD:      gallbladder, adrenal gland, celiac trunk (celiac_aa), duodenum.
VESSEL_LABEL = [1, 9, 15, 20, 21]    # clDice:   aorta, postcava, superior_mesenteric_artery, veins, renal vein
NON_PDAC_LABEL = [24, 25]            # NSD:      Cyst, PNET
PDAC_LABEL = [23]                    # NSD:      PDAC
TUBULAR_LABEL = [6, 16]                # clDice:      colon, intestine
    


def cal_dice(pred, true):
    intersection = np.sum(pred[true == 1]) * 2.0
    dice = intersection / (np.sum(pred) + np.sum(true)) if (np.sum(pred) + np.sum(true)) > 0 else np.nan
    return dice

def cal_dice_nsd(pred, truth, spacing_mm, tolerance=2):
    if np.sum(truth) == 0:
        return np.nan, np.nan
    dice = cal_dice(pred, truth)
    surface_distances = compute_surface_distances(truth.astype(bool), pred.astype(bool), spacing_mm=spacing_mm)
    nsd = compute_surface_dice_at_tolerance(surface_distances, tolerance)
    return dice, nsd

def process_case(case):
    case, CARE = case
    case_id = os.path.basename(case)
    gt_path = os.path.join(case, "gt.nii.gz")
    if CARE:
        pred_path = os.path.join(pred_mask_root, case_id, "pred_care.nii.gz")
    else:
        pred_path = os.path.join(pred_mask_root, case_id, "pred.nii.gz")
    
    if not os.path.exists(gt_path) or not os.path.exists(pred_path):
        raise FileNotFoundError("nnUNet predictions not complete")
    
    gt_nib = nib.load(gt_path)
    pred_nib = nib.load(pred_path)
    spacing = gt_nib.header.get_zooms()  # Extract voxel spacing
    
    gt = gt_nib.get_fdata().astype(np.uint8)
    pred = pred_nib.get_fdata().astype(np.uint8)
    # # if ((gt==23)|(gt==24)|(gt==25)).sum()==0:
    # # print((((gt==23)|(gt==24)|(gt==25))*((pred==23)|(pred==24)|(pred==25))).sum())
    # gt_tumor = ((gt==23)|(gt==24)|(gt==25))
    # assert gt_tumor.sum() > 0
    # pred_tumor = ((pred==23)|(pred==24)|(pred==25))
    # if (gt_tumor*pred_tumor).sum() > 0:
    #     flag=1
    # else:
    #     flag=0
    # return flag
    # print(((pred==23)|(pred==24)|(pred==25)).sum())


    
    large_nsd_list =        [clDice(pred == label, gt == label) * 100 for label in LARGE_LABEL]
    small_nsd_list =        [clDice(pred == label, gt == label) * 100 for label in SMALL_LABEL]
    vessel_cldice_list =    [clDice(pred == label, gt == label) * 100 for label in VESSEL_LABEL]
    nonpdac_nsd_list =      [clDice(pred == label, gt == label) * 100 for label in NON_PDAC_LABEL]
    pdac_nsd_list =         [clDice(pred == label, gt == label) * 100 for label in PDAC_LABEL]
    tubular_cldice_list =   [clDice(pred == label, gt == label) * 100 for label in TUBULAR_LABEL]
    # tubular_cldice_list =   [cal_dice_nsd(pred == label, gt == label, spacing)[1] * 100 for label in TUBULAR_LABEL]
    # tubular_cldice_list =   [clDice(pred == label, gt == label) * 100 for label in TUBULAR_LABEL]
    # print(dice_list + nsd_list + cldice_list)
    
    return [case_id] + large_nsd_list + small_nsd_list + vessel_cldice_list + nonpdac_nsd_list + pdac_nsd_list + tubular_cldice_list

def write_results_to_csv(results, output_csv):
    headers = ["case_name"] \
                + [f"large_nsd_{label}" for label in LARGE_LABEL] \
                + [f"small_nsd_{label}" for label in SMALL_LABEL] \
                + [f"vessel_cldice_{label}" for label in VESSEL_LABEL] \
                + [f"nonpdac_nsd_{label}" for label in NON_PDAC_LABEL] \
                + [f"pdac_nsd_{label}" for label in PDAC_LABEL] \
                + [f"tubular_cldice_{label}" for label in TUBULAR_LABEL]
                
    
    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for row in results:
            if row:
                writer.writerow([x for x in row])
    print(f"Metrics saved to {output_csv}")
            
                
def compute_stats(scores):
    scores = np.array(scores)
    median = np.nanmedian(scores)
    q1, q3 = np.nanpercentile(scores, [25, 75])
    mean = np.nanmean(scores)
    std = np.nanstd(scores)
    return median, q1, q3, mean, std

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate metrics")
    parser.add_argument('--pred_path', type=str, default='BDMAP_O_nerf_200', 
                        help="Path to prediction images folder")
    parser.add_argument('--CARE', action="store_true")  # TODO
    args = parser.parse_args()
    print(f"\033[31m{args.pred_path}\033[0m")
    gt_mask_root = "BDMAP_O"
    pred_mask_root = args.pred_path # "BDMAP_O_FDK_50"
    if args.CARE:
        output_csv = f"resultsCSVcldice/{pred_mask_root}_seg_care.csv"
    else:
        output_csv = f"resultsCSVcldice/{pred_mask_root}_seg.csv"

    
    
    ignore_ids = [""]
    cases = []
    if args.CARE:
        bdmap_id_test = pd.read_csv("../STEP3-CAREModel/splits/BDMAP_O_AV_meta_test.csv")["bdmap_id"].apply(lambda x: x[:-2]).tolist()
    for x in glob.glob(os.path.join(pred_mask_root, "BDMAP_O*")):
        case_name = x.split("/")[-1]
        if case_name not in ignore_ids:
            if not args.CARE or (args.CARE and case_name in bdmap_id_test):
                cases.append(x.replace(pred_mask_root, gt_mask_root))
    # cases = sorted(list(map(lambda x: x.replace(pred_mask_root, gt_mask_root), glob.glob(os.path.join(pred_mask_root, "BDMAP_O*")))))
    print("calculate on total of ", len(cases), "cases")
    results = []
    with ProcessPoolExecutor(max_workers=36) as exe:
        tasks = [exe.submit(process_case, (case, args.CARE)) for case in cases]
        for fut in tqdm(as_completed(tasks), total=len(tasks),
                        desc=args.pred_path, dynamic_ncols=True):
            res = fut.result()
            if res is None:
                continue
            results.append(res)
    write_results_to_csv(results, output_csv)

    # Compute statistics
    large_nsd_scores =   [row[
        1:len(LARGE_LABEL)+1
            ] for row in results if row] # NOTE: named `dice` but they are NSD now!!!
    small_nsd_scores =    [row[
        len(LARGE_LABEL)+1:len(LARGE_LABEL)+len(SMALL_LABEL)+1
            ] for row in results if row]
    vessel_cldice_scores = [row[
        len(LARGE_LABEL)+len(SMALL_LABEL)+1:len(LARGE_LABEL)+len(SMALL_LABEL)+len(VESSEL_LABEL)+1
            ] for row in results if row]
    tumor_scores =      [row[
        len(LARGE_LABEL)+len(SMALL_LABEL)+len(VESSEL_LABEL)+1:len(LARGE_LABEL)+len(SMALL_LABEL)+len(VESSEL_LABEL)+len(NON_PDAC_LABEL)+len(PDAC_LABEL)+1
            ] for row in results if row]
    tubular_scores =    [row[
        len(LARGE_LABEL)+len(SMALL_LABEL)+len(VESSEL_LABEL)+len(NON_PDAC_LABEL)+len(PDAC_LABEL)+1:len(LARGE_LABEL)+len(SMALL_LABEL)+len(VESSEL_LABEL)+len(NON_PDAC_LABEL)+len(PDAC_LABEL)+len(TUBULAR_LABEL)+1
            ] for row in results if row]
    # nonpdac_scores = [row[len(LARGE_LABEL)+len(SMALL_LABEL)+len(NON_PDAC_LABEL)+1:len(LARGE_LABEL)+len(SMALL_LABEL)+len(NON_PDAC_LABEL)+len(PDAC_LABEL)+1] for row in results if row]
    # pdac_scores = [row[len(LARGE_LABEL)+len(SMALL_LABEL)+len(NON_PDAC_LABEL)+len(PDAC_LABEL)+1:] for row in results if row]
    
    large_nsd_median, large_nsd_q1, large_nsd_q3, large_nsd_mean, large_nsd_std = compute_stats(np.concatenate(large_nsd_scores))
    small_nsd_median, small_nsd_q1, small_nsd_q3, small_nsd_mean, small_nsd_std = compute_stats(np.concatenate(small_nsd_scores))
    vessel_cldice_median, vessel_cldice_q1, vessel_cldice_q3, vessel_cldice_mean, vessel_cldice_std = compute_stats(np.concatenate(vessel_cldice_scores))
    tumor_median, tumor_q1, tumor_q3, tumor_mean, tumor_std = compute_stats(np.concatenate(tumor_scores))
    tubular_median, tubular_q1, tubular_q3, tubular_mean, tubular_std = compute_stats(np.concatenate(tubular_scores))
    # nonpdac_median, nonpdac_q1, nonpdac_q3, nonpdac_mean, nonpdac_std = compute_stats(np.concatenate(nonpdac_scores))
    # pdac_median, pdac_q1, pdac_q3, pdac_mean, pdac_std = compute_stats(np.concatenate(pdac_scores))
    
    print(f"{pred_mask_root}")
    print(f"\tlarge - Dice 3D - Median: {large_nsd_median:.1f}\\tiny{{~({large_nsd_q1:.1f},{large_nsd_q3:.1f})}}, Average: {large_nsd_mean:.1f}\\tiny{{~\pm {large_nsd_std:.1f}}}")
    print(f"\tsmall - NSD 3D - Median: {small_nsd_median:.1f}\\tiny{{~({small_nsd_q1:.1f},{small_nsd_q3:.1f})}}, Average: {small_nsd_mean:.1f}\\tiny{{~\pm {small_nsd_std:.1f}}}")
    print(f"\tvessel- clDice - Median: {vessel_cldice_median:.1f}\\tiny{{({vessel_cldice_q1:.1f},{vessel_cldice_q3:.1f})}}, Average: {vessel_cldice_mean:.1f}\\tiny{{~\pm {vessel_cldice_std:.1f}}}")
    print(f"\ttumor - NSD 3D - Median: {tumor_median:.1f}\\tiny{{~({tumor_q1:.1f},{tumor_q3:.1f})}}, Average: {tumor_mean:.1f}\\tiny{{\pm {tumor_std:.1f}}}")
    print(f"\ttubular - clDice 3D - Median: {tubular_median:.1f}\\tiny{{~({tubular_q1:.1f},{tubular_q3:.1f})}}, Average: {tubular_mean:.1f}\\tiny{{~\pm {tubular_std:.1f}}}")
    # print(f"\t{pred_mask_root} - non-PDAC - Median: {nonpdac_median:.1f}\\tiny{{({nonpdac_q1:.1f},{nonpdac_q3:.1f})}}, Average: {nonpdac_mean:.1f}\\tiny{{\pm {nonpdac_std:.1f}}}")
    # print(f"\t{pred_mask_root} - PDAC - Median: {pdac_median:.1f}\\tiny{{({pdac_q1:.1f},{pdac_q3:.1f})}}, Average: {pdac_mean:.1f}\\tiny{{\pm {pdac_std:.1f}}}")
    
