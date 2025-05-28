"""
This script collects the experimental results from the logs folder and prints the median, average, and standard deviation of the PSNR 3D and SSIM 3D values.
Should make sure all the experiments are finished before running this script.
"""
import glob
import argparse
import os
from datetime import datetime
import torch
from tqdm import tqdm
import numpy as np
import nibabel as nib
import csv
from skimage.metrics import structural_similarity
import yaml
import pandas as pd

from concurrent.futures import ProcessPoolExecutor, as_completed

from metric_utils import get_ssim_3d, get_psnr_3d

def _gather_latest_eval(method_folder):
    """提取最新 eval 子目录路径（如果有则返回绝对路径，否则返回 None）。"""
    date_folders = [d for d in os.listdir(method_folder)
                    if os.path.isdir(os.path.join(method_folder, d))]
    if not date_folders:
        return None
    latest_date_folder = max(date_folders,
                             key=lambda d: datetime.strptime(d, '%Y_%m_%d_%H_%M_%S'))
    eval_folder = os.path.join(method_folder, latest_date_folder, "eval")
    if not os.path.exists(eval_folder):
        return None
    eval_sub = [d for d in os.listdir(eval_folder)
                if os.path.isdir(os.path.join(eval_folder, d))]
    if not eval_sub:
        return None
    latest_eval = max(eval_sub, key=lambda d: int(d.split('_')[1]))
    return os.path.join(eval_folder, latest_eval)

# ------------------------------ 子进程函数 --------------------------------- #
def process_folder(args):
    """对单个 method_folder 进行完整处理，返回 (case_id, ssim_3d, psnr_3d)。"""
    method_folder, nii_out_folder, CARE = args
    case_id = "_".join(method_folder.split("/")[-1].split("-")[-1].split("_")[:2])

    # 建好输出目录
    out_case_dir = os.path.join(nii_out_folder, case_id)
    os.makedirs(out_case_dir, exist_ok=True)

    eval_path = _gather_latest_eval(method_folder)
    # if not eval_path:                                # 找不到 eval 文件夹时返回 None
    #     return None

    # stats_file = os.path.join(eval_path, "stats.txt")
    # if not os.path.exists(stats_file):
    #     return None

    # 读取 stats.txt（我们不再信任里面的 psnr/ssim，而是自己重新算）
    if CARE:
        image_pred = nib.load(os.path.join(out_case_dir, "ct_care.nii.gz"))
        image_pred = image_pred.get_fdata() / 1000 / 2 + 0.5
    else:
        image_pred = nib.load(os.path.join(out_case_dir, "ct.nii.gz"))
        image_pred = image_pred.get_fdata() / 1000 / 2 + 0.5
        # image_pred = np.load(os.path.join(eval_path, "image_pred.npy"))

    # ground-truth
    nii_gt = nib.load(os.path.join("BDMAP_O", case_id, "ct.nii.gz"))
    gt_data = nii_gt.get_fdata() / 1000 / 2 + 0.5

    # 重新计算指标
    ssim_3d = get_ssim_3d(image_pred.clip(0, 1), gt_data.clip(0, 1)) * 100
    psnr_3d = get_psnr_3d(image_pred.clip(0, 1), gt_data.clip(0, 1))

    # 保存预测 nifti
    if not CARE:
        pred_save = ((image_pred * 2 - 1) * 1000).astype(np.int16).clip(-1000, 1000)
        nib.save(nib.Nifti1Image(pred_save, nii_gt.affine, nii_gt.header),
                os.path.join(out_case_dir, "ct.nii.gz"))

    return case_id, ssim_3d, psnr_3d
# --------------------------------------------------------------------------- #

def extract_experimental_results_mp(logs_folder, num_views, CARE=False, n_workers=None):
    methods = ["intratomo", 'nerf', 'tensorf', 'naf', 'FDK', "SART", "ASD_POCS", "Lineformer"]
    ignore_ids = ["lty"]

    for method in methods:
        # ------------------------ 收集所有 folder --------------------------- #
        # pat = os.path.join(logs_folder, method, f'FELIX-BDMAP_O*_{num_views}')
        pat = os.path.join(f'BDMAP_O_{method}_{num_views}', "*")
        method_folders = [p for p in glob.glob(pat)
                          if all(ig not in p for ig in ignore_ids)]

        if CARE:
            bdmap_id_test = pd.read_csv("../STEP3-CAREModel/splits/BDMAP_O_AV_meta_test.csv")["bdmap_id"].apply(lambda x: x[:-2]).tolist()
            method_folders_tmp = []
            # for idx, bdmap_id in enumerate(list(map(lambda x: "_".join(x.split("-")[-1].split("_")[:2]), method_folders))):
            for idx, bdmap_id in enumerate(list(map(lambda x: "_".join(x.split("/")[-1].split("_")[:2]), method_folders))):
                if bdmap_id in bdmap_id_test:
                    method_folders_tmp.append(method_folders[idx])
            method_folders = method_folders_tmp


        if not method_folders:
            continue

        nii_out_folder = f"BDMAP_O_{method}_{num_views}"
        os.makedirs(nii_out_folder, exist_ok=True)
        if CARE:
            output_csv = f"resultsCSV/{nii_out_folder}_pixel_care.csv"  
        else:
            output_csv = f"resultsCSV/{nii_out_folder}_pixel.csv"

        # -------------------------- 多进程处理 ------------------------------ #
        results, psnr_vals, ssim_vals = [], [], []
        with ProcessPoolExecutor(max_workers=n_workers) as exe:
            tasks = [exe.submit(process_folder,
                                (mf, nii_out_folder, CARE))
                     for mf in method_folders]

            for fut in tqdm(as_completed(tasks), total=len(tasks),
                            desc=f"{method}", dynamic_ncols=True):
                r = fut.result()
                if r is None:       # 跳过无效 case
                    continue
                case_id, ssim_3d, psnr_3d = r
                results.append([case_id, ssim_3d, psnr_3d])
                psnr_vals.append(psnr_3d)
                ssim_vals.append(ssim_3d)

        # -------------------------- 统计与输出 ------------------------------ #
        if psnr_vals:
            psnr_arr, ssim_arr = map(np.asarray, (psnr_vals, ssim_vals))
            def stats(arr):
                return (np.median(arr), np.percentile(arr, 25), np.percentile(arr, 75),
                        np.mean(arr), np.std(arr))
            p_med, p_q1, p_q3, p_avg, p_std = stats(psnr_arr)
            s_med, s_q1, s_q3, s_avg, s_std = stats(ssim_arr)

            print(f"{method} - PSNR 3D - "
                  f"Median: {p_med:.1f}\\tiny{{({p_q1:.1f},{p_q3:.1f})}}, "
                  f"Average: {p_avg:.1f}\\tiny{{\\pm {p_std:.1f}}}")
            print(f"{method} - SSIM 3D - "
                  f"Median: {s_med:.1f}\\tiny{{({s_q1:.1f},{s_q3:.1f})}}, "
                  f"Average: {s_avg:.1f}\\tiny{{\\pm {s_std:.1f}}}")

        write_results_to_csv(results, output_csv)

def write_results_to_csv(results, output_csv):
    headers = ["case_name"] + ["ssim_3d", "psnr_3d"]
                # + [f"dice_{label}" for label in LARGE_LABEL] \
                # + [f"nsd_{label}" for label in SMALL_LABEL] \
                # + [f"cldice_{label}" for label in VESSEL_LABEL] \
                # + [f"nsd_{label}" for label in NON_PDAC_LABEL] \
                # + [f"nsd_{label}" for label in PDAC_LABEL]
    
    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for row in results:
            if row:
                writer.writerow([x for x in row])



# --------------------------------------------------------------------------
# r2-gaussian 专用：单 case 处理逻辑
# --------------------------------------------------------------------------
def _process_r2_case(args):
    """对单个 r2_gaussian method_folder 完整计算并返回结果."""
    (method_folder, nii_out_folder, CARE) = args
    # import yaml, nibabel as nib, numpy as np

    # 解析 case_id
    case_id = "_".join(method_folder.split("/")[-1].split("-")[-1].split("_")[:2])

    out_case_dir = os.path.join(nii_out_folder, case_id)
    os.makedirs(out_case_dir, exist_ok=True)

    # # ----------------------- 找到最新 eval 子目录 ------------------------ #
    # eval_folder = os.path.join(method_folder, "eval")
    # if not os.path.exists(eval_folder):
    #     return None
    # subfolders = [d for d in os.listdir(eval_folder)
    #               if os.path.isdir(os.path.join(eval_folder, d))]
    # if not subfolders:
    #     return None
    # latest_eval = max(subfolders, key=lambda d: int(d.split('_')[1]))
    # stats_file = os.path.join(eval_folder, latest_eval, "eval3d.yml")
    # if not os.path.exists(stats_file):
    #     return None

    # ------------------------- 读取指标 + 重算 --------------------------- #
    # with open(stats_file, "r") as f:
    #     stats = yaml.safe_load(f)
    # try:
    #     psnr_3d = float(stats["psnr_3d"])
    #     ssim_3d = float(stats["ssim_3d"]) * 100
    # except Exception:
    #     raise RuntimeError(f"读取 {stats_file} 失败！")


    if CARE:
        image_pred_raw = nib.load(os.path.join(out_case_dir, "ct_care.nii.gz"))
        image_pred_raw = image_pred_raw.get_fdata() / 1000 / 2 + 0.5
    else:
        image_pred_raw = nib.load(os.path.join(out_case_dir, "ct.nii.gz"))
        image_pred_raw = image_pred_raw.get_fdata() / 1000 / 2 + 0.5
        # image_pred_path = os.path.join(method_folder,
        #                             "point_cloud", "iteration_30000", "vol_pred.npy")
        # image_pred_raw = np.load(image_pred_path)

    # ground-truth
    nii_gt = nib.load(os.path.join("BDMAP_O", case_id, "ct.nii.gz"))
    gt_data = nii_gt.get_fdata() / 1000 / 2 + 0.5

    # 重新计算更可信的指标
    ssim_3d = get_ssim_3d(image_pred_raw.clip(0, 1), gt_data.clip(0, 1)) * 100
    psnr_3d = get_psnr_3d(image_pred_raw.clip(0, 1), gt_data.clip(0, 1))

    # 如需保存预测 NIfTI，可取消注释
    if not CARE:
        pred_save = np.clip(((image_pred_raw * 2 - 1) * 1000).astype(np.int16),
                            -1000, 1000)
        nib.save(nib.Nifti1Image(pred_save, nii_gt.affine, nii_gt.header),
                os.path.join(out_case_dir, "ct.nii.gz"))

    return case_id, ssim_3d, psnr_3d


# --------------------------------------------------------------------------
# 并行入口函数
# --------------------------------------------------------------------------
def extract_experimental_results_r2_gaussian_mp(logs_folder, num_views,
                                                CARE=False, n_workers=None):
    """
    r2_gaussian 版本的多进程计算函数。
    与 extract_experimental_results_mp 并存，不会存在重名冲突。
    """
    import glob, numpy as np
    from tqdm.auto import tqdm
    from concurrent.futures import ProcessPoolExecutor, as_completed

    methods = ["r2_gaussian"]
    ignore_ids = ["lty"]

    for method in methods:
        # pattern = os.path.join(logs_folder, method, "BDMAP_O",
        #                        f"cone_ntrain_{num_views}_angle_180",
        #                        f"FELIX-BDMAP_O*_{num_views}.pickle")
        pattern = os.path.join(f'BDMAP_O_{method}_{num_views}', "*")
        method_folders = [p for p in glob.glob(pattern)
                          if all(ig not in p for ig in ignore_ids)]

        if CARE:
            bdmap_id_test = pd.read_csv("../STEP3-CAREModel/splits/BDMAP_O_AV_meta_test.csv")["bdmap_id"].apply(lambda x: x[:-2]).tolist()
            method_folders_tmp = []
            # for idx, bdmap_id in enumerate(list(map(lambda x: "_".join(x.split("-")[-1].split("_")[:2]), method_folders))):
            for idx, bdmap_id in enumerate(list(map(lambda x: "_".join(x.split("/")[-1].split("_")[:2]), method_folders))):
                if bdmap_id in bdmap_id_test:
                    method_folders_tmp.append(method_folders[idx])
            method_folders = method_folders_tmp

        if not method_folders:
            continue

        nii_out_folder = f"BDMAP_O_{method}_{num_views}"
        os.makedirs(nii_out_folder, exist_ok=True)
        if CARE:
            output_csv = f"resultsCSV/{nii_out_folder}_pixel_care.csv"  # TODO
        else:
            output_csv = f"resultsCSV/{nii_out_folder}_pixel.csv"

        results, psnr_vals, ssim_vals = [], [], []
        with ProcessPoolExecutor(max_workers=n_workers) as exe:
            tasks = [exe.submit(_process_r2_case, (mf, nii_out_folder, CARE))
                     for mf in method_folders]

            for fut in tqdm(as_completed(tasks), total=len(tasks),
                            desc=method, dynamic_ncols=True):
                res = fut.result()
                if res is None:
                    continue
                cid, ssim, psnr = res
                results.append([cid, ssim, psnr])
                psnr_vals.append(psnr)
                ssim_vals.append(ssim)

        # ------------------------ 统计输出 ------------------------ #
        if psnr_vals:
            arr_p, arr_s = map(np.asarray, (psnr_vals, ssim_vals))
            def _stat(a):  # median, q1, q3, avg, std
                return (np.median(a), np.percentile(a, 25),
                        np.percentile(a, 75), np.mean(a), np.std(a))
            p_med, p_q1, p_q3, p_avg, p_std = _stat(arr_p)
            s_med, s_q1, s_q3, s_avg, s_std = _stat(arr_s)
            print(f"{method} - PSNR3D  Median {p_med:.1f}\\tiny{{({p_q1:.1f},{p_q3:.1f})}}  "
                  f"Avg {p_avg:.1f}\\tiny{{\\pm {p_std:.1f}}}")
            print(f"{method} - SSIM3D  Median {s_med:.1f}\\tiny{{({s_q1:.1f},{s_q3:.1f})}}  "
                  f"Avg {s_avg:.1f}\\tiny{{\\pm {s_std:.1f}}}")

        write_results_to_csv(results, output_csv)

if __name__ == "__main__":
    logs_folder = 'logs'  # Replace with the actual path to the logs folder

    parser = argparse.ArgumentParser(description="Calculate metrics")
    parser.add_argument('--num_views', type=int, default=50)
    parser.add_argument('--CARE', action="store_true")  # TODO
    args = parser.parse_args()

    num_views = args.num_views  # Replace with the actual number of views to filter

    # NOTE: single core (deprecated)
    # extract_experimental_results(logs_folder, num_views, args.CARE)
    # extract_experimental_results_r2_gaussian(logs_folder, num_views, args.CARE)

    # NOTE: multi-processing
    extract_experimental_results_mp(logs_folder, num_views, args.CARE, n_workers=32)
    extract_experimental_results_r2_gaussian_mp(logs_folder, num_views, args.CARE, n_workers=32)

