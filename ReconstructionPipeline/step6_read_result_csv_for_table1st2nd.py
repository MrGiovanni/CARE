import pandas as pd 
import numpy as np
import os
import math
from scipy.stats import ttest_ind, mannwhitneyu
from step5_calculateMetrics import (
    LARGE_LABEL,
    SMALL_LABEL,
    VESSEL_LABEL,
    NON_PDAC_LABEL,
    PDAC_LABEL,
    TUBULAR_LABEL
)


def compute_stats(scores):
    scores = np.array(scores)
    median = np.nanmedian(scores)
    q1, q3 = np.nanpercentile(scores, [25, 75])
    mean = np.nanmean(scores)
    std = np.nanstd(scores)
    return median, q1, q3, mean, std

def parse_metric_files(result_csv_filename_pixel, result_csv_filename_seg, split, phase):
    # Check if the files exist
    if not os.path.exists(result_csv_filename_pixel):
        raise FileNotFoundError(f"File {result_csv_filename_pixel} does not exist.")
    if not os.path.exists(result_csv_filename_seg):
        raise FileNotFoundError(f"File {result_csv_filename_seg} does not exist.")
    df_pixel = pd.read_csv(result_csv_filename_pixel)
    df_seg = pd.read_csv(result_csv_filename_seg)
    if split in "test":
        if phase is not None:
            pass
        else:
            bdmap_id_test = pd.read_csv("../STEP3-CAREModel/splits/BDMAP_O_AV_meta_test.csv")["bdmap_id"].apply(lambda x: x[:-2])
            df_pixel = df_pixel[df_pixel["case_name"].isin(bdmap_id_test)]
            df_seg = df_seg[df_seg["case_name"].isin(bdmap_id_test)]
    else:   # split `all`
        pass
    return df_pixel, df_seg


def compute_metrics(df_pixel, df_seg):
    # pixel-wise metrics
    psnr_scores, ssim_scores = map(np.asarray, (df_pixel["psnr_3d"], df_pixel["ssim_3d"]))
    psnr_scores = psnr_scores[~np.isnan(psnr_scores)]
    ssim_scores = ssim_scores[~np.isnan(psnr_scores)]
    p_med, p_q1, p_q3, p_avg, p_std = compute_stats(psnr_scores)
    s_med, s_q1, s_q3, s_avg, s_std = compute_stats(ssim_scores)
    # anatomy-aware metrics
    results = [row[1].tolist() for row in df_seg.iterrows()]
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

    large_nsd_median, large_nsd_q1, large_nsd_q3, large_nsd_mean, large_nsd_std = compute_stats(np.nanmean(np.asarray(large_nsd_scores), axis=1))
    large_nsd_nanmean = np.nanmean(np.asarray(large_nsd_scores), axis=1, keepdims=False)
    large_nsd_nonan = large_nsd_nanmean[~np.isnan(large_nsd_nanmean)]

    small_nsd_median, small_nsd_q1, small_nsd_q3, small_nsd_mean, small_nsd_std = compute_stats(np.nanmean(np.asarray(small_nsd_scores), axis=1))
    small_nsd_nanmean = np.nanmean(np.asarray(small_nsd_scores), axis=1, keepdims=False)
    small_nsd_nonan = small_nsd_nanmean[~np.isnan(small_nsd_nanmean)]

    vessel_cldice_median, vessel_cldice_q1, vessel_cldice_q3, vessel_cldice_mean, vessel_cldice_std = compute_stats(np.nanmean(np.asarray(vessel_cldice_scores), axis=1))
    vessel_cldice_nanmean = np.nanmean(np.asarray(vessel_cldice_scores), axis=1, keepdims=False)
    vessel_cldice_nonan = vessel_cldice_nanmean[~np.isnan(vessel_cldice_nanmean)]

    tumor_median, tumor_q1, tumor_q3, tumor_mean, tumor_std = compute_stats(np.nanmean(np.asarray(tumor_scores), axis=1))
    tumor_nanmean = np.nanmean(np.asarray(tumor_scores), axis=1, keepdims=False)
    tumor_nonan = tumor_nanmean[~np.isnan(tumor_nanmean)]

    tubular_median, tubular_q1, tubular_q3, tubular_mean, tubular_std = compute_stats(np.nanmean(np.asarray(tubular_scores), axis=1))
    tubular_nanmean = np.nanmean(np.asarray(tubular_scores), axis=1, keepdims=False)
    tubular_nonan = tubular_nanmean[~np.isnan(tubular_nanmean)]

    return dict(
        #original data (without any nan, for t-test)
        psnr_scores=psnr_scores,
        ssim_scores=ssim_scores,
        large_nsd_scores=large_nsd_nonan,
        small_nsd_scores=small_nsd_nonan,
        vessel_cldice_scores=vessel_cldice_nonan,
        tumor_scores=tumor_nonan,
        tubular_scores=tubular_nonan,
        #statistics
        psnr_median=p_med, psnr_q1=p_q1, psnr_q3=p_q3,
        ssim_median=s_med, ssim_q1=s_q1, ssim_q3=s_q3,
        large_nsd_median=large_nsd_median, large_nsd_q1=large_nsd_q1, large_nsd_q3=large_nsd_q3,
        small_nsd_median=small_nsd_median, small_nsd_q1=small_nsd_q1, small_nsd_q3=small_nsd_q3,
        vessel_cldice_median=vessel_cldice_median, vessel_cldice_q1=vessel_cldice_q1, vessel_cldice_q3=vessel_cldice_q3,
        tumor_median=tumor_median, tumor_q1=tumor_q1, tumor_q3=tumor_q3,
        tubular_median=tubular_median, tubular_q1=tubular_q1, tubular_q3=tubular_q3 
    )

def printLaTeX_table_line(method, num_view, CARE, split, phase=None):
    assert split in ["test", "all"]
    result_csv_filename_base = os.path.join(csv_root, f"BDMAP_O_{method}_{num_view}")
    result_csv_filename_pixel = result_csv_filename_base + f"_pixel"
    result_csv_filename_seg = result_csv_filename_base + f"_seg"
    if CARE:
        result_csv_filename_pixel  += "_care.csv"
        result_csv_filename_seg  += "_care.csv"
    else:
        result_csv_filename_pixel  += ".csv"
        result_csv_filename_seg  += ".csv"


    df_pixel, df_seg = parse_metric_files(result_csv_filename_pixel, result_csv_filename_seg, split, phase)
    if CARE:
        df_pixel_original, df_seg_original = parse_metric_files(
                                                result_csv_filename_pixel.replace("_care.csv", ".csv"), 
                                                result_csv_filename_seg.replace("_care.csv", ".csv"), 
                                                split, phase)

    metrics = compute_metrics(df_pixel, df_seg)
    if CARE:
        original_metrics = compute_metrics(df_pixel_original, df_seg_original)

    # Print LaTeX table line
    if split=="test":
        if not CARE:
            prefix = "200 views$^*$" if num_view == "200" else f"{num_view} views"
        else:
            prefix = "\multicolumn{1}{r}{\\textbf{+\loss}}"
    if CARE:
        if split == "test":
            print(f"& {prefix}", end="")
        def round_up_to_10(x_with_significance):
            x, is_significant = x_with_significance
            if not is_significant:
                return 0
            return int(math.ceil(x / 10.0)) * 10
        # NOTE: color_string is not used in the print statement, but it is used to determine the color of the cell
        metric_nicknames = ["ssim", "psnr", "large_nsd", "small_nsd", "tubular", "vessel_cldice", ]
        significance_dict = {}
        for key in metric_nicknames:
            # Perform mannwhitneyu test (since we only have a little data)
            t_stat, p_value = mannwhitneyu(metrics[f'{key}_scores'], original_metrics[f'{key}_scores'], alternative='two-sided')
            if np.isnan(p_value):
                raise ValueError(f"p_value is NaN for {key}")
            significance_dict[key] = p_value < 0.05  # standard threshold
            # Calculate the difference
            # NOTE: the difference is not used in the print statement, but it is used to determine the color of the cell
            values_care = np.asarray(metrics[f'{key}_median'])
            values_original = np.asarray(original_metrics[f'{key}_median'])
            diff = values_care - values_original
            
            # color_string =('red' if diff>=0 else 'green') + '!' + f"{round_up_to_10_if_gt_5(abs(diff))}"
            color_string = ('green' if diff >= 0 else 'red') + '!' + f"{round_up_to_10((abs(diff), significance_dict[key]))}"
            print(f"  &\cellcolor{{{color_string}}}{metrics[f'{key}_median']:.1f}\\tiny{{~({metrics[f'{key}_q1']:.1f},{metrics[f'{key}_q3']:.1f})}}", end="" if key != metric_nicknames[-1] else "\\\\ \n")

        # print(f"   & \cellcolor{}{metrics['s_med']:.1f}\\tiny{{~({metrics['s_q1']:.1f},{metrics['s_q3']:.1f})}}"
        #         color_string =('red' if metrics['p_med'] - original_metrics['p_med']>=0 else 'green') + '!' + 
        #         f" & \cellcolor{}{metrics['p_med']:.1f}\\tiny{{~({metrics['p_q1']:.1f},{metrics['p_q3']:.1f})}}"
        #         color_string =('red' if metrics['large_nsd_median'] - original_metrics['large_nsd_median']>=0 else 'green') + '!' + 
        #         f" & \cellcolor{}{metrics['large_nsd_median']:.1f}\\tiny{{~({metrics['large_nsd_q1']:.1f},{metrics['large_nsd_q3']:.1f})}}"
        #         color_string =('red' if metrics['small_nsd_median'] - original_metrics['small_nsd_median']>=0 else 'green') + '!' + 
        #         f" & \cellcolor{}{metrics['small_nsd_median']:.1f}\\tiny{{~({metrics['small_nsd_q1']:.1f},{metrics['small_nsd_q3']:.1f})}}"
        #         color_string =('red' if metrics['vessel_cldice_median'] - original_metrics['vessel_cldice_median']>=0 else 'green') + '!' + 
        #         f" & \cellcolor{}{metrics['vessel_cldice_median']:.1f}\\tiny{{~({metrics['vessel_cldice_q1']:.1f},{metrics['vessel_cldice_q3']:.1f})}}"
        #         color_string =('red' if metrics['tubular_median'] - original_metrics['tubular_median']>=0 else 'green') + '!' + 
        #         f" & \cellcolor{}{metrics['tubular_median']:.1f}\\tiny{{~({metrics['tubular_q1']:.1f},{metrics['tubular_q3']:.1f})}}"
        #         # f" &\cellcolor{{TODO}} {metrics['tumor_median']:.1f}\\tiny{{~({metrics['tumor_q1']:.1f},{metrics['tumor_q3']:.1f})}}"
        # )
    else:
        if split == "test":
            print(f"& {prefix}", end="")
        print(  f"  &{metrics['ssim_median']:.1f}\\tiny{{~({metrics['ssim_q1']:.1f},{metrics['ssim_q3']:.1f})}}"
                f"  &{metrics['psnr_median']:.1f}\\tiny{{~({metrics['psnr_q1']:.1f},{metrics['psnr_q3']:.1f})}}"
                f"  &{metrics['large_nsd_median']:.1f}\\tiny{{~({metrics['large_nsd_q1']:.1f},{metrics['large_nsd_q3']:.1f})}}"
                f"  &{metrics['small_nsd_median']:.1f}\\tiny{{~({metrics['small_nsd_q1']:.1f},{metrics['small_nsd_q3']:.1f})}}"
                f"  &{metrics['tubular_median']:.1f}\\tiny{{~({metrics['tubular_q1']:.1f},{metrics['tubular_q3']:.1f})}}"
                f"  &{metrics['vessel_cldice_median']:.1f}\\tiny{{~({metrics['vessel_cldice_q1']:.1f},{metrics['vessel_cldice_q3']:.1f})}} \\\\"
                # f" & {metrics['tumor_median']:.1f}\\tiny{{~({metrics['tumor_q1']:.1f},{metrics['tumor_q3']:.1f})}}"
        )
    

if __name__ == "__main__":
    methods = ["intratomo", "nerf", "tensorf", "r2_gaussian", "naf", "FDK", "SART", "ASD_POCS", "Lineformer"]#[:4]
    num_views_lists = ["50", "200"]
    CARE_list = [True, False]

    csv_root = "resultsCSVddim50"



    # NOTE: table 1 (method, 50, False)
    print("*"*50, "Table 1", "*"*50)
    for method in methods:
        print(f"\033[31m{method}\033[0m")
        printLaTeX_table_line(method=method, num_view="50", CARE=False, split="all")
    print("*"*50, "*"*7, "*"*50)


    # NOTE: table 2 
    print("*"*50, "Table 2", "*"*50)
    for method in methods:
        print(f"\033[31m{method}\033[0m")
        printLaTeX_table_line(method=method, num_view="50", CARE=False, split="test")
        printLaTeX_table_line(method=method, num_view="50", CARE=True, split="test")
        # printLaTeX_table_line(method=method, num_view="200", CARE=False, split="test")
        # printLaTeX_table_line(method=method, num_view="200", CARE=True, split="test")
    print("*"*50, "*"*7, "*"*50)


    