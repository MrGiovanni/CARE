"""
For FELIX data:
```bash
python niigz2h5.py --input_dir /mnt/realccvl15/zzhou82/data/AbdomenAtlasPro/ --output_dir /path/to/output_dir -A -V
```

For all the BDMAP data in the folder:
```bash
python niigz2h5.py --input_dir /mnt/realccvl15/zzhou82/data/AbdomenAtlasPro/ --output_dir /path/to/output_dir
```
"""



import os
import numpy as np
import nibabel as nib
import multiprocessing as mp
import argparse
import sys
import glob
from tqdm import tqdm
import h5py


def niigz2h5(in_path):

    def varifyh5(filename): # read the h5 file to see if the conversion is finished or not
        try:
            with h5py.File(filename, "r") as hf:   # can read successfully
                data = hf["image"]
                shape_test = data.shape
                slice_test = data[3:6]
                pass
            return True
        except:     # transform not complete
            print(filename, "has error")
            return False

    def saveh5(filename):
        save_dtype = 'uint8' if "pred" in filename else "int16"
        # load ct to convert
        try:
            nii_array = nib.load(os.path.join(in_path, f"{filename}.nii.gz")).get_fdata()    # float64, but save as int16 in h5
            nii_shape = nii_array.shape     # not all CT is 512 x 512
        except:
            print(f"broken {filename}", in_path)
            return

        # save as h5 file in int16
        try:
            with h5py.File(os.path.join(output_path_h5, f"{filename}.h5"), 'w') as hf:
                hf.create_dataset('image', 
                    data=nii_array, 
                    compression='gzip',     
                    chunks=(nii_shape[0], nii_shape[1], 1),   # chunks for better loading speed!
                    dtype=save_dtype)   # int16 for ct, uint8 for segmentations
        except ValueError:
            print(os.path.join(in_path, f"{filename}.nii.gz"))
            raise ValueError
    # parse paths
    bdmap_id = in_path.split("/")[-1]   
    output_path_h5 = os.path.join(root, bdmap_id)

    # check if had been transformed (for resuming)
    output_path_h5_filename = os.path.join(output_path_h5, "ct.h5")
          
    # converting nii.gz to h5 files!
    if not os.path.exists(os.path.join(output_path_h5, "ct.h5")):
        saveh5("ct")    # convert nii.gz to h5 (int16)
    if not os.path.exists(os.path.join(output_path_h5, "pred.h5")):# and os.path.exists(os.path.join(output_path_h5, "pred.nii.gz")):
        saveh5("pred")
    


    


if __name__ == "__main__":
    # input_dir = "/mnt/realccvl15/zzhou82/data/AbdomenAtlasPro/"
    # input_dir = "/ccvl/net/ccvl15/tlin67/3DReconstruction/Tianyu/STEP3-ControlNetModel/dataset/data"
    # output_dir = "/ccvl/net/ccvl15/tlin67/Dataset_raw/FELIXtemp/FELIXh5"

    # # Create ArgumentParser object
    parser = argparse.ArgumentParser()
    parser.add_argument("--pth", type=str, help="h5 data input folder", required=True)
    # parser.add_argument("--output_dir", type=str, help="h5 data output folder", required=True)
    # parser.add_argument("-A", action='store_true', help="BDMAP data starts with A ")
    # parser.add_argument("-V", action='store_true', help="BDMAP data starts with V ")
    # parser.add_argument("-O", action='store_true', help="BDMAP data starts with O ")
    args = parser.parse_args()

    print(f"\033[31m{args.pth}\033[0m")

    # os.makedirs(args.output_dir, exist_ok=True)

    # prefixes = [prefix for prefix, enabled in [("BDMAP_A", args.A), ("BDMAP_O", args.O), ("BDMAP_V", args.V)] if enabled]

    root = args.pth#"BDMAP_O_FDK_50"

    paths = sorted([entry.path for entry in os.scandir(root)])     # default: all the data in AbdomenAtlasPro
    # print(paths)
    
    print(len(paths), "CT scans found in given filtering condition")


    # for path in tqdm(paths):
    #     niigz2h5(path)
    #     break

    num_workers = int(mp.cpu_count() * 0.8)
    with mp.Pool(num_workers) as pool:
        # Wrap the pool.imap() function with tqdm for progress tracking
        results = list(tqdm(pool.imap(niigz2h5, paths), total=len(paths), desc=f"Converting {root.split('/')[-1]}'s nii.gz to h5"))
    print("Processing complete!")  # Print first 10 results

    