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
        save_dtype = 'uint8' if "segmentations" in filename else "int16"
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
    output_path_h5 = os.path.join(args.output_dir, bdmap_id)
    has_labels = os.path.exists(os.path.join(in_path, "segmentations")) # has segmentation labels

    # prepare dirs
    os.makedirs(output_path_h5, exist_ok=True)
    if has_labels:  
        os.makedirs(os.path.join(output_path_h5, "segmentations"), exist_ok=True)   # prepare to convert labels
        segmentation_paths = glob.glob(os.path.join(in_path, "segmentations", "*.nii.gz"))
        num_labels = len(segmentation_paths) # for checking
        assert len(num_labels) > 0

    # check if had been transformed (for resuming)
    output_path_h5_filename = os.path.join(output_path_h5, "ct.h5")
    if os.path.exists(output_path_h5_filename) and varifyh5(output_path_h5_filename):   # exist and finished
        if has_labels:
            label_h5_paths = glob.glob(os.path.join(output_path_h5, "segmentations", "*.h5"))
            if len(label_h5_paths) == num_labels and all([varifyh5(path) for path in label_h5_paths]):    
                return
            else:       # has labels but not all the labels have been transformed --> still need converting
                pass
        else:
            return
          
    # converting nii.gz to h5 files!
    saveh5("ct")    # convert nii.gz to h5 (int16)
    if has_labels:
        for segmentation_path in segmentation_paths:
            label_name = segmentation_path.split("/")[-1].split(".")[0]
            saveh5(f"segmentations/{label_name}")   # convert nii.gz to h5 (uint8)


    


if __name__ == "__main__":

    # Create ArgumentParser object
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, help="h5 data input folder", required=True)
    parser.add_argument("--output_dir", type=str, help="h5 data output folder", required=True)
    parser.add_argument("-A", action='store_true', help="BDMAP data starts with A ")
    parser.add_argument("-V", action='store_true', help="BDMAP data starts with V ")
    parser.add_argument("-O", action='store_true', help="BDMAP data starts with O ")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    prefixes = [prefix for prefix, enabled in [("BDMAP_A", args.A), ("BDMAP_O", args.O), ("BDMAP_V", args.V)] if enabled]

    paths = sorted([entry.path for entry in os.scandir(args.input_dir) 
                if not prefixes or any(entry.name.startswith(p) for p in prefixes)])     # default: all the data in AbdomenAtlasPro
    
    print(len(paths), "CT scans found in given filtering condition")


    num_workers = int(mp.cpu_count() * 0.8)
    with mp.Pool(num_workers) as pool:
        # Wrap the pool.imap() function with tqdm for progress tracking
        results = list(tqdm(pool.imap(niigz2h5, paths), total=len(paths), desc="Converting nii.gz to h5"))
    print("Processing complete!")  # Print first 10 results

    