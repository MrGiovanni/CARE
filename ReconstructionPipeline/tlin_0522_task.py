import nibabel as nib
import numpy as np
import cv2
import os
from glob import glob

# === Fixed list of case IDs ===
case_ids = [
    "BDMAP_O0000001",
    "BDMAP_O0000002",
    "BDMAP_O0000003",
    "BDMAP_O0000004",
    # "BDMAP_O0000005",
    "BDMAP_O0000008"
]

# === Paths ===
base_root = "/mnt/ccvl15/tlin67/3DReconstruction/CARE/ReconstructionPipeline"
output_root = "tlin_0522"
os.makedirs(output_root, exist_ok=True)

# === Parameters ===
overlap = 20
gap = 10
slices_per_case = 8
min_unique_labels = 5
pad = 20
target_strip_height = 512
target_strip_width = target_strip_height*4

# === Color Map ===
color_map = {
    1: (255, 0, 0),    2: (0, 255, 0),    3: (0, 0, 255),
    4: (255, 255, 0),  5: (255, 0, 255),  6: (0, 255, 255),
    7: (100, 100, 255), 8: (255, 100, 100), 9: (150, 255, 150),
    10: (255, 150, 255), 11: (150, 150, 255), 12: (180, 180, 0),
    13: (180, 0, 180), 14: (0, 180, 180), 15: (200, 200, 100),
    16: (100, 200, 200), 17: (255, 200, 100), 18: (200, 100, 255),
    19: (100, 255, 200), 20: (0, 0, 0), 21: (255, 255, 255)
}

# === Helper Functions ===
def overlay_segmentation(ct_slice, seg_slice, alpha=0.25):
    ct_norm = cv2.normalize(ct_slice, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    ct_rgb = cv2.cvtColor(ct_norm, cv2.COLOR_GRAY2RGB)
    overlay = ct_rgb.copy()
    for label, color in color_map.items():
        mask = seg_slice == label
        overlay[mask] = (1 - alpha) * overlay[mask] + alpha * np.array(color)
    return overlay.astype(np.uint8)

def get_sampled_labeled_slices(pred_data, num_samples, min_unique):
    z_dim = pred_data.shape[2]
    def count_unique(z): return len(np.unique(pred_data[:, :, z][pred_data[:, :, z] > 0]))
    start = next((z for z in range(z_dim) if count_unique(z) >= min_unique), None)
    end = next((z for z in reversed(range(z_dim)) if count_unique(z) >= min_unique), None)
    if start is None or end is None or end - start + 1 < num_samples:
        return []
    return list(np.linspace(start, end, num_samples, dtype=int))

def get_zoom_bbox(pred_slices, pad):
    coords = [np.argwhere(p > 0) for p in pred_slices if np.any(p > 0)]
    if not coords:
        return None
    coords = np.concatenate(coords, axis=0)
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    cy, cx = (y_min + y_max) // 2, (x_min + x_max) // 2
    half = max(y_max - y_min, x_max - x_min) // 2 + pad
    h, w = pred_slices[0].shape
    return (
        max(cy - half, 0),
        min(cy + half, h - 1),
        max(cx - half, 0),
        min(cx + half, w - 1)
    )

def render_case_strip(case_path):
    try:
        ct_base = nib.load(os.path.join(case_path, "ct.nii.gz")).get_fdata()
        pred_base = nib.load(os.path.join(case_path, "pred.nii.gz")).get_fdata()
        ct_care = nib.load(os.path.join(case_path, "ct_care.nii.gz")).get_fdata()
        pred_care = nib.load(os.path.join(case_path, "pred_care.nii.gz")).get_fdata()
    except Exception as e:
        print(f"❌ Failed to load: {case_path}: {e}")
        return None

    slice_ids = get_sampled_labeled_slices(pred_care, slices_per_case, min_unique_labels)
    if not slice_ids:
        print(f"⚠️ Skipping {os.path.basename(case_path)} (no good slices)")
        return None

    # Rotate and overlay
    ct_base_rot = [np.rot90(ct_base[:, :, z], k=1) for z in slice_ids]
    ct_care_rot = [np.rot90(ct_care[:, :, z], k=1) for z in slice_ids]
    pred_base_rot = [np.rot90(pred_base[:, :, z], k=1) for z in slice_ids]
    pred_care_rot = [np.rot90(pred_care[:, :, z], k=1) for z in slice_ids]

    overlays_base = [overlay_segmentation(c, p) for c, p in zip(ct_base_rot, pred_base_rot)]
    overlays_care = [overlay_segmentation(c, p) for c, p in zip(ct_care_rot, pred_care_rot)]

    bbox = get_zoom_bbox(pred_care_rot, pad)
    if not bbox:
        return None
    y0, y1, x0, x1 = bbox
    H, W = y1 - y0 + 1, x1 - x0 + 1
    step = W - overlap
    total_w = slices_per_case * W - (slices_per_case - 1) * overlap
    strip = np.zeros((H * 2, total_w, 3), dtype=np.uint8)

    for i in range(slices_per_case):
        xb = i * step
        strip[0:H, xb:xb + W] = overlays_base[i][y0:y1 + 1, x0:x1 + 1]
        strip[H:H * 2, xb:xb + W] = overlays_care[i][y0:y1 + 1, x0:x1 + 1]

    # Resize to consistent size
    return cv2.resize(strip, (target_strip_width, target_strip_height), interpolation=cv2.INTER_AREA)

# === Discover all relevant method folders
method_folders = sorted(glob(os.path.join(base_root, "BDMAP_O_*_50")))

for method_path in method_folders:
    method_name = os.path.basename(method_path)
    print(f"🔍 Processing: {method_name}")

    case_dirs = [os.path.join(method_path, cid) for cid in case_ids if os.path.isdir(os.path.join(method_path, cid))]
    all_strips = []

    for case_path in case_dirs:
        strip = render_case_strip(case_path)
        if strip is not None:
            if all_strips:
                all_strips.append(np.full((gap, target_strip_width, 3), 255, dtype=np.uint8))
            all_strips.append(strip)

    if not all_strips:
        print(f"⚠️ No valid cases for {method_name}")
        continue

    final_img = np.vstack(all_strips)
    save_path = os.path.join(output_root, f"{method_name}.png")
    cv2.imwrite(save_path, cv2.cvtColor(final_img, cv2.COLOR_RGB2BGR))
    print(f"✅ Saved: {save_path}")
