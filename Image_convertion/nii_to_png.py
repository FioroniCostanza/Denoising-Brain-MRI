import nibabel as nib
import numpy as np
import os
from tqdm import tqdm
import imageio

def normalize_image_16bit(data):
    """Normalize the image intensity to the range [0, 65535]."""
    data_min = np.min(data)
    data_max = np.max(data)
    normalized_data = (data - data_min) / (data_max - data_min) * 65535
    return normalized_data.astype(np.uint16)

def nifti_to_png(nifti_path, output_dir):
    img = nib.load(nifti_path)
    data = img.get_fdata()
    normalized_data = normalize_image_16bit(data)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for i in range(normalized_data.shape[2]):
        slice_data = normalized_data[:, :, i]

        file_name = os.path.basename(nifti_path).replace('.nii.gz', '').replace('.nii', '')
        output_path = os.path.join(output_dir, f'{file_name}_slice_{i:03d}.png')
        imageio.imwrite(output_path, slice_data)

def convert_folder(input_folder, output_folder):
    nifti_files = []
    for root, dirs, files in os.walk(input_folder):
        for file in files:
            if file.endswith('.nii') or file.endswith('.nii.gz'):
                nifti_files.append(os.path.join(root, file))

    with tqdm(total=len(nifti_files), desc='Processing files', unit='file') as pbar:
        for nifti_file in nifti_files:
            relative_path = os.path.relpath(os.path.dirname(nifti_file), input_folder)
            output_dir = os.path.join(output_folder, relative_path)
            nifti_to_png(nifti_file, output_dir)
            pbar.update(1)

input_folder = '/scratch/Costanza/Registration/ADNI_DDPM_S_R1_10'
output_folder = '/scratch/Costanza/ADNI_DDPM_S_R_PNG'
convert_folder(input_folder, output_folder)
