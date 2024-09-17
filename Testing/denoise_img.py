import os
import json
import torch
import numpy as np
import pandas as pd
import monai as mn
import nibabel as nib
from tqdm import tqdm
import matplotlib.pyplot as plt
from mediffusion import DiffusionModule

def filter_slices_by_threshold(img_slice, concat_slice, threshold=0.01):
    brain_mask_img = img_slice > threshold
    brain_mask_concat = concat_slice > threshold
    brain_volume_img = np.sum(brain_mask_img)
    brain_volume_concat = np.sum(brain_mask_concat)
    return brain_volume_img, brain_volume_concat

def load_patient_data_with_1_5T(dataset_dir):
    patient_data_dicts = {}
    for patient_id in sorted(os.listdir(dataset_dir)):
        patient_dir = os.path.join(dataset_dir, patient_id)
        if os.path.isdir(patient_dir):
            slice_dict = {}
            for file_name in sorted(os.listdir(patient_dir)):
                if file_name.endswith('.nii.gz'):
                    file_path = os.path.join(patient_dir, file_name)
                    nii_image = nib.load(file_path)
                    image_data = nii_image.get_fdata()

                    if '1.5T' in file_name or '1_5T' in file_name:
                        slice_dict['concat'] = image_data
                    elif '3.0T' in file_name or '3_0T' in file_name:
                        slice_dict['img'] = image_data

            if 'img' in slice_dict and 'concat' in slice_dict:
                img_slices = slice_dict['img']
                concat_slices = slice_dict['concat']
                num_slices = img_slices.shape[2]  # axial slices

                patient_slices = []
                for i in range(num_slices):
                    brain_volume_img, brain_volume_concat = filter_slices_by_threshold(
                        img_slices[:, :, i], 
                        concat_slices[:, :, i], 
                        threshold=0.01
                    )
                    if brain_volume_img > 100 and brain_volume_concat > 100:
                        patient_slices.append({
                            'img': img_slices[:, :, i][np.newaxis, :, :], 
                            'concat': concat_slices[:, :, i][np.newaxis, :, :],
                            'cls': 0  
                        })

                if patient_slices:
                    patient_data_dicts[patient_id] = patient_slices
    
    return patient_data_dicts

def save_nifti_image(image_slices, affine, patient_id, save_dir, image_type):
    image_3d = np.stack(image_slices, axis=-1).astype(np.float32)
    nifti_image = nib.Nifti1Image(image_3d, affine)
    save_path = os.path.join(save_dir, f'{patient_id}_{image_type}.nii.gz')
    nib.save(nifti_image, save_path)

def process_batch(batch, model, patient_id, slice_number, nifti_save_dir):
    img = batch['img'].to('cuda')
    concat = batch['concat'].to('cuda')

    noise = torch.randn(1, 1, 256, 256, device='cuda').half()

    model_kwargs = {'concat': concat}
    denoised_img = model.predict(
        noise,
        model_kwargs=model_kwargs,
        classifier_cond_scale=4,
        inference_protocol="DDIM100"
    )[0].cpu().numpy().squeeze()

    ground_truth_img = img[0].cpu().numpy().squeeze()
    original_img = concat[0].cpu().numpy().squeeze()

    if patient_id not in nifti_save_dir:
        nifti_save_dir[patient_id] = {
            '1.5T': [],
            '3.0T': [],
            'denoised': []
        }

    nifti_save_dir[patient_id]['1.5T'].append(original_img)
    nifti_save_dir[patient_id]['3.0T'].append(ground_truth_img)
    nifti_save_dir[patient_id]['denoised'].append(denoised_img)
    
def main(dataset_dir1, checkpoint_path, config_path, save_dir):
    # Load dataset
    patient_data_dicts = load_patient_data_with_1_5T(dataset_dir1)

    with open(split_path, 'r') as f:
        split_dict = json.load(f)
    test_patient_ids = split_dict['test']

    # Collect test data
    test_data = []
    slice_numbers = []
    test_patient_slices = []

    for pid in test_patient_ids:
        patient_slices = patient_data_dicts[pid]
        for slice_idx, slice_data in enumerate(patient_slices):
            test_data.append(slice_data)
            slice_numbers.append(slice_idx) 
            test_patient_slices.append(pid)

    transforms = mn.transforms.Compose([
        mn.transforms.ResizeD(keys=['img', 'concat'], size_mode="longest", mode="bilinear", spatial_size=256, align_corners=False),
        mn.transforms.ScaleIntensityD(keys=['img', 'concat'], minv=-1, maxv=1),
        mn.transforms.SpatialPadD(keys=['img', 'concat'], spatial_size=(256, 256), mode="constant", constant_values=-1),
        mn.transforms.ToTensorD(keys=['img', 'concat'], dtype=torch.float, track_meta=False),
    ])

    batch_size = 1 
    test_ds = mn.data.Dataset(data=test_data, transform=transforms)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    # Load model
    model = DiffusionModule(config_path)
    model.load_ckpt(checkpoint_path, ema=True)
    model.cuda().half()
    model.eval()

    nifti_save_dir = {}

    # Run inference
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader, desc="Processing images")):
            process_batch(batch, model, test_patient_slices[batch_idx], slice_numbers[batch_idx], nifti_save_dir)

    for patient_id, images in nifti_save_dir.items():
        patient_save_dir = os.path.join(save_dir, patient_id)
        os.makedirs(patient_save_dir, exist_ok=True)

        save_nifti_image(images['1.5T'], np.eye(4), patient_id, patient_save_dir, '1.5T')
        save_nifti_image(images['3.0T'], np.eye(4), patient_id, patient_save_dir, '3.0T')
        save_nifti_image(images['denoised'], np.eye(4), patient_id, patient_save_dir, 'denoised')

if __name__ == "__main__":
    dataset_dir_1_5T = "/scratch/Costanza/Registration/ADNI_F_SK_Registered" 
    checkpoint_path = os.path.abspath("../Training/outputs/pl/last-v26.ckpt")
    config_path = os.path.abspath("../Training/config_data/config_super_res.yaml")
    split_path = os.path.abspath("../Training/last_patient_splits_5Noisy.json")
    save_dir = "./last_output27_5Noisy_NIFTI"

    main(dataset_dir_1_5T, checkpoint_path, config_path, save_dir)