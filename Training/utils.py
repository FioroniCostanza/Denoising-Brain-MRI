import os
import numpy as np
import nibabel as nib

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

def load_patient_data_only_3T(dataset_dir, max_patients):
    patient_data_dicts = {}
    patient_count = 0  
    
    for patient_id in sorted(os.listdir(dataset_dir)):
        if patient_count >= max_patients:
            break  
        
        patient_dir = os.path.join(dataset_dir, patient_id)
        if os.path.isdir(patient_dir):
            patient_slices = []
            images = {}
            
            for file_name in sorted(os.listdir(patient_dir)):
                if file_name.endswith('.nii.gz'):
                    file_path = os.path.join(patient_dir, file_name)
                    nii_image = nib.load(file_path)
                    image_data = nii_image.get_fdata()

                    if '_rician_noise' in file_name:
                        base_name = file_name.replace('_rician_noise', '').split('.')[0]
                        if base_name not in images:
                            images[base_name] = {}
                        images[base_name]['concat'] = image_data
                    else:
                        base_name = file_name.split('.')[0]
                        if base_name not in images:
                            images[base_name] = {}
                        images[base_name]['img'] = image_data

            for base_name, img_pair in images.items():
                if 'img' in img_pair and 'concat' in img_pair:
                    img_data = img_pair['img']
                    concat_data = img_pair['concat']

                    num_slices = img_data.shape[2]

                    for i in range(num_slices):
                        brain_volume_img, brain_volume_concat = filter_slices_by_threshold(
                            img_data[:, :, i], concat_data[:, :, i], threshold=0.01
                        )
                        if brain_volume_img > 100 and brain_volume_concat > 100:  
                            patient_slices.append({
                                'img': img_data[:, :, i][np.newaxis, :, :],  
                                'concat': concat_data[:, :, i][np.newaxis, :, :],  
                                'cls': 0  
                            })
            
            if patient_slices:
                patient_data_dicts[patient_id] = patient_slices
                patient_count += 1  
    
    return patient_data_dicts