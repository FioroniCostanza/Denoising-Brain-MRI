import os
import monai as mn
import torch
import json
import math
import numpy as np
import nibabel as nib
import random
from mediffusion import DiffusionModule, Trainer
from utils import load_patient_data_with_1_5T, load_patient_data_only_3T
from sklearn.model_selection import train_test_split, KFold

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['WANDB_API_KEY'] = "dca54d84a120fda6d88a6ccf3ea043c914104635"

TOTAL_IMAGE_SEEN = int(10e5)  
BATCH_SIZE = 32
NUM_DEVICES = 1
TRAIN_ITERATIONS = int(TOTAL_IMAGE_SEEN / (BATCH_SIZE * NUM_DEVICES))

dataset_dir_1_5T = "/scratch/Costanza/Registration/ADNI_F_SK_Registered" 
dataset_dir_only_3T = "/scratch/Costanza/PPMI_SkullStripping"

max_patients = 5 # change the max_patients number also to 30
patient_data_dicts_1_5T = load_patient_data_with_1_5T(dataset_dir_1_5T)
patient_data_dicts_3T = load_patient_data_only_3T(dataset_dir_only_3T, max_patients=max_patients) 

test_size = 0.1  

train_1_5T, test_1_5T = train_test_split(list(patient_data_dicts_1_5T.keys()), test_size=test_size, random_state=44)
train_3T = list(patient_data_dicts_3T.keys())

# Test set contains only data from the 1.5T dataset
test_patient_ids = test_1_5T
combined_train_valid_ids = train_1_5T + train_3T

split_dict = {
    "train_valid": combined_train_valid_ids,
    "test": test_patient_ids
}
with open(f"last_patient_splits_{max_patients}Noisy.json", "w") as f:
    json.dump(split_dict, f)

# Cross-Validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

fold_idx = 1
for train_index, valid_index in kf.split(combined_train_valid_ids):
    print(f"Fold {fold_idx}:")

    fold_train_ids = [combined_train_valid_ids[i] for i in train_index]
    fold_valid_ids = [combined_train_valid_ids[i] for i in valid_index]

    fold_train_data = []
    fold_valid_data = []

    for pid in fold_train_ids:
        if pid in patient_data_dicts_1_5T:
            fold_train_data.extend(patient_data_dicts_1_5T[pid])
        elif pid in patient_data_dicts_3T:
            fold_train_data.extend(patient_data_dicts_3T[pid])

    for pid in fold_valid_ids:
        if pid in patient_data_dicts_1_5T:
            fold_valid_data.extend(patient_data_dicts_1_5T[pid])
        elif pid in patient_data_dicts_3T:
            fold_valid_data.extend(patient_data_dicts_3T[pid])

    fold_idx += 1

# Define transformations with augmentations for NIfTI data
train_transforms = mn.transforms.Compose([
    mn.transforms.ResizeD(keys=['img', 'concat'], size_mode="longest", mode="bilinear", spatial_size=256, align_corners=False),
    mn.transforms.ScaleIntensityD(keys=['img', 'concat'], minv=-1, maxv=1),
    mn.transforms.SpatialPadD(keys=['img', 'concat'], spatial_size=(256, 256), mode="constant", constant_values=-1),
    # Augmentations:
    mn.transforms.RandRotateD(keys=['img', 'concat'], range_x=(math.radians(5), math.radians(5)), prob=0.5, keep_size=True, mode='bilinear'), # (range +5, -5) degrees
    mn.transforms.RandFlipD(keys=['img', 'concat'], prob=0.5, spatial_axis=0),  # Flip along left-right axis
    mn.transforms.RandAffined(keys=['img', 'concat'], 
                              translate_range=(15, 15), 
                              scale_range=(0.05, 0.05), 
                              rotate_range=(math.radians(5), math.radians(5)), 
                              padding_mode='border', 
                              prob=0.5,
                              mode='bilinear'),
    # Apply Gaussian Noise only to the 1.5T image (concat)
    mn.transforms.RandGaussianNoised(keys=['concat'], prob=0.5, mean=0.0, std=0.2),
    mn.transforms.ToTensorD(keys=['img', 'concat'], dtype=torch.float, track_meta=False),
])

valid_transforms = mn.transforms.Compose([
    mn.transforms.ResizeD(keys=['img', 'concat'], size_mode="longest", mode="bilinear", spatial_size=256, align_corners=False),
    mn.transforms.ScaleIntensityD(keys=['img', 'concat'], minv=-1, maxv=1),
    mn.transforms.SpatialPadD(keys=['img', 'concat'], spatial_size=(256, 256), mode="constant", constant_values=-1),
    mn.transforms.ToTensorD(keys=['img', 'concat'], dtype=torch.float, track_meta=False),
])

train_ds = mn.data.Dataset(data=fold_train_data, transform=train_transforms)
valid_ds = mn.data.Dataset(data=fold_valid_data, transform=valid_transforms)

train_sampler = torch.utils.data.RandomSampler(train_ds, replacement=True, num_samples=int(TOTAL_IMAGE_SEEN))

# Set up the model
model = DiffusionModule(
    config_file="./config_data/config_super_res.yaml",
    train_ds=train_ds,
    val_ds=valid_ds,
    dl_workers=4,
    train_sampler=train_sampler,
    batch_size=BATCH_SIZE,
    val_batch_size=BATCH_SIZE//2
)

# Set up the trainer
trainer = Trainer(
    max_steps=TRAIN_ITERATIONS,
    val_check_interval=2000,
    root_directory="./outputs",
    precision="16-mixed",
    devices=-1,  
    nodes=1,
    wandb_project="Denoising", 
    logger_instance=f"last_denoised_img_{max_patients}Noisy",  
)

# Start training
trainer.fit(model)
