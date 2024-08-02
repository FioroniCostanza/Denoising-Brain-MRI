import os
import monai as mn
import torch
import re
from mediffusion import DiffusionModule, Trainer

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,3"
os.environ['WANDB_API_KEY'] = "dca54d84a120fda6d88a6ccf3ea043c914104635"

TOTAL_IMAGE_SEEN = int(40e6)  
BATCH_SIZE = 32
NUM_DEVICES = 3
TRAIN_ITERATIONS = int(TOTAL_IMAGE_SEEN / (BATCH_SIZE * NUM_DEVICES))

dataset_dir = "/scratch/Costanza/ADNI_DDPM_PNG_Clean"

data_dicts = []

slice_pattern = re.compile(r'_slice_(\d+)\.png')

# Dictionary to hold data grouped by patient
patient_data_dicts = {}

for patient_id in sorted(os.listdir(dataset_dir)):
    patient_dir = os.path.join(dataset_dir, patient_id)
    if os.path.isdir(patient_dir):
        slice_dict = {}
        
        for file_name in sorted(os.listdir(patient_dir)):
            match = slice_pattern.search(file_name)
            if match:
                slice_number = match.group(1)
                file_path = os.path.join(patient_dir, file_name)
                if '1.5T' in file_name:
                    if slice_number not in slice_dict:
                        slice_dict[slice_number] = {}
                    slice_dict[slice_number]['concat'] = file_path
                elif '3.0T' in file_name:
                    if slice_number not in slice_dict:
                        slice_dict[slice_number] = {}
                    slice_dict[slice_number]['img'] = file_path
        
        patient_slices = []
        for slice_number, paths in slice_dict.items():
            if 'img' in paths and 'concat' in paths:
                patient_slices.append({
                    'img': paths['img'],
                    'concat': paths['concat'],
                    'cls': 0  
                })
        
        if patient_slices:
            patient_data_dicts[patient_id] = patient_slices

# Split patients into train and validation sets
patient_ids = list(patient_data_dicts.keys())
train_patient_ids = patient_ids[:int(len(patient_ids) * 0.8)]
valid_patient_ids = patient_ids[int(len(patient_ids) * 0.8):]

train_data = []
valid_data = []

for pid in train_patient_ids:
    train_data.extend(patient_data_dicts[pid])

for pid in valid_patient_ids:
    valid_data.extend(patient_data_dicts[pid])
    
transforms = mn.transforms.Compose([
    mn.transforms.LoadImageD(keys=['img', 'concat']),
    mn.transforms.EnsureChannelFirstd(keys=['img', 'concat']),
    mn.transforms.ResizeD(keys=['img', 'concat'], size_mode="longest", mode="bilinear", spatial_size=256, align_corners=False),
    mn.transforms.ScaleIntensityD(keys=['img', 'concat'], minv=-1, maxv=1),
    mn.transforms.SpatialPadD(keys=['img', 'concat'], spatial_size=(256, 256), mode="constant", constant_values=-1),
    mn.transforms.ToTensorD(keys=['img', 'concat'], dtype=torch.float, track_meta=False),
])

train_ds = mn.data.Dataset(data=train_data, transform=transforms)
valid_ds = mn.data.Dataset(data=valid_data, transform=transforms)

train_sampler = torch.utils.data.RandomSampler(train_ds, replacement=True, num_samples=int(TOTAL_IMAGE_SEEN))


model = DiffusionModule(
    config_file="./config_data/config.yaml",
    train_ds=train_ds,
    val_ds=valid_ds,
    dl_workers=4,
    train_sampler=train_sampler,
    batch_size=BATCH_SIZE,
    val_batch_size=BATCH_SIZE//2
)

trainer = Trainer(
    max_steps=TRAIN_ITERATIONS,
    val_check_interval=2000,
    root_directory="./outputs",
    precision="16-mixed",
    devices=-1,  
    nodes=1,
    wandb_project="Denoising", 
    logger_instance="denoised_images_1",  
)

trainer.fit(model)

