import os
import subprocess

input_dir = "/scratch/Costanza/New_ADNI/dcm"
output_dir = "/scratch/Costanza/New_ADNI/nifti"

os.makedirs(output_dir, exist_ok=True)

for patient_folder in os.listdir(input_dir):
    patient_path = os.path.join(input_dir, patient_folder)

    if os.path.isdir(patient_path):
        # Extract the patient name (example 013_S_0996)
        patient_name = os.path.basename(patient_path)

        for root, _, files in os.walk(patient_path):
            if files:  
                # Run dcm2niix to convert DICOM files to NIfTI
                try:
                    # Format the output filename to include the main patient folder name
                    subprocess.run(
                        ["dcm2niix", "-z", "y", "-f", f"{patient_name}_%p_%t_%s", "-o", output_dir, root],
                        check=True
                    )
                    print(f"Conversion completed for {patient_name}.")
                except subprocess.CalledProcessError as e:
                    print(f"Error during conversion for {patient_name}: {e}")

print("All conversions are complete.")
