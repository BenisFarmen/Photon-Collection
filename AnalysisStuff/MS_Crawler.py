import glob
import nibabel
from nilearn.image import load_img
import pandas as pd
import os

wmlist = glob.glob("/spm-data/Scratch/spielwiese_vincent/MS_Prediction_Cat12/nifti/*/mri/wmnifti.nii")

ID_excel = pd.read_excel("/spm-data/Scratch/spielwiese_vincent/MS_Prediction_Cat12/Subjects_rms_processed.xlsx")
ID = ID_excel['nifti_t1_dataset_id']
print(ID)

file_path_list = []

for i in ID:
    search_term = i
    file_path = os.path.join('/spm-data/Scratch/spielwiese_vincent/MS_Prediction_Cat12/nifti/', search_term, '/mri/wmnifti.nii')
    file_path_list.append(file_path)

X = load_img(file_path_list).get_data()

print(X.shape)

#man_map = image.load_img(DATEINAME).get_data()
