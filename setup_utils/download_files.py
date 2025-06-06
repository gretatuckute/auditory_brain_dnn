import requests
import tarfile
import sys
import os
from os.path import join

ROOT = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..'))

# Use the flags below to indicate what you want to download. The flags get_data and get_model_actv are sufficient to run your own models.
get_data = True # ~180MB
get_model_actv = False # Obs: large file (~2.7GB)
get_results = False # Obs: large file (~2.1GB)
get_fsavg_surf = False # ~480MB
get_vox_matrix = False # ~200MB This is only used as metadata for surface plotting

print(f'Downloading files to {ROOT}')

def download_extract_remove(url, extract_location):
    """
    Base function from Jenelle Feather: https://github.com/jenellefeather/model_metamers_pytorch/blob/master/download_large_files.py
    """
    temp_file_location = os.path.join(extract_location, 'temp.tar')
    print('Downloading %s to %s'%(url, temp_file_location))
    with open(temp_file_location, 'wb') as f:
        r = requests.get(url, stream=True)
        for chunk in r.raw.stream(1024, decode_content=False):
            if chunk:
                f.write(chunk)
                f.flush()
    print('Extracting %s'%temp_file_location)
    tar = tarfile.open(temp_file_location)
    # Check if there is the extraction would overwrite an existing file
    for member in tar.getmembers():
        if os.path.exists(member.name):
            print('File %s already exists, aborting'%member.name)
            sys.exit()

    tar.extractall(path=extract_location) # untar file into same directory
    tar.close()

    print('Removing temp file %s'%temp_file_location)
    os.remove(temp_file_location)

# Download the data folder (~180MB). This folder contains the two neural datasets (NH2015, B2021) and the component dataset (NH2015comp)
if get_data:
    url_data_folder = 'https://mcdermottlab.mit.edu//tuckute_feather_2023/data.tar'
    download_extract_remove(url_data_folder, ROOT)

# Download the model activations folder (~2.7GB). This folder contains the activations of the models used in the paper.
if get_model_actv:
    url_model_actv_folder = 'https://mcdermottlab.mit.edu//tuckute_feather_2023/model_actv.tar'
    download_extract_remove(url_model_actv_folder, ROOT)

# Download the results folder (~2.1GB). This folder contains the results of the analyses used in the paper.
if get_results:
    url_results_folder = 'https://mcdermottlab.mit.edu//tuckute_feather_2023/results.tar'
    download_extract_remove(url_results_folder, ROOT)

# Download the fsavg_surf folder (~480MB). This folder contains the surface-projected layer position maps for each model.
if get_fsavg_surf:
    url_fsavg_surf_folder = 'https://mcdermottlab.mit.edu//tuckute_feather_2023/fsavg_surf.tar'
    download_extract_remove(url_fsavg_surf_folder, ROOT)


# Download the voxel matrix mat file (~200MB). This file is used as metadata for the surface plotting.
if get_vox_matrix:
    url_vox_matrix_folder = 'https://mcdermottlab.mit.edu//tuckute_feather_2023/voxel_matrix_for_alex.tar'
    download_extract_remove(url_vox_matrix_folder, ROOT)
