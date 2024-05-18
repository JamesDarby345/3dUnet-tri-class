import os
import h5py
import nrrd
import numpy as np

import os
import h5py
import nrrd
import numpy as np

def create_hdf5_files(raw_dir, label_dir, output_dir, weight_dir=None):
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    for raw_filename in os.listdir(raw_dir):
        if raw_filename.endswith('.nrrd'):
            # Construct full file paths for raw and label
            raw_path = os.path.join(raw_dir, raw_filename)
            base_name = '_'.join(raw_filename.split('_')[:-1])
            label_filename = "tri_class_"+base_name+'_label.nrrd'
            label_path = os.path.join(label_dir, label_filename)

            print(f"Processing: {raw_filename} and {label_filename}")
            
            if not os.path.exists(label_path):
                print(f"Label file not found for {label_filename}")
                label_filename = f"tri_class_{raw_filename}"
                print(f"trying {label_filename}")
                label_path = os.path.join(label_dir, label_filename)
            if not os.path.exists(label_path):
                print(f"Label file not found for {raw_filename}, skipping.")
                continue
            
            # Read raw and label data
            raw_data, raw_header = nrrd.read(raw_path)
            label_data, label_header = nrrd.read(label_path)
            
            # Optionally read weight data
            weight_data = None
            if weight_dir:
                weight_path = os.path.join(weight_dir, label_filename)
                if os.path.exists(weight_path):
                    weight_data, weight_header = nrrd.read(weight_path)
            
            # Convert to numpy arrays
            raw_data = np.asarray(raw_data, dtype=np.float32)
            label_data = np.asarray(label_data, dtype=np.uint8)
            if weight_data is not None:
                weight_data = np.asarray(weight_data, dtype=np.float32)
            
            # Construct output file path
            output_filename = base_name + '.h5'
            output_path = os.path.join(output_dir, output_filename)
            
            # Create HDF5 file
            with h5py.File(output_path, 'w') as hdf5_file:
                hdf5_file.create_dataset('raw', data=raw_data, dtype='float32')
                hdf5_file.create_dataset('label', data=label_data, dtype='uint8')
                if weight_data is not None:
                    hdf5_file.create_dataset('weight', data=weight_data, dtype='float32')
            
            print(f"Processed and saved {raw_filename} and {label_filename} to {output_path}")



current_directory = os.getcwd()

sub_dir = 'test'
raw_directory = f'{current_directory}/data/Vesuvius/{sub_dir}/raw'
label_directory = f'{current_directory}/data/Vesuvius/{sub_dir}/label'
output_hdf5_path = f'{current_directory}/data/Vesuvius/{sub_dir}/dataset'
create_hdf5_files(raw_directory, label_directory, output_hdf5_path)

sub_dir = 'train'
raw_directory = f'{current_directory}/data/Vesuvius/{sub_dir}/raw'
label_directory = f'{current_directory}/data/Vesuvius/{sub_dir}/label'
output_hdf5_path = f'{current_directory}/data/Vesuvius/{sub_dir}/dataset'
create_hdf5_files(raw_directory, label_directory, output_hdf5_path)

sub_dir = 'val'
raw_directory = f'{current_directory}/data/Vesuvius/{sub_dir}/raw'
label_directory = f'{current_directory}/data/Vesuvius/{sub_dir}/label'
output_hdf5_path = f'{current_directory}/data/Vesuvius/{sub_dir}/dataset'
create_hdf5_files(raw_directory, label_directory, output_hdf5_path)
