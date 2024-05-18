import h5py
import os

def inspect_hdf5_file(file_path):
    print(f"Inspecting HDF5 file: {file_path}")
    with h5py.File(file_path, 'r') as f:
        def print_attrs(name, obj):
            if isinstance(obj, h5py.Dataset):
                print(f"Dataset: {name}")
                print(f"Shape: {obj.shape}")
                print(f"Dtype: {obj.dtype}")
            elif isinstance(obj, h5py.Group):
                print(f"Group: {name}")
        
        f.visititems(print_attrs)

current_directory = os.getcwd()
# Example usage
file_path = f'{current_directory}/data/Vesuvius/train/dataset/3350_4000_8450_xyz_256_res1_s4.h5'
# file_path = '/home/james/Documents/VS/pytorch-3dunet-instanceSeg/resources/sample_ovule.h5'
inspect_hdf5_file(file_path)
