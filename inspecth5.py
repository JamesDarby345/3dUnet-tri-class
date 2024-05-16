import h5py

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

# Example usage
file_path = '/home/james/Documents/VS/pytorch-3dunet-instanceSeg/data/Vesuvius/test/dataset/layers_1.h5'
# file_path = '/home/james/Documents/VS/pytorch-3dunet-instanceSeg/resources/sample_ovule.h5'
inspect_hdf5_file(file_path)
