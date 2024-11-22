import cv2
import json
import numpy as np
import h5py
from tensorflow.keras.utils import Sequence
from sklearn.model_selection import train_test_split


class DatasetLoader(Sequence):
    '''Load data from dataset in batches'''
    
    def __init__(self, dataset, batch_size, shuffle=False, **kwargs):
        super().__init__(**kwargs)
        
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = np.arange(len(dataset))
        
        self.on_epoch_end()

    def __getitem__(self, i):
        '''Get a batch of point and dent masks'''
        
        start = i * self.batch_size
        stop = (i + 1) * self.batch_size
        batch_points = []
        batch_masks = []

        for j in range(start, stop):
            point, mask = self.dataset[j]
            batch_points.append(point)
            batch_masks.append(mask)

        batch_points = np.array(batch_points)
        batch_masks = np.array(batch_masks)

        batch_points = batch_points.astype(np.float32)
        batch_masks = batch_masks.astype(np.float32)

        batch_masks = batch_masks[..., np.newaxis]

        return batch_points, batch_masks
    
    def on_epoch_end(self):
        '''Shuffle indices at the end of each epoch'''
        
        if self.shuffle:
            self.indices = np.random.permutation(self.indices)
            
    def __len__(self):
        '''Return the number of steps per epoch'''
                
        return len(self.indices) // self.batch_size


class Dataset:
    def __init__(self, file_paths):
        self.file_paths = file_paths
        self.NUM_POINTS = 100
        self.sample_index = self.build_index()
        
    def build_index(self):
        '''Build index of all point-mask pairs (i.e., samples)'''
        
        sample_index = []
        
        for file_index, file_path in enumerate(self.file_paths):
            with h5py.File(file_path, 'r') as h5f:
                for point_group in h5f.keys():
                    sample_index.append((file_index, point_group))

        return sample_index
    
    def __getitem__(self, i):
        '''Return a point-mask pair'''
        
        file_index, point_group = self.sample_index[i]
        file_path = self.file_paths[file_index]

        with h5py.File(file_path, 'r') as h5f:
            point = np.array(h5f[point_group]['point'], dtype=np.float32)
            mask = np.array(h5f[point_group]['mask'], dtype=np.float32)

        return point, mask
            
    def __len__(self):
        '''Return the number of point-mask pairs'''
        
        return len(self.sample_index)


def prepare_datasets(batch_size):
    '''Batch-load generated training and validation data for model'''
    
    TEST_SIZE = 0.1
    RANDOM_STATE = 50
    
    file_paths = []
    
    print('Loading file paths...')
    with open('dent-segmentation-model/generator/config.json', 'r') as f:
        configs = json.load(f)
        
        for hailpad_type, config in configs.items():
            hailpad_count = config['hailpad_count']
            
            for i in range(hailpad_count):
                file_paths.append(f'dent-segmentation-model/generator/output/{hailpad_type}/hailpad_{i}.h5')

    train_paths, val_paths = train_test_split(file_paths, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    
    print('Instantiating training and validation datasets...')
    train_dataset = Dataset(train_paths)
    val_dataset = Dataset(val_paths)
    
    print('Instantiating training and validation dataset loaders...')
    train_dataset_loader = DatasetLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataset_loader = DatasetLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_dataset_loader, val_dataset_loader


# --- Uncomment to visualize .h5 point-mask group ---
# import matplotlib.pyplot as plt

# '''Test input-output pair creation'''
# with h5py.File('', 'r') as h5f:
#     print('Keys: ', list(h5f.keys()))
    
#     first_group_key = list(h5f.keys())[0] 
#     first_group = h5f[first_group_key]
    
#     point_image = first_group['point'][:, :, 1]
#     mask_image = first_group['mask'][:]
    
#     plt.figure(figsize=(10, 5))
#     plt.subplot(1, 2, 1)
#     plt.title('Point Image')
#     plt.imshow(point_image, cmap='gray')
#     plt.axis('off')

#     plt.subplot(1, 2, 2)
#     plt.title('Mask Image')
#     plt.imshow(mask_image, cmap='gray')
#     plt.axis('off')

#     plt.show()
