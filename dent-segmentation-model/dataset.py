import cv2
import json
import numpy as np
import h5py
from sklearn.model_selection import train_test_split

    
def prepare_dataset():
    '''Load and prepare generated training and validation data for model'''
    
    TEST_SIZE = 0.1
    RANDOM_STATE = 50
    
    file_paths = []
    points = []
    masks = []
    
    print('Loading file paths...')
    with open('dent-segmentation-model/generator/config.json', 'r') as f:
        configs = json.load(f)
        
        for hailpad_type, config in configs.items():
            hailpad_count = config['hailpad_count']
            
            for i in range(hailpad_count):
                file_paths.append(f'dent-segmentation-model/generator/output/{hailpad_type}/hailpad_{i}.h5')
                  
    print('Loading point and dent masks...')  
    for file_path in file_paths:
        with h5py.File(file_path, 'r') as h5f:
            for point_group in h5f.keys():
                points.append(np.array(h5f[point_group]['point']))
                masks.append(np.array(h5f[point_group]['mask']))
    
    points = np.array(points)
    masks = np.array(masks)
    
    print('Splitting dataset into training and validation...')
    x_train, x_val, y_train, y_val = train_test_split(
        points, masks, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    return x_train, y_train, x_val, y_val



# --- TODO: Remove (for testing):

# import matplotlib.pyplot as plt

# '''Test input-output pair creation'''
# with h5py.File("", "r") as h5f:
#     print("Keys: ", list(h5f.keys()))
    
#     first_group_key = list(h5f.keys())[0] 
#     first_group = h5f[first_group_key]
    
#     point_image = first_group['point'][:, :, 1]
#     mask_image = first_group['mask'][:]
    
#     plt.figure(figsize=(10, 5))
#     plt.subplot(1, 2, 1)
#     plt.title("Point Image")
#     plt.imshow(point_image, cmap="gray")
#     plt.axis('off')

#     plt.subplot(1, 2, 2)
#     plt.title("Mask Image")
#     plt.imshow(mask_image, cmap="gray")
#     plt.axis('off')

#     plt.show()
