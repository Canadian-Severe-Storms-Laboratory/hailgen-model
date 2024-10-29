import json
import cv2
import numpy as np


def prepare_dataset():
    """Load and prepare generated training data for model"""
    
    # paths = []
    # images = []
    
    # # Get image file paths
    # with open('dent-segmentation-model/generator/config.json', 'r') as f:
    #     configs = json.load(f)
        
    #     for hailpad_type, config in configs.items():
    #         hailpad_count = config['hailpad_count']
            
    #         for i in range(hailpad_count):
    #             paths.append(f'dent-segmentation-model/generator/output/{hailpad_type}/hailpad_{i}.png')
    
    # for path in paths:
    #     image = cv2.imread(path, cv2.IMREAD_COLOR)
    #     image = image / 255
    #     image = image.astype(np.float32)

import h5py
import numpy as np
import matplotlib.pyplot as plt

"""Test input-output pair creation"""
with h5py.File("", "r") as h5f:
    print("Keys: ", list(h5f.keys()))
    
    first_group_key = list(h5f.keys())[0] 
    first_group = h5f[first_group_key]
    
    point_image = first_group['point_image'][:]
    mask_image = first_group['mask_image'][:]
    
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Point Image")
    plt.imshow(point_image)
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title("Mask Image")
    plt.imshow(mask_image)
    plt.axis('off')

    plt.show()
