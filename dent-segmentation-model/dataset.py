import json
import cv2
import numpy as np


def prepare_dataset():
    """Load and prepare generated training images for model"""
    
    paths = []
    imgs = []
    
    # Get image file paths
    with open('dent-segmentation-model/generator/config.json', 'r') as f:
        configs = json.load(f)
        
        for hailpad_type, config in configs.items():
            hailpad_count = config['hailpad_count']
            
            for i in range(hailpad_count):
                paths.append(f'dent-segmentation-model/generator/output/{hailpad_type}/hailpad_{i}.png')
    
    for path in paths:
        # TODO:
        # - Determine whether 1000x1000 is appropriate or if should be resized
        # - Ground truth masks
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        img = img / 255
        img = img.astype(np.float32)
        
        imgs.append(img)
        