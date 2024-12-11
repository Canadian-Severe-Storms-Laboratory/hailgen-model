import os
os.environ['SM_FRAMEWORK'] = 'tf.keras'

import cv2
import numpy as np
from skimage.morphology import skeletonize
import segmentation_models as sm


input_data = []
FOLDER_PATH = 'dent-segmentation-model/test-binarizations'
WEIGHTS_PATH = ''
BACKBONE = 'vgg19'


def get_points(image):
    '''
    Create points along dent cluster skeletons
    '''
    
    NUM_POINTS = 5
    
    points = []
    image_color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        cluster_mask = np.zeros_like(image, dtype=np.uint8)
        cv2.drawContours(cluster_mask, [contour], -1, 255, -1)
        
        binary_dent = cluster_mask > 0 
        skeleton = skeletonize(binary_dent)
        skeleton = (skeleton * 255).astype(np.uint8)
        skeleton_points = np.column_stack(np.where(skeleton > 0))

        if len(skeleton_points) < NUM_POINTS:
            selected_points = skeleton_points
        else:
            distances = [0]
            for i in range(1, len(skeleton_points)):
                previous_point = skeleton_points[i - 1]
                current_point = skeleton_points[i]
                distances.append(distances[-1] + np.linalg.norm(current_point - previous_point))

            distances = np.array(distances) / distances[-1]

            target_distances = np.linspace(0, 1, NUM_POINTS)
            selected_points = []
            for target in target_distances:
                index = np.searchsorted(distances, target)
                selected_points.append(skeleton_points[index])

        selected_points = [(int(x), int(y)) for y, x in selected_points]

        for point in selected_points:
            points.append(point)
            # cv2.circle(image_color, point, 3, (0, 0, 255), -1) # Uncomment to visualize points
            

    # --- Uncomment to visualize points ---
    # cv2.imshow('Point Queries', image_color)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    
    return points


def prepare_input_data(image, points):
    '''
    Create point-mask pairs for inputs
    '''
        
    height, width = image.shape # TODO: Determine if need to resize from 1000x1000 to 256x256 (?)
    
    for (x, y) in points:
        point_image = np.zeros((height, width, 2), dtype=np.float32)
        point_image[:, :, 0] = image
        point_image[x, y, 1] = 1

        input_data.append(point_image)
        
    
def main():
    '''
    Validate model on real hailpad binarizations
    '''
    
    model = sm.Unet(
        BACKBONE,
        encoder_weights=None,
        input_shape=(None, None, 2)
    )
    model.compile(optimizer='Adam', loss='dice', metrics=[sm.metrics.iou_score])
    model.load_weights(WEIGHTS_PATH)
    
    for file_name in os.listdir(FOLDER_PATH):
        file_path = os.path.join(FOLDER_PATH, file_name)
        image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        points = get_points(image)
        input_data = prepare_input_data(image, points)
        predictions = model.predict(input_data)
    

    # Idea:
        # - Iteratively load files from test-binarizations folder
        # - For each binarization, get/fill contours to create individual cluster masks
        # - For each mask, fit an ellipse and get major // minor (TBD)
        # - Subdivide mask by major // minor and create point at each subdivision centroid (or: use skeleton to get x points along contour)
        # - Invoke prepare_test_data to prepare as input data
        # - Run model on input data to get dent mask predictions
        # - Compare mask predicions; if 90% overlap b/w any 2 masks, keep the larger and discard the smaller
        # - Fit ellipses to final predicted masks and output a single image


if __name__ == '__main__':
    main()
