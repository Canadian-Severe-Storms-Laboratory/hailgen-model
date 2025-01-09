import os
os.environ['SM_FRAMEWORK'] = 'tf.keras'

import keras
import segmentation_models as sm
from skimage.morphology import skeletonize
import numpy as np
import cv2


def get_points(image):
    '''
    Create points along dent cluster skeletons
    '''

    NUM_POINTS = 5

    points = []
    image_color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    contours, _ = cv2.findContours(
        image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

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
                distances.append(
                    distances[-1] + np.linalg.norm(current_point - previous_point))

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
    Create point-mask pair for input
    '''

    input_data = []
    height, width = image.shape

    for (x, y) in points:
        point_image = np.zeros((height, width, 2), dtype=np.float32)
        point_image[:, :, 0] = image / 255
        point_image[x, y, 1] = 1

        input_data.append(point_image)
        
    return input_data


def get_iou(mask1, mask2):
    '''Get IoU between two mask outputs'''

    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()

    return intersection / union if union > 0 else 0


def refine_outputs(outputs):
    '''
    Filter mask outputs with significant overlap
    '''

    filtered_outputs = []
    THRESHOLD = 0.9

    for output in outputs:
        is_overlapping = False
        for filtered_output in filtered_outputs:
            iou = get_iou(output, filtered_output)
            if iou > THRESHOLD:
                is_overlapping = True
                break
        if not is_overlapping:
            filtered_outputs.append(output)

    return filtered_outputs


def batch_predict(model, input_data, batch_size):
    '''
    Perform predictions on inputs in batches
    '''
    
    predictions = []
        
    for i in range(0, len(input_data), batch_size):
        batch = input_data[i:i+batch_size]
        print(f'Batch input data shape: {batch.shape}')
        predictions.append(model.predict(batch))
        
    return np.concatenate(predictions, axis=0)

def main():
    '''
    Validate model on real hailpad binarizations
    '''
    
    FOLDER_PATH = 'dent-segmentation-model/test-binarizations'
    WEIGHTS_PATH = 'best_model.weights.h5'
    BACKBONE = 'vgg19'
    BATCH_SIZE = 4

    model = sm.Unet(
        BACKBONE,
        encoder_weights=None,
        input_shape=(None, None, 2)
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.00001),
        loss='dice',
        metrics=[sm.metrics.iou_score]
    )
    model.load_weights(WEIGHTS_PATH)

    for file_name in os.listdir(FOLDER_PATH):
        file_path = os.path.join(FOLDER_PATH, file_name)
        
        image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        points = get_points(image)
        
        input_data = prepare_input_data(image, points)
        input_data = np.stack(input_data, axis=0)
        print(f'Input data shape: {input_data.shape}')
        
        predictions = batch_predict(model, input_data, BATCH_SIZE)
        
        for i, prediction in enumerate(predictions):
            np.save(f'predictions-/prediction_{i}', prediction)

            # --- Uncomment to visualize predictions ---        
            # combined_image = None
            # random.seed(42)

            # if combined_image is None:
            #     combined_image = np.zeros((prediction.shape[0], prediction.shape[1], 3), dtype=np.uint8)

            # color = random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)
            # white_pixels = np.where(image == 1)
            # combined_image[white_pixels[0], white_pixels[1]] = color

            # combined_image_bgr = cv2.cvtColor(combined_image, cv2.COLOR_RGB2BGR)

            # cv2.imshow("Combined Predictions", combined_image_bgr)
            # cv2.waitKey(0)


if __name__ == '__main__':
    main()
