import os
os.environ['SM_FRAMEWORK'] = 'tf.keras'

import keras
import segmentation_models as sm
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
import numpy as np
# import matplotlib.pyplot as plt
import cv2
import random
# from dataset import Dataset, DatasetLoader


def get_random_color():
    '''
    Auxiliary function that returns a random color for visualization purposes
    '''

    return random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)


def get_points(image):
    '''
    Create point queries using K-Means clustering to identify distinct regions in dent clusters
    '''
    MAX_CLUSTER_SIZE = 5
    points = []
    cluster_image = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
    
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        cluster_mask = np.zeros_like(image, dtype=np.uint8)
        cv2.drawContours(cluster_mask, [contour], -1, 255, -1)
        
        cluster_points = np.column_stack(np.where(cluster_mask > 0))
        
        if len(cluster_points) < MAX_CLUSTER_SIZE:
            continue
        
        color = get_random_color()
        for point in cluster_points:
            cluster_image[point[1], point[0]] = color
        
        n_points = len(cluster_points) // 20
        n_points = max(n_points, 1)
        
        kmeans = KMeans(n_clusters=n_points, random_state=42)
        kmeans.fit(cluster_points)
        
        for center in kmeans.cluster_centers_:
            points.append((int(center[0]), int(center[1])))
            cv2.circle(cluster_image, (int(center[0]), int(center[1])), 1, (255, 255, 255), -1)
    
    return points, cluster_image


def prepare_input_data(image, points):
    '''
    Create 2-channel input samples
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
    '''
    Get IoU between two mask outputs
    '''

    mask1 = mask1 == 1
    mask2 = mask2 == 1 

    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()

    return intersection / union if union > 0 else 0

import numpy as np
from sklearn.cluster import DBSCAN

def refine_outputs(masks):
    '''
    Perform DBSCAN clustering to filter out mask outputs with significant overlap (prioritizing smaller prediction masks)
    '''

    n = len(masks)
    dist_mat = np.zeros((n, n), dtype=float)

    masks = [mask > 0.5 for mask in masks]  

    for i in range(n):
        for j in range(i + 1, n):
            and_count = float(np.bitwise_and(masks[i], masks[j]).sum())
            or_count = float(np.bitwise_or(masks[i], masks[j]).sum() + 1)

            dist = 1.0 - and_count / or_count
            
            dist_mat[i, j] = dist
            dist_mat[j, i] = dist

    dbscan = DBSCAN(eps=0.3, min_samples=1, metric='precomputed')
    labels = dbscan.fit(dist_mat).labels_

    filtered_masks = []

    for label in set(labels):
        cluster_predictions = [masks[i] for i in range(n) if labels[i] == label]
        if label == -1:
            filtered_masks.extend(cluster_predictions)
        else:
            smallest_prediction = min(cluster_predictions, key=lambda x: np.sum(x))
            filtered_masks.append(smallest_prediction)

    return filtered_masks


def batch_predict(model, input_data, batch_size):
    '''
    Perform predictions on sample binarization inputs in batches
    '''
    
    predictions = []
        
    for i in range(0, len(input_data), batch_size):
        batch = input_data[i:i+batch_size]
        print(f'Batch input data shape: {batch.shape}')
        predictions.append(model.predict(batch))
        
    return np.concatenate(predictions, axis=0)


def test_model_accuracy(model, dataset_loader):
    '''
    Perform predictions on test dataset and assess against ground truth masks
    '''

    total_iou = 0
    total_samples = 0
    results = []

    for batch_points, batch_true_masks in dataset_loader:
        batch_predictions = model.predict(batch_points)

        for prediction, ground_truth in zip(batch_predictions, batch_true_masks):
            iou = get_iou(prediction, ground_truth)
            results.append({
                'iou': iou,
                'prediction': prediction,
                'ground_truth': ground_truth
            })
            total_iou += iou
            total_samples += 1

    # --- Uncomment to visualize predictions ---
    # for i, result in enumerate(results):
    #     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    #     ax1.imshow(result['prediction'], cmap='gray')
    #     ax1.set_title(f'Prediction (IoU: {result["iou"]:.3f})')
    #     ax1.axis('off')

    #     ax2.imshow(result['ground_truth'], cmap='gray')
    #     ax2.set_title('Ground Truth')
    #     ax2.axis('off')

    return total_iou / total_samples


def main():
    '''
    Sample hailpad binarizations and save predictions as Numpy/PNG data
    '''
    
    FOLDER_PATH_1 = 'dent-segmentation-model/test-dataset'
    FOLDER_PATH_2 = 'dent-segmentation-model/test-binarizations'
    WEIGHTS_PATH = 'dent-segmentation-model/best_model.weights.h5'
    BACKBONE = 'vgg19'
    BATCH_SIZE = 4

    random.seed(42)

    # --- Uncomment to run validation dataset ---
    # print('Preparing validation dataset...')
    # test_paths = [os.path.join(FOLDER_PATH_1, f) for f in os.listdir(FOLDER_PATH_1) if f.endswith('.h5')]
    # test_dataset = Dataset(test_paths)
    # test_dataset_loader = DatasetLoader(test_dataset)

    print('Compiling model...')
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

    # --- Uncomment to run validation dataset ---
    # print('Running model on validation dataset...')
    # results_iou = test_model_accuracy(model, test_dataset_loader)
    # print(f'Mean IoU score is {results_iou}')

    print('Running model on sample hailpads...')
    for file_name in os.listdir(FOLDER_PATH_2):
        file_path = os.path.join(FOLDER_PATH_2, file_name)
        
        image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        points, cluster_image = get_points(image)
        
        input_data = prepare_input_data(image, points)
        input_data = np.stack(input_data, axis=0)
        print(f'Input data shape: {input_data.shape}')
        
        predictions = refine_outputs(batch_predict(model, input_data, BATCH_SIZE))
        output_dir = f'dent-segmentation-model/predictions/predictions-{file_name}'
        os.makedirs(f'{output_dir}/predictions-visualized', exist_ok=True)
        
        combined_image = None

        for i, prediction in enumerate(predictions):
            np.save(f'{output_dir}/prediction_{i}', prediction)

            if combined_image is None:
                combined_image = np.zeros((prediction.shape[0], prediction.shape[1], 3), dtype=np.uint8)

            single_image = np.zeros((prediction.shape[0], prediction.shape[1], 4), dtype=np.uint8)
            single_image[:, :, 3] = 0

            color = get_random_color()
            white_pixels = np.where(prediction == 1)
            single_image[white_pixels[0], white_pixels[1]] = [*color, 255]
            combined_image[white_pixels[0], white_pixels[1]] = color

            output_path = os.path.join(f'{output_dir}/predictions-visualized', f'prediction_{i}.png')
            cv2.imwrite(output_path, single_image)
        
        combined_image_bgr = cv2.cvtColor(combined_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(f'{output_dir}/predictions-visualized/predictions_combined.png', combined_image_bgr)
        cv2.imwrite(f'{output_dir}/predictions-visualized/clusters.png', cluster_image)


if __name__ == '__main__':
    main()
