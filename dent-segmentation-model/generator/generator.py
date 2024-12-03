import json
import cv2
import numpy as np
import noise
from scipy.ndimage import label
import h5py
import matplotlib.pyplot as plt


def main():
    '''
    Run simulated hailpad generation pipeline based on config data
    '''

    with open('dent-segmentation-model/generator/config.json', 'r') as f:
        configs = json.load(f)

    for hailpad_type, config in configs.items():
        print(f'Generating hailpads for [Category {hailpad_type}]...')
        generate(
            diameter_range=tuple(config['diameter_range']),
            axis_variation=config['axis_variation'],
            dent_count=config['dent_count'],
            exp=config['exp'],
            hailpad_count=config['hailpad_count'],
            hailpad_type=hailpad_type,
            directory=config['directory']
        )


def generate(diameter_range: [float, float],
             axis_variation: float,
             dent_count: int,
             exp: bool,
             hailpad_count: int,
             hailpad_type: str,
             directory: str):
    '''
    Generate simulated hailpad depth map binarizations based on config data
    '''

    IMAGE_SIZE = 1000
    SCALE = 0.3

    for i in range(hailpad_count):
        print(f'[Category {hailpad_type}] {i + 1}/{hailpad_count}')
        image = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 1), dtype=np.float32)

        wind_angle = np.random.rand() * 360

        if exp:
            # Exponential approximation: https://stackoverflow.com/questions/69828806/scaling-numpy-exponential-random-generator
            beta = -(diameter_range[1])/np.log(1 - 0.999)
            diameter_dist = np.random.exponential(scale=beta, size=dent_count)
            diameter_dist = diameter_dist[diameter_dist >= diameter_range[0]]
        else:
            diameter_dist = np.random.uniform(
                diameter_range[0], diameter_range[1], size=dent_count)
            
        masks = []

        for j in range(len(diameter_dist)):
            x = np.random.randint(0, IMAGE_SIZE)
            y = np.random.randint(0, IMAGE_SIZE)
            
            # TODO: Determine axis ratio to classify as wind-driven dent
            major_axis = diameter_dist[j]
            minor_axis = major_axis * (1 - axis_variation * np.random.rand())

            if (minor_axis / major_axis) <= 0.5:
                angle = wind_angle
            else:
                angle = np.random.rand() * 360

            dent = create_dent(
                x, y, major_axis, minor_axis, angle, SCALE)
            
            dent_mask = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 1), dtype=np.float32)
            cv2.drawContours(dent_mask, [dent], -1, 1, -1)
            
            dent_mask = cv2.resize(dent_mask, (256, 256), interpolation=cv2.INTER_AREA)
            _, dent_mask = cv2.threshold(dent_mask, 0, 1, cv2.THRESH_BINARY)
            
            masks.append(dent_mask)
            
            cv2.drawContours(image, [dent], -1, 1, -1)
         
        image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_AREA)
        _, image = cv2.threshold(image, 0, 1, cv2.THRESH_BINARY)

        create_point_query(image, masks, directory, i)
        cv2.imwrite(f'{directory}/hailpad_{i}.png', image * 255) # TODO: Remove


def create_dent(cx: float,
                cy: float,
                major_axis: float,
                minor_axis: float,
                angle: float,
                scale: float):
    '''
    Create irregular dent shape from base ellipse using Perlin noise along ellipse segments
    '''

    NUM_POINTS = 50
    points = []

    for i in range(NUM_POINTS):
        theta = (2 * np.pi / NUM_POINTS) * i
        x = (major_axis / 2) * np.cos(theta)
        y = (minor_axis / 2) * np.sin(theta)

        x_rot = x * np.cos(np.radians(angle)) - y * np.sin(np.radians(angle))
        y_rot = x * np.sin(np.radians(angle)) + y * np.cos(np.radians(angle))

        noise_value = noise.pnoise2(x_rot * scale,
                                    y_rot * scale,
                                    octaves=6,
                                    lacunarity=0.5,
                                    persistence=1)
        x_noise = x_rot * (1 + noise_value)
        y_noise = y_rot * (1 + noise_value)

        x_final = int(cx + x_noise)
        y_final = int(cy + y_noise)

        points.append([x_final, y_final])

    return np.array(points, dtype=np.int32)


def create_point_query(image,
                       masks,
                       directory: str,
                       hailpad_index: int):
    '''
    Add a second channel for random point queries on each dent and create the corresponding point-dent pairs;
    for clustered dents, the largest dent is chosen if a point lands in the overlapping region
    '''
    
    NUM_POINTS = 2 # Number of randomly-selected points per mask

    height, width = image.shape
        
    combined_mask = np.sum(masks, axis=0)
    overlapping_region = (combined_mask > 1)
        
    with h5py.File(f'{directory}/hailpad_{hailpad_index}.h5', 'w') as h5f:
        for mask_index, mask in enumerate(masks):           
            mask_pixels = np.column_stack(np.where(mask == 1))
            
            point_indices = np.random.choice(len(mask_pixels), size=NUM_POINTS, replace=False)
            points = mask_pixels[point_indices]
            
            for point_index, point in enumerate(points):
                x, y = point[:2]
            
                if overlapping_region[x, y]:
                    largest_mask = max(
                        (m for m in masks if m[x, y] == 1),
                        key=lambda m: np.sum(m == 1)
                    )
                    selected_mask = largest_mask
                else:
                    selected_mask = mask
                
                point_image = np.zeros((height, width, 2), dtype=np.float32)
                point_image[:, :, 0] = image
                point_image[x, y, 1] = 1
                
                point_group = h5f.create_group(f'point_mask_{mask_index}_{point_index}')
                point_group.create_dataset('point', data=point_image)
                point_group.create_dataset('mask', data=selected_mask)


if __name__ == '__main__':
    main()
