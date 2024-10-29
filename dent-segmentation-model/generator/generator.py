import json
import cv2
import numpy as np
import noise
import h5py
import matplotlib.pyplot as plt


def generate(diameter_range: [float, float],
             axis_variation: float,
             dent_count: int,
             exp: bool,
             hailpad_count: int,
             directory: str):
    """Generate simulated hailpad depth map binarizations based on config data"""

    IMAGE_SIZE = 1000
    SCALE = 0.3

    for i in range(hailpad_count):
        image = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)

        wind_angle = np.random.rand() * 360

        if exp:
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
            
            dent_mask = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
            cv2.drawContours(dent_mask, [dent], -1, (255, 255, 255), -1)
            masks.append(dent_mask)
            
            print(len(masks))
            
            cv2.drawContours(image, [dent], -1, (255, 255, 255), -1)
                        
        # image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_AREA)
        # _, image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY) # TODO
                
        create_point_query(image, masks, directory, i)
        cv2.imwrite(f"{directory}/hailpad_{i}.png", image) # TODO: Remove
        

def create_dent(cx, cy, major_axis, minor_axis, angle, scale):
    """Helper function to create irregular dent shape from base ellipse using Perlin noise along ellipse segments"""

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


def create_point_query(image, masks, directory, hailpad_index):
    """Helper function to add a fourth channel for random point queries and create the corresponding dent mask output"""

    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height, width = image.shape
    
    dent_pixels = np.column_stack(np.where(image == 255))
    NUM_POINTS = 100
    
    point_indices = np.random.choice(len(dent_pixels), size=NUM_POINTS, replace=False)
    points = dent_pixels[point_indices]
        
    with h5py.File(f"{directory}/hailpad_{hailpad_index}.h5", "w") as h5f:      
        for point_index, point in enumerate(points):
            x, y = point[:2]
            
            point_image = np.zeros((height, width, 4), dtype=np.uint8)
            point_image[:, :, 0] = image
            point_image[x, y, 3] = 1
            
            for mask_index, mask in enumerate(masks):
                if np.array_equal(mask[x, y], [255, 255, 255]):
                    point_group = h5f.create_group(f"point_{point_index}_dent_{mask_index}")
                    point_group.create_dataset("point_image", data=point_image)
                    point_group.create_dataset("mask_image", data=mask)


with open('dent-segmentation-model/generator/config.json', 'r') as f:
    configs = json.load(f)

for hailpad_type, config in configs.items():
    print(f"Generating hailpads for Category {hailpad_type}...")
    generate(
        diameter_range=tuple(config['diameter_range']),
        axis_variation=config['axis_variation'],
        dent_count=config['dent_count'],
        exp=config['exp'],
        hailpad_count=config['hailpad_count'],
        directory=config['directory']
    )