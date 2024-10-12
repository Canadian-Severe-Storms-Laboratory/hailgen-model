import json
import cv2
import numpy as np
import noise


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
        mask = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)

        wind_angle = np.random.rand() * 360

        if exp:
            beta = -(diameter_range[1])/np.log(1 - 0.999)
            diameter_dist = np.random.exponential(scale=beta, size=dent_count)
            diameter_dist = diameter_dist[diameter_dist >= diameter_range[0]]
        else:
            diameter_dist = np.random.uniform(
                diameter_range[0], diameter_range[1], size=dent_count)

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
            
            dent_mask = np.zeros_like(mask)
            cv2.drawContours(dent_mask, [dent], -1, 255, -1)

            overlap = cv2.bitwise_and(mask, dent_mask)
            non_overlap = cv2.bitwise_and(dent_mask, cv2.bitwise_not(overlap))

            # Label overlapping regions in red
            image[overlap == 255] = (0, 0, 255)
            image[non_overlap == 255] = (255, 255, 255)

            mask = cv2.bitwise_or(mask, dent_mask)

        cv2.imwrite(f"{directory}/masks/hailpad_{i}.png", image) # Ground truth
        
        image[np.all(image == [0, 0, 255], axis=-1)] = [255, 255, 255] # Normal binary image
        cv2.imwrite(f"{directory}/hailpad_{i}.png", image)
        

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
