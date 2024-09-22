import json
import cv2
import numpy as np


def generate(diameter_range: [float, float],
             axis_variation: float,
             dent_count: int,
             exp: bool,
             hailpad_count: int,
             directory: str):
    """Generate simulated hailpad depth map binarizations based on config data"""

    IMAGE_SIZE = 1000

    for i in range(hailpad_count):
        image = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)

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

            # TODO: Wind-driven hail considerations:
            #   - Determine axis ratio to classify as wind-driven dent
            #   - Determine if wind-driven dents should be of similar angles
            major_axis = diameter_dist[j]
            minor_axis = major_axis * (1 - axis_variation * np.random.rand())
            angle = np.random.rand() * 360

            cv2.ellipse(
                image,
                (x, y),
                (int(major_axis / 2), int(minor_axis / 2)),
                angle,
                0,
                360,
                (255, 255, 255),
                -1
            )

        cv2.imwrite(f"{directory}/hailpad_{i}.png", image)


with open('training-model/generator/config.json', 'r') as f:
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
