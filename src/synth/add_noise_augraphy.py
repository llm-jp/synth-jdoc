import os
import random
import argparse
import colorsys
from glob import glob
from concurrent.futures import ProcessPoolExecutor

import cv2
import matplotlib.font_manager
from tqdm import tqdm
from augraphy import *


def get_hsv_vibrant_color():
    h = random.random()
    s = random.uniform(0.5, 0.8)
    v = 1.0

    r, g, b = colorsys.hsv_to_rgb(h, s, v)

    r = int(r * 255)
    g = int(g * 255)
    b = int(b * 255)

    return (b, g, r)


def make_pipeline():
    ink_phase = [
        InkBleed(
            intensity_range=(0.2, 0.4),
            kernel_size=random.choice([(5, 5), (3, 3)]),
            severity=(0.2, 0.4),
            p=0.333,
        ),
        InkMottling(
            ink_mottling_alpha_range=(0.1, 0.2),
            ink_mottling_noise_scale_range=(1, 1),
            ink_mottling_gaussian_kernel_range=(3, 5),
            p=0.333,
        )
    ]

    paper_phase = [
        ColorPaper(
            hue_range=(28, 45),
            saturation_range=(10, 40),
            p=0.666,
        ),
        OneOf(
            [
                DelaunayTessellation(
                    n_points_range=(500, 800),
                    n_horizontal_points_range=(500, 800),
                    n_vertical_points_range=(500, 800),
                    noise_type="random",
                    color_list="default",
                    color_list_alternate="default",
                ),
                PatternGenerator(
                    imgx=random.randint(256, 512),
                    imgy=random.randint(256, 512),
                    n_rotation_range=(10, 15),
                    color=(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)),
                    alpha_range=(0.25, 0.5),
                ),
                VoronoiTessellation(
                    mult_range=(50, 80),
                    seed=random.randint(0, 2**32 - 1),
                    num_cells_range=(500, 1000),
                    noise_type="random",
                    background_value=(200, 255),
                ),
            ],
            p=0.5,
        ),
        AugmentationSequence(
            [
                NoiseTexturize(
                    sigma_range=(3, 10),
                    turbulence_range=(2, 5),
                    texture_width_range=(100, 500),
                    texture_height_range=(100, 500),
                    p=0.8,
                ),
                BrightnessTexturize(
                    texturize_range=(0.8, 0.99),
                    deviation=0.03,
                    p=0.8,
                ),
            ],
            p=0.8,
        ),
    ]

    post_phase = [
        OneOf(
            [
                DirtyDrum(
                    line_width_range=(1, 1),
                    line_concentration=random.uniform(0.05, 0.1),
                    direction=random.randint(0, 2),
                    noise_intensity=random.uniform(0.1, 0.3),
                    noise_value=(0, 15),
                    ksize=(3, 3),
                    sigmaX=0,
                ),
                DirtyRollers(
                    line_width_range=(1, 1),
                    scanline_type=0,
                ),
            ],
            p=0.333,
        ),
        SubtleNoise(
            subtle_range=random.randint(4, 6),
            p=0.9,
        ),
        OneOf(
            [
                Markup(
                    num_lines_range=(2, 7),
                    markup_length_range=(0.5, 1),
                    markup_thickness_range=(1, 1),
                    markup_type=random.choice(["strikethrough", "crossed", "highlight", "underline"]),
                    markup_ink=random.choice(["pencil", "pen", "marker", "highlighter"]),
                    markup_color=get_hsv_vibrant_color(),
                    large_word_mode=random.choice([True, False]),
                    single_word_mode=False,
                    repetitions=1,
                ),
                Scribbles(
                    scribbles_type=random.choice(["lines", "texts"]),
                    scribbles_ink=random.choice(["pencil", "pen", "marker", "highlighter"]),
                    scribbles_location="random",
                    scribbles_size_range=(200, 350),
                    scribbles_count_range=(1, 6),
                    scribbles_thickness_range=(1, 1),
                    scribbles_brightness_change=[32, 64, 128],
                    scribbles_color=get_hsv_vibrant_color(),
                    scribbles_text="random",
                    scribbles_text_font="random",
                    scribbles_text_rotate_range=(0, 360),
                    scribbles_lines_stroke_count_range=(1, 4),
                ),
            ],
            p=0.5,
        ),
        OneOf(
            [
                ShadowCast(
                    shadow_side=random.choice(["left", "right", "top", "bottom"]),
                ),
                LightingGradient(
                    light_position=None,
                    max_brightness=255,
                    min_brightness=0,
                    mode=random.choice(("gaussian", "linear_static")),
                ),
            ],
            p=0.666,
        ),
        Folding(
            fold_count=random.choice((1, 2)),
            fold_noise=0.0,
            fold_angle_range=(-360, 360),
            gradient_width=(0.05, 0.15),
            gradient_height=(0.005, 0.015),
            backdrop_color=(random.randint(0, 10), random.randint(0, 10), random.randint(0, 10)),
            p=0.333,
        ),
    ]

    return AugraphyPipeline(
        ink_phase=ink_phase,
        paper_phase=paper_phase,
        post_phase=post_phase,
        overlay_alpha=0.3,
        fixed_dpi=True,
    )


def run(image_path, output_dir, config_output_dir):
    filename = os.path.basename(image_path)
    base, ext = os.path.splitext(filename)
    dir_base_name = os.path.basename(os.path.dirname(image_path))
    output_path = os.path.join(output_dir, dir_base_name, base + "_augraphy" + ext)
    config_output_path =  os.path.join(config_output_dir, dir_base_name, base + "_augraphy.txt")

    if os.path.isfile(output_path):
        return

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    os.makedirs(os.path.dirname(config_output_path), exist_ok=True)

    image = cv2.imread(image_path)

    assert image is not None

    pipeline = make_pipeline()

    data = pipeline.augment(image)
    augmented_image = data["output"]
    augmentation_name = data['log']['augmentation_name']
    augmentation_status = data['log']['augmentation_status']
    augmentation_parameters = data['log']['augmentation_parameters']

    cv2.imwrite(output_path, augmented_image)

    with open(config_output_path, "w", encoding="utf-8") as f:
        for (name, status, parameters) in zip(
            augmentation_name,
            augmentation_status,
            augmentation_parameters
        ):
            if isinstance(parameters, dict):
                parameters.pop("results", None)
            elif isinstance(parameters, list):
                for parameter in parameters:
                    parameter.pop("results", None)
            f.write(f"{name},{status},{parameters}\n")
    
    return


def make_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir",
        type=str,
        default="data/JSSODa/export/images/",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/JSSODa/export/images_augraphy/",
    )
    parser.add_argument(
        "--config_output_dir",
        type=str,
        default="data/JSSODa/export/metadata_augraphy/",
    )
    parser.add_argument(
        "--num_processes",
        type=int,
        default=4,
    )
    return parser.parse_args()


def main():
    args = make_args()
    input_dir = args.input_dir
    output_dir = args.output_dir
    config_output_dir = args.config_output_dir
    num_processes = args.num_processes
    
    image_paths = sorted(glob(os.path.join(input_dir, "**", "*.png"), recursive=True))
    length = len(image_paths)
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        list(tqdm(
            executor.map(
                run,
                image_paths,
                [output_dir for _ in range(length)],
                [config_output_dir for _ in range(length)],
            ),
            total=length,
            desc="Adding noise",
        ))


if __name__ == "__main__":
    main()