import os
import json
import argparse
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np
from tqdm import tqdm
from PIL import Image, ImageDraw
from augraphy import *


INK_NAMES = ("InkBleed", "InkMottling")
PAPER_NAMES = (
    "ColorPaper",
    "DelaunayTessellation",
    "PatternGenerator",
    "VoronoiTessellation",
    "NoiseTexturize",
    "BrightnessTexturize",
) 
POST_NAMES = (
    "DirtyDrum",
    "DirtyRollers",
    "SubtleNoise",
    "Markup",
    "Scribbles",
    "ShadowCast",
    "LightingGradient",
    "Folding",
)


def del_unnecessary_field(params_dict, cls_name):
    if cls_name == "InkBleed":
        params_dict.pop("mask")
        params_dict.pop("keypoints")
        params_dict.pop("bounding_boxes")
        params_dict.pop("numba_jit")
    elif cls_name == "InkMottling":
        params_dict.pop("mask")
        params_dict.pop("keypoints")
        params_dict.pop("bounding_boxes")
        params_dict.pop("numba_jit")
    elif cls_name == "ColorPaper":
        params_dict.pop("mask")
        params_dict.pop("keypoints")
        params_dict.pop("bounding_boxes")
        params_dict.pop("numba_jit")
    elif cls_name == "DelaunayTessellation":
        params_dict.pop("mask")
        params_dict.pop("keypoints")
        params_dict.pop("bounding_boxes")
        params_dict.pop("numba_jit")
        params_dict.pop("width")
        params_dict.pop("height")
        params_dict.pop("n_points")
        params_dict.pop("n_horizontal_points")
        params_dict.pop("n_vertical_points")
        params_dict.pop("perlin")
        params_dict.pop("ws")
    elif cls_name == "PatternGenerator":
        params_dict.pop("mask")
        params_dict.pop("keypoints")
        params_dict.pop("bounding_boxes")
        params_dict.pop("n_rotation")
    elif cls_name == "VoronoiTessellation":
        params_dict.pop("mask")
        params_dict.pop("keypoints")
        params_dict.pop("bounding_boxes")
        params_dict.pop("width")
        params_dict.pop("height")
        params_dict.pop("perlin")
        params_dict.pop("ws")
        params_dict.pop("mult")
        params_dict.pop("num_cells")
    elif cls_name == "NoiseTexturize":
        params_dict.pop("mask")
        params_dict.pop("keypoints")
        params_dict.pop("bounding_boxes")
        params_dict.pop("numba_jit")
    elif cls_name == "BrightnessTexturize":
        params_dict.pop("mask")
        params_dict.pop("keypoints")
        params_dict.pop("bounding_boxes")
        params_dict.pop("numba_jit")
        params_dict.pop("low")
        params_dict.pop("high")
    elif cls_name == "DirtyDrum":
        params_dict.pop("mask")
        params_dict.pop("keypoints")
        params_dict.pop("bounding_boxes")
        params_dict.pop("numba_jit")
    elif cls_name == "DirtyRollers":
        params_dict.pop("mask")
        params_dict.pop("keypoints")
        params_dict.pop("bounding_boxes")
    elif cls_name == "SubtleNoise":
        params_dict.pop("mask")
        params_dict.pop("keypoints")
        params_dict.pop("bounding_boxes")
        params_dict.pop("numba_jit")
    elif cls_name == "Markup":
        params_dict.pop("mask")
        params_dict.pop("keypoints")
        params_dict.pop("bounding_boxes")
        params_dict.pop("numba_jit")
    elif cls_name == "Scribbles":
        params_dict.pop("mask")
        params_dict.pop("keypoints")
        params_dict.pop("bounding_boxes")
        params_dict.pop("numba_jit")
        params_dict.pop("fonts_directory")
    elif cls_name == "ShadowCast":
        params_dict.pop("mask")
        params_dict.pop("keypoints")
        params_dict.pop("bounding_boxes")
        params_dict.pop("numba_jit")
    elif cls_name == "LightingGradient":
        params_dict.pop("mask")
        params_dict.pop("keypoints")
        params_dict.pop("bounding_boxes")
    elif cls_name == "Folding":
        params_dict.pop("mask")
        params_dict.pop("keypoints")
        params_dict.pop("bounding_boxes")
        params_dict.pop("numba_jit")
    return params_dict


def reproduce_augraphy_noise(img, meta_augraphy_path, output_path):
    with open(meta_augraphy_path) as f:
        lines = f.read().strip().split("\n")

    ink_phase = []
    paper_phase = []
    post_phase = []

    # ink_phase
    for i in range(2):
        name, applied, params = lines[i].split(",", 2)
        if applied == "True":
            assert name in INK_NAMES
            augment_class = eval(name)
            params_dict = eval(params)
            params_dict["p"] = 1
            params_dict = del_unnecessary_field(params_dict, name)
            ink_phase.append(augment_class(**params_dict))

    # paper_phase
    i = 2
    appear_aug_seq = False
    while i < len(lines):
        name, applied, params = lines[i].split(",", 2)

        if name == "OneOf":
            if appear_aug_seq:
                # move to post_phase
                break
        elif name == "AugmentationSequence":
            appear_aug_seq = True
        else:
            if applied == "True":
                assert name in PAPER_NAMES
                augment_class = eval(name)
                params_dict = eval(params)
                params_dict["p"] = 1
                params_dict = del_unnecessary_field(params_dict, name)
                paper_phase.append(augment_class(**params_dict))
        i += 1
    
    # post_phase
    while i < len(lines):
        name, applied, params = lines[i].split(",", 2)

        if not (name == "OneOf" or name == "AugmentationSequence"):
            if applied == "True":
                assert name in POST_NAMES
                augment_class = eval(name)
                params_dict = eval(params)
                params_dict["p"] = 1
                params_dict = del_unnecessary_field(params_dict, name)
                post_phase.append(augment_class(**params_dict))
        i += 1

    pipeline = AugraphyPipeline(
        ink_phase=ink_phase,
        paper_phase=paper_phase,
        post_phase=post_phase,
        overlay_alpha=0.3,
        fixed_dpi=True,
    )

    data = pipeline.augment(img)
    augmented_image = data["output"]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, augmented_image)


def mask_img_and_cap_augraphy(original_image_path, metadata_path, meta_augraphy_path, output_path, obj="img_cap"):
    with open(metadata_path) as f:
        metadata = json.load(f)

    img = Image.open(original_image_path)
    draw = ImageDraw.Draw(img)

    fill_color = metadata["style"]["figure_bg_color"]
    elements = metadata.get("elements", [])
    for el in elements:
        if el.get("type") == "image":
            bboxes = el.get("bboxes", [])
            for bbox in bboxes:
                # 画像かcaptionの一方だけをマスクする場合は、textが空かどうか調べる
                if obj == "img" and not bbox["text"] == "":
                    continue
                elif obj == "cap" and bbox["text"] == "":
                    continue
                x = bbox["x"]
                y = bbox["y"]
                w = bbox["width"]
                h = bbox["height"]
                
                rect_coords = [x, y, x + w, y + h]
                draw.rectangle(rect_coords, fill=fill_color)

    cv_image = np.array(img, dtype=np.uint8)
    cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)
    
    reproduce_augraphy_noise(cv_image, meta_augraphy_path, output_path)
    return


def main(args):
    with open(args.image_id_file) as f:
        id_list = f.read().strip().split("\n")
    
    original_image_path_list = []
    metadata_path_list = []
    metadata_augraphy_path_list = []
    output_path_list = []
    for image_id in id_list:
        original_image_path = os.path.join(args.images_dir, image_id[:3], f"{image_id}.png")
        metadata_path = os.path.join(args.metadata_dir, image_id[:3], f"{image_id}.json")
        metadata_augraphy_path = os.path.join(args.metadata_augraphy_dir, image_id[:3], f"{image_id}_augraphy.txt")
        output_path = os.path.join(args.output_dir, image_id[:3], f"{image_id}_mask_{args.obj}_augraphy.png")

        original_image_path_list.append(original_image_path)
        metadata_path_list.append(metadata_path)
        metadata_augraphy_path_list.append(metadata_augraphy_path)
        output_path_list.append(output_path)

    length = len(original_image_path_list)
    with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
        list(tqdm(
            executor.map(
                mask_img_and_cap_augraphy,
                original_image_path_list,
                metadata_path_list,
                metadata_augraphy_path_list,
                output_path_list,
                [args.obj] * length,
            ),
            total=length,
            desc="Mask images and captions:",
        ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_id_file", type=str, help="noise_type == 2の画像リスト")
    parser.add_argument("--images_dir", type=str, help="Directory containing the original images.")
    parser.add_argument("--metadata_dir", type=str)
    parser.add_argument("--metadata_augraphy_dir", type=str)
    parser.add_argument("--output_dir", type=str, default="./data/JSSODa/export/ablation/mask_img_cap_augraphy")
    parser.add_argument("--obj", type=str, default="img_cap", help="mask target", choices=["img_cap", "img", "cap"])
    parser.add_argument("--max_workers", type=int, default=2)

    args = parser.parse_args()

    main(args)
