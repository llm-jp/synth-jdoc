import os
import json
import argparse
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np
from tqdm import tqdm
from PIL import Image, ImageDraw


def apply_paper_texture_fixed(image, config):
    h, w, c = image.shape
    
    intensity = config["intensity"]
    seed = config["seed"]

    rng = np.random.default_rng(seed)

    paper_color_offset = np.array([[-5, -2, 0]])
    base = image.astype(np.int16) + paper_color_offset
    base = np.clip(base, 0, 255).astype(np.uint8)

    noise = rng.normal(0, 10, (h, w, c)).astype(np.float32)

    noisy_image = base.astype(np.float32) + noise * intensity

    return np.clip(noisy_image, 0, 255).astype(np.uint8)


def apply_rotation_fixed(image, config):
    h, w = image.shape[:2]
    
    angle = config["angle"]
    bg_color = tuple(config["bg_color"])

    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)

    rotated = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=bg_color)

    return rotated


def apply_perspective_transform_fixed(image, config):
    h, w = image.shape[:2]
    
    deltas = config["deltas"]
    bg_color = tuple(config["bg_color"])

    # 元の4点
    src_pts = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
    
    # 移動後の4点
    dst_pts = np.float32([
        [deltas[0][0], deltas[0][1]],
        [w - deltas[1][0], deltas[1][1]],
        [deltas[2][0], h - deltas[2][1]],
        [w - deltas[3][0], h - deltas[3][1]]
    ])
    
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped = cv2.warpPerspective(image, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=bg_color)

    return warped


def apply_lighting_gradient_fixed(image, config):
    h, w = image.shape[:2]

    mask = np.ones((h, w), dtype=np.float32)

    h_conf = config["horizontal_gradient"]
    if h_conf["applied"]:
        limit = h_conf["limit"]
        flipped = h_conf["flipped"]

        gradient = np.linspace(limit, 1.0, w)
        if flipped:
            gradient = gradient[::-1]

        mask *= np.tile(gradient, (h, 1))

    v_conf = config["vertical_gradient"]
    if v_conf["applied"]:
        limit = v_conf["limit"]
        flipped = v_conf["flipped"]

        gradient = np.linspace(limit, 1.0, h).reshape(-1, 1)
        if flipped:
            gradient = gradient[::-1]

        mask *= np.tile(gradient, (1, w))

    vig_conf = config["vignette"]
    if vig_conf["applied"]:
        center_x, center_y = vig_conf["center"]
        strength = vig_conf["strength"]

        Y, X = np.ogrid[:h, :w]

        dist_sq = (X - center_x)**2 + (Y - center_y)**2
        max_dist_sq = (w**2 + h**2) / 2

        vignette = 1 - (dist_sq / max_dist_sq) * strength
        vignette = np.clip(vignette, 0, 1.0)

        mask *= vignette

    mask = np.dstack([mask] * 3)
    shaded = image.astype(np.float32) * mask

    return np.clip(shaded, 0, 255).astype(np.uint8)


def apply_blur_fixed(image, config):
    if config["applied"]:
        image = cv2.GaussianBlur(image, (3, 3), 0)
    return image


def reproduce_noise(img, metadata_scan_path, output_path):
    with open(metadata_scan_path, 'r') as f:
        data = json.load(f)

    config = data["config"]

    processed = apply_paper_texture_fixed(img, config["paper_texture"])
    processed = apply_rotation_fixed(processed, config["rotation"])
    processed = apply_perspective_transform_fixed(processed, config["perspective_transform"])
    processed = apply_lighting_gradient_fixed(processed, config["lighting_gradient"])
    processed = apply_blur_fixed(processed, config["blur"])
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, processed)


def mask_img_and_cap_scan(original_image_path, metadata_path, metadata_scan_path, output_path, obj="img_cap"):
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
    
    reproduce_noise(cv_image, metadata_scan_path, output_path)


def main(args):
    with open(args.image_id_file) as f:
        id_list = f.read().strip().split("\n")
    
    original_image_path_list = []
    metadata_path_list = []
    metadata_scan_path_list = []
    output_path_list = []
    for image_id in id_list:
        original_image_path = os.path.join(args.images_dir, image_id[:3], f"{image_id}.png")
        metadata_path = os.path.join(args.metadata_dir, image_id[:3], f"{image_id}.json")
        metadata_scan_path = os.path.join(args.metadata_scan_dir, image_id[:3], f"{image_id}_scan.json")
        output_path = os.path.join(args.output_dir, image_id[:3], f"{image_id}_mask_{args.obj}_scan.png")

        original_image_path_list.append(original_image_path)
        metadata_path_list.append(metadata_path)
        metadata_scan_path_list.append(metadata_scan_path)
        output_path_list.append(output_path)

    length = len(original_image_path_list)
    with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
        list(tqdm(
            executor.map(
                mask_img_and_cap_scan,
                original_image_path_list,
                metadata_path_list,
                metadata_scan_path_list,
                output_path_list,
                [args.obj] * length,
            ),
            total=length,
            desc="Mask images and captions:",
        ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_id_file", type=str, help="noise_type == 1の画像リスト")
    parser.add_argument("--images_dir", type=str, help="Directory containing the original images.")
    parser.add_argument("--metadata_dir", type=str)
    parser.add_argument("--metadata_scan_dir", type=str)
    parser.add_argument("--output_dir", type=str, default="./data/JSSODa/export/ablation/mask_img_cap_scan")
    parser.add_argument("--obj", type=str, default="img_cap", help="mask target", choices=["img_cap", "img", "cap"])
    parser.add_argument("--max_workers", type=int, default=2)

    args = parser.parse_args()

    main(args)
