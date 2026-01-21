import os
import random
import argparse
import json
from glob import glob
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np
from tqdm import tqdm


def get_random_background_color(thresh: float = 0.5):
    rand = random.random()

    if rand < thresh:
        r = random.randint(245, 255)
        g = random.randint(245, 255)
        b = random.randint(245, 255)
        return (r, g, b)
    else:
        r = random.randint(0, 10)
        g = random.randint(0, 10)
        b = random.randint(0, 10)
        return (r, g, b)


def apply_paper_texture(image, intensity=0.1):
    h, w, c = image.shape

    seed = random.randint(0, 2**32 - 1)
    rng = np.random.default_rng(seed)
    
    # 紙の色を修正
    paper_color_offset = np.array([[-5, -2, 0]])

    base = image.astype(np.int16) + paper_color_offset
    base = np.clip(base, 0, 255).astype(np.uint8)

    # ガウシアンノイズ
    noise = rng.normal(0, 10, (h, w, c)).astype(np.float32)

    noisy_image = base.astype(np.float32) + noise * intensity

    config = {
        "intensity": intensity,
        "seed": seed
    }

    return np.clip(noisy_image, 0, 255).astype(np.uint8), config


def apply_rotation(image, max_angle=1.5, bg_color=(255, 255, 255)):
    h, w = image.shape[:2]
    angle = random.uniform(-max_angle, max_angle)

    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)

    rotated = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=bg_color)

    config = {
        "angle": angle,
        "bg_color": bg_color
    }

    return rotated, config


def apply_perspective_transform(image, magnitude=10, bg_color=(255, 255, 255)):
    h, w = image.shape[:2]
    
    # 元の4点
    src_pts = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
    
    # 移動後の4点（ランダムに少しずらす）
    deltas = [
        [random.randint(-magnitude//2, magnitude), random.randint(-magnitude//2, magnitude)],
        [random.randint(-magnitude//2, magnitude), random.randint(-magnitude//2, magnitude)],
        [random.randint(-magnitude//2, magnitude), random.randint(-magnitude//2, magnitude)],
        [random.randint(-magnitude//2, magnitude), random.randint(-magnitude//2, magnitude)]
    ]
    dst_pts = np.float32([
        [deltas[0][0], deltas[0][1]],
        [w - deltas[1][0], deltas[1][1]],
        [deltas[2][0], h - deltas[2][1]],
        [w - deltas[3][0], h - deltas[3][1]]
    ])
    
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped = cv2.warpPerspective(image, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=bg_color)

    config = {
        "magnitude": magnitude,
        "deltas": deltas,
        "bg_color": bg_color
    }

    return warped, config


def apply_lighting_gradient(image, h_thresh: float = 0.7, v_thresh: float = 0.7, vignette_thresh: float = 0.333):
    """
    画像に影を加える。
    縦、横、ビネット効果（スポットライト）をランダムに組み合わせる。
    """
    h, w = image.shape[:2]

    mask = np.ones((h, w), dtype=np.float32)
    config = {}

    # 横方向
    use_h = random.random() < h_thresh
    config["horizontal_gradient"] = {
        "applied": use_h,
        "limit": None,
        "flipped": None,
    }
    if use_h:
        limit = random.uniform(0.7, 0.9)
        is_flipped = random.random() < 0.5

        config["horizontal_gradient"]["limit"] = limit
        config["horizontal_gradient"]["flipped"] = is_flipped

        gradient = np.linspace(limit, 1.0, w)
        # 左右反転
        if is_flipped:
            gradient = gradient[::-1]

        mask *= np.tile(gradient, (h, 1))

    # 縦方向
    use_v = random.random() < v_thresh
    config["vertical_gradient"] = {
        "applied": use_v,
        "limit": None,
        "flipped": None,
    }
    if use_v:
        limit = random.uniform(0.7, 0.9)
        is_flipped = random.random() < 0.5

        config["vertical_gradient"]["limit"] = limit
        config["vertical_gradient"]["flipped"] = is_flipped

        gradient = np.linspace(limit, 1.0, h).reshape(-1, 1)
        # 上下反転
        if is_flipped:
            gradient = gradient[::-1]

        mask *= np.tile(gradient, (1, w))

    # ビネット
    use_vignette = random.random() < vignette_thresh
    config["vignette"] = {
        "applied": use_vignette,
        "center": None,
        "strength": None,
    }
    if use_vignette:
        # 中心座標
        center_x = random.randint(w // 3, 2 * w // 3)
        center_y = random.randint(h // 3, 2 * h // 3)
        strength = random.uniform(0.2, 0.5)

        config["vignette"]["center"] = (center_x, center_y)
        config["vignette"]["strength"] = strength

        Y, X = np.ogrid[:h, :w]

        # 中心からの二乗距離
        dist_sq = (X - center_x)**2 + (Y - center_y)**2
        max_dist_sq = (w**2 + h**2) / 2

        # 距離に応じて暗くする
        vignette = 1 - (dist_sq / max_dist_sq) * strength
        vignette = np.clip(vignette, 0, 1.0)

        mask *= vignette

    mask = np.dstack([mask] * 3)

    shaded = image.astype(np.float32) * mask

    return np.clip(shaded, 0, 255).astype(np.uint8), config


def apply_blur(image, thresh: float = 0.5):
    use_blur = random.random() < thresh
    if use_blur:
        image = cv2.GaussianBlur(image, (3, 3), 0)

    return image, {"applied": use_blur}


def run(image_path, output_dir, config_output_dir):
    filename = os.path.basename(image_path)
    base, ext = os.path.splitext(filename)
    dir_base_name = os.path.basename(os.path.dirname(image_path))
    output_path = os.path.join(output_dir, dir_base_name, base + "_scan" + ext)
    config_output_path =  os.path.join(config_output_dir, dir_base_name, base + "_scan.json")

    if os.path.isfile(output_path):
        return
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    os.makedirs(os.path.dirname(config_output_path), exist_ok=True)

    img = cv2.imread(image_path)

    assert not img is None

    config_record = {
        "original_file": filename,
        "config": {}
    }

    bg_color = get_random_background_color()
    config_record["config"]["background_color"] = bg_color

    processed, config = apply_paper_texture(img, intensity=0.5)
    config_record["config"]["paper_texture"] = config

    processed, config = apply_rotation(processed, max_angle=1.5, bg_color=bg_color)
    config_record["config"]["rotation"] = config

    processed, config = apply_perspective_transform(processed, magnitude=10, bg_color=bg_color)
    config_record["config"]["perspective_transform"] = config

    processed, config = apply_lighting_gradient(processed)
    config_record["config"]["lighting_gradient"] = config

    processed, config = apply_blur(processed)
    config_record["config"]["blur"] = config

    cv2.imwrite(output_path, processed)

    with open(config_output_path, "w", encoding="utf-8") as f:
        json.dump(config_record, f, indent=4)


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
        default="data/JSSODa/export/images_scan/",
    )
    parser.add_argument(
        "--config_output_dir",
        type=str,
        default="data/JSSODa/export/metadata_scan/",
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

    os.makedirs(output_dir, exist_ok=True)

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