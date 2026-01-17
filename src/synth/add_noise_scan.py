import os
import random
import argparse
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
    
    # 紙の色を修正
    paper_color_offset = np.array([[-5, -2, 0]])

    base = image.astype(np.int16) + paper_color_offset
    base = np.clip(base, 0, 255).astype(np.uint8)

    # ガウシアンノイズ
    noise = np.random.normal(0, 10, (h, w, c)).astype(np.float32)

    noisy_image = base.astype(np.float32) + noise * intensity

    return np.clip(noisy_image, 0, 255).astype(np.uint8)


def apply_rotation(image, max_angle=1.5, bg_color=(255, 255, 255)):
    h, w = image.shape[:2]
    angle = random.uniform(-max_angle, max_angle)

    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)

    rotated = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=bg_color)

    return rotated


def apply_perspective_transform(image, magnitude=10, bg_color=(255, 255, 255)):
    h, w = image.shape[:2]
    
    # 元の4点
    src_pts = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
    
    # 移動後の4点（ランダムに少しずらす）
    dst_pts = np.float32([
        [random.randint(0, magnitude), random.randint(0, magnitude)],
        [w - random.randint(0, magnitude), random.randint(0, magnitude)],
        [random.randint(0, magnitude), h - random.randint(0, magnitude)],
        [w - random.randint(0, magnitude), h - random.randint(0, magnitude)]
    ])
    
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped = cv2.warpPerspective(image, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=bg_color)

    return warped


def apply_lighting_gradient(image, h_thresh: float = 0.7, v_thresh: float = 0.7, vignette_thresh: float = 0.4):
    """
    画像に影を加える。
    縦、横、ビネット効果（スポットライト）をランダムに組み合わせる。
    """
    h, w = image.shape[:2]

    mask = np.ones((h, w), dtype=np.float32)

    # 横方向
    if random.random() < h_thresh:
        limit = random.uniform(0.7, 0.9)
        gradient = np.linspace(limit, 1.0, w)
        # 左右反転
        if random.random() < 0.5:
            gradient = gradient[::-1]

        mask *= np.tile(gradient, (h, 1))

    # 縦方向
    if random.random() < v_thresh:
        limit = random.uniform(0.7, 0.9)
        gradient = np.linspace(limit, 1.0, h).reshape(-1, 1)
        # 上下反転
        if random.random() < 0.5:
            gradient = gradient[::-1]

        mask *= np.tile(gradient, (1, w))

    # ビネット
    if random.random() < vignette_thresh:
        # 中心座標
        center_x = random.randint(w // 3, 2 * w // 3)
        center_y = random.randint(h // 3, 2 * h // 3)

        Y, X = np.ogrid[:h, :w]

        # 中心からの二乗距離
        dist_sq = (X - center_x)**2 + (Y - center_y)**2
        max_dist_sq = (w**2 + h**2) / 2

        # 距離に応じて暗くする
        vignette = 1 - (dist_sq / max_dist_sq) * random.uniform(0.2, 0.5)
        vignette = np.clip(vignette, 0, 1.0)

        mask *= vignette

    mask = np.dstack([mask] * 3)

    shaded = image.astype(np.float32) * mask

    return np.clip(shaded, 0, 255).astype(np.uint8)


def apply_blur(image, thresh: float = 0.5):
    if random.random() < thresh:
        image = cv2.GaussianBlur(image, (3, 3), 0)

    return image


def run(image_path, output_dir):
    filename = os.path.basename(image_path)
    base, ext = os.path.splitext(filename)
    dir_base_name = os.path.basename(os.path.dirname(image_path))
    output_path = os.path.join(output_dir, dir_base_name, base + "_scan" + ext)

    if os.path.isfile(output_path):
        return
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    img = cv2.imread(image_path)

    assert not img is None

    bg_color = get_random_background_color()

    processed = apply_paper_texture(img, intensity=0.5)

    processed = apply_rotation(processed, max_angle=1.5, bg_color=bg_color)

    processed = apply_perspective_transform(processed, magnitude=10, bg_color=bg_color)

    processed = apply_lighting_gradient(processed)

    processed = apply_blur(processed)

    cv2.imwrite(output_path, processed)


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
        "--num_processes",
        type=int,
        default=4,
    )
    return parser.parse_args()


def main():
    args = make_args()
    input_dir = args.input_dir
    output_dir = args.output_dir
    num_processes = args.num_processes

    os.makedirs(output_dir, exist_ok=True)

    image_paths = sorted(glob(os.path.join(input_dir, "**", "*.png"), recursive=True))[:50]

    length = len(image_paths)
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        tqdm(
            executor.map(run, image_paths, [output_dir for _ in range(length)]),
            total=length,
            desc="Adding noise",
        )


if __name__ == "__main__":
    main()