import os
import random
import json

import numpy as np
from scipy.optimize import linear_sum_assignment


def match_images_to_texts(data, seed: int = None) -> tuple[dict[str, int | float], float]:
    text_indices = [i for i, item in enumerate(data) if item.get("type") == "text"]
    image_indices = [i for i, item in enumerate(data) if item.get("type") == "image"]

    assert not len(image_indices) > len(text_indices)

    # コスト行列を作成する
    # 行がimage 列がtext 値はそのペアの距離の2乗
    base_cost_matrix = (np.array(image_indices)[:, None] - np.array(text_indices)) ** 2

    # 距離が同じ時、ランダムに選択するためにノイズを加える
    rng = np.random.default_rng(seed)
    epsilon = 1e-6 
    noise = rng.random(base_cost_matrix.shape) * epsilon
    cost_matrix = base_cost_matrix + noise

    # 線形和割当問題を解く
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    matches = []
    total_distance = 0
    
    for r, c in zip(row_ind, col_ind):
        img_idx = image_indices[r]
        txt_idx = text_indices[c]
        distance = cost_matrix[r, c]
        
        matches.append({
            "image_index": img_idx,
            "text_index": txt_idx,
            "distance": distance
        })
        total_distance += distance

    return matches, total_distance


def main():
    data_list = []
    with open("data/JSSODa/jssoda_train_text_elements.jsonl") as f:
        for line in f:
            data_list.append(json.loads(line))

    for data_idx, data in enumerate(data_list):
        num_paragraph = len(data["elements"])
        num_img = random.randint(0, num_paragraph//2)

        img_idx_list = sorted([random.randint(0, num_paragraph) for _ in range(num_img)])

        for idx in reversed(img_idx_list):
            data["elements"].insert(idx, {"type": "image", "src": None, "caption": "", "span_all": None})
        
        img_count = 0
        for el in data["elements"]:
            if el["type"] == "image":
                el["src"] = f"{data['id'][:3]}/{data['id']}_{img_count:03d}.jpg"
                img_count += 1

        pairs, _ = match_images_to_texts(data["elements"], data_idx)
        for p in pairs:
            data["elements"][p["image_index"]]["text_index"] = p["text_index"]


    output_dir = "data/JSSODa/"
    os.makedirs(output_dir, exist_ok=True)

    json_str = "\n".join([json.dumps(d, ensure_ascii=False, separators=(",", ":")) for d in data_list]) + "\n"
    with open(os.path.join(output_dir, "jssoda_train_img_match.jsonl"), "w", encoding="utf-8") as f:
        f.write(json_str)


if __name__ == "__main__":
    main()