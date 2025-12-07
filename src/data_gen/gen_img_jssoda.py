import os
import json
import argparse
import random

import torch
from tqdm import tqdm
from diffusers import ZImagePipeline

# from official demo
RES = (
    (1024, 1024),
    (1152, 896),
    (896, 1152),
    (1152, 864),
    (864, 1152),
    (1248, 832),
    (832, 1248),
    (1280, 720),
    (720, 1280),
    (1344, 576),
    (576, 1344),
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--file_path",
        type=str,
        default="data/JSSODa/jssoda_train_img_match.jsonl",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="data/JSSODa/images",
    )
    parser.add_argument(
        "--start_idx",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--end_idx",
        type=int,
        default=1,
    )
    args = parser.parse_args()
    file_path = args.file_path
    output_path = args.output_path

    data_list = []
    with open(file_path) as f:
        for line in f:
            data_list.append(json.loads(line))

    start_idx = args.start_idx
    end_idx = min(args.end_idx, len(data_list))

    pipe = ZImagePipeline.from_pretrained(
        "Tongyi-MAI/Z-Image-Turbo",
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=False,
    )
    pipe.to("cuda")

    total_count = sum(
        1
        for data in data_list[start_idx:end_idx]
        for el in data["elements"]
        if el["type"] == "image"
    )
    with tqdm(total=total_count, desc="Generating images:") as pbar:
        for data in data_list[start_idx:end_idx]:
            for el in data["elements"]:
                if el["type"] == "text":
                    continue

                prompt = data["elements"][el["text_index"]]["content"]
                width, height = random.choice(RES)

                image = pipe(
                    prompt=prompt,
                    height=height,
                    width=width,
                    num_inference_steps=9,
                    guidance_scale=0.0,
                    generator=torch.Generator("cuda").manual_seed(42),
                ).images[0]

                img_path = os.path.join(output_path, el["src"])
                os.makedirs(os.path.dirname(img_path), exist_ok=True)
                image.save(img_path)
                pbar.update(1)


if __name__ == "__main__":
    main()