import os
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
        "--dir_path",
        type=str,
        required=True,
    )
    args = parser.parse_args()
    dir_path = args.dir_path

    prompt_path_list = [os.path.join(dir_path, file) for file in sorted(os.listdir(dir_path))]

    pipe = ZImagePipeline.from_pretrained(
        "Tongyi-MAI/Z-Image-Turbo",
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=False,
    )
    pipe.to("cuda")
    pipe.transformer.set_attention_backend("flash")

    for prompt_path in tqdm(prompt_path_list):
        with open(prompt_path) as f:
            prompt = f.read()

        width, height = random.choice(RES)

        image = pipe(
            prompt=prompt,
            height=height,
            width=width,
            num_inference_steps=9,
            guidance_scale=0.0,
            generator=torch.Generator("cuda").manual_seed(42),
        ).images[0]

        output_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(prompt_path))),
            "images",
            os.path.basename(os.path.dirname(prompt_path)),
        )
        img_path = os.path.join(
            output_dir,
            os.path.splitext(os.path.basename(prompt_path))[0].removesuffix("_prompt") + ".jpg",
        )
        os.makedirs(os.path.dirname(img_path), exist_ok=True)
        image.save(img_path)


if __name__ == "__main__":
    main()