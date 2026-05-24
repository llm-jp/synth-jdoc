import argparse
import os
import json

import torch
from tqdm import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor, set_seed


MODEL_PATH = "./checkpoints/InternVL3-8B-hf-Nano-Banana-train"
MAX_NEW_TOKENS = 3072
USER_PROMPT = "この画像内のテキストを日本語の読み順に従って全て出力してください。出力は画像内のテキストのみとしてください。"
SEED = 42
set_seed(SEED)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image_directory_path",
        type=str,
        default="./data/VJRODa/images",
        help="Path to test image directory",
    )
    parser.add_argument(
        "--output_directory_path",
        type=str,
        default="./output_vjroda",
        help="Path to output directory",
    )

    args = parser.parse_args()
    image_directory_path = args.image_directory_path
    output_directory_path = args.output_directory_path

    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="eager",
    ).eval()


    processor = AutoProcessor.from_pretrained(
        MODEL_PATH,
        use_fast=False,
    )

    image_file_list = os.listdir(image_directory_path)
    
    output_list = []
    for image_file in tqdm(image_file_list):
        image_path = os.path.join(image_directory_path, image_file)

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image_path,
                    },
                    {"type": "text", "text": USER_PROMPT},
                ],
            }
        ]

        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            padding=False,
            truncation=False,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device, dtype=torch.bfloat16)

        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                num_beams=1,
            )

        decoded_output = processor.decode(
            generated_ids[0, inputs["input_ids"].shape[1] :],
            skip_special_tokens=True
        )
        
        assert isinstance(decoded_output, str)

        output_dict = {"id": image_file.rstrip(".jpg"), "pred": decoded_output}
        output_list.append(output_dict)
    
    output_file_path = os.path.join(
        output_directory_path,
        os.path.basename(MODEL_PATH.removesuffix("/")),
        f"pred_rw_vert_{os.path.basename(MODEL_PATH.removesuffix("/"))}.jsonl",
    )

    json_str = "\n".join([json.dumps(d, ensure_ascii=False, separators=(",", ":")) for d in output_list]) + "\n"
    dir_name = os.path.dirname(output_file_path)
    if not os.path.isdir(dir_name):
        os.makedirs(dir_name, exist_ok=True)
    with open(output_file_path, "w") as fout:
        fout.write(json_str)


if __name__ == "__main__":
    main()