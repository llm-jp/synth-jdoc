import argparse
import os
import json

import torch
from tqdm import tqdm
from transformers import AutoProcessor, Gemma3ForConditionalGeneration, set_seed


MODEL_PATH = "./checkpoints/gemma-3-12b-it-HTML-Synth-train"
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

    model = Gemma3ForConditionalGeneration.from_pretrained(
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
                ]
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

        input_len = inputs["input_ids"].shape[-1]

        with torch.no_grad():
            generation = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                num_beams=1,
            )
            generation = generation[0][input_len:]
        
        decoded = processor.decode(generation, skip_special_tokens=True)
    
        assert isinstance(decoded, str)

        output_dict = {"id": image_file.rstrip(".jpg"), "pred": decoded}
        output_list.append(output_dict)
    
    output_file_path = os.path.join(
        output_directory_path,
        os.path.basename(MODEL_PATH),
        f"pred_rw_vert_{os.path.basename(MODEL_PATH)}.jsonl",
    )

    json_str = "\n".join([json.dumps(d, ensure_ascii=False, separators=(",", ":")) for d in output_list]) + "\n"
    dir_name = os.path.dirname(output_file_path)
    if not os.path.isdir(dir_name):
        os.makedirs(dir_name, exist_ok=True)
    with open(output_file_path, "w") as fout:
        fout.write(json_str)


if __name__ == "__main__":
    main()