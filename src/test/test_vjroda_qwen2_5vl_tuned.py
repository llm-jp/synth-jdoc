import argparse
import os
import json

import torch
from tqdm import tqdm
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, set_seed
from qwen_vl_utils import process_vision_info


MODEL_PATH = "./checkpoints/Qwen2.5-VL-7B-Instruct-HTML-Synth-train"
MIN_PIXELS = 1*28*28
MAX_PIXELS = 1280*28*28
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

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="eager",
    )

    processor = AutoProcessor.from_pretrained(
        MODEL_PATH,
        min_pixels=MIN_PIXELS,
        max_pixels=MAX_PIXELS,
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

        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, _ = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            padding=False,
            truncation=False,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")

        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                num_beams=1,
            )
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        
        assert len(output_text) == 1

        output_dict = {"id": image_file.rstrip(".jpg"), "pred": output_text[0]}
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