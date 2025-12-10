import os
import argparse
import random

import torch
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor
from vllm import LLM, SamplingParams


os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
PROMPT_SHORT = "この画像を図として挿入します。図のタイトルとして適切で簡潔な日本語キャプションを生成してください。ただし、固有名詞は使わないでください。"
PROMPT_LONG = "この画像を図として挿入します。図のタイトルとして適切な、複数文にわたる日本語キャプションを生成してください。ただし、固有名詞は使わないでください。"


def prepare_inputs_for_vllm(messages, processor):
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs, video_kwargs = process_vision_info(
        messages,
        image_patch_size=processor.image_processor.patch_size,
        return_video_kwargs=True,
        return_video_metadata=True
    )

    mm_data = {}
    if image_inputs is not None:
        mm_data['image'] = image_inputs
    if video_inputs is not None:
        mm_data['video'] = video_inputs

    return {
        'prompt': text,
        'multi_modal_data': mm_data,
        'mm_processor_kwargs': video_kwargs
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dir_path",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
    )
    args = parser.parse_args()
    dir_path = args.dir_path

    random.seed(args.seed)

    image_path_list = [os.path.join(dir_path, file) for file in sorted(os.listdir(dir_path))]

    messages_list = [
        [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image_path,
                    },
                    {"type": "text", "text": PROMPT_SHORT if random.random() < 0.9 else PROMPT_LONG},
                ],
            },
        ] for image_path in image_path_list
    ]


    checkpoint_path = "Qwen/Qwen3-VL-30B-A3B-Instruct"
    processor = AutoProcessor.from_pretrained(checkpoint_path)
    inputs = [prepare_inputs_for_vllm(message, processor) for message in messages_list]

    llm = LLM(
        model=checkpoint_path,
        mm_encoder_tp_mode="data",
        tensor_parallel_size=torch.cuda.device_count(),
        limit_mm_per_prompt={
            "image": 1,
            "video": 0,
            "audio": 0,
        },
        seed=0,
    )

    sampling_params = SamplingParams(
        temperature=0.7,
        max_tokens=1024,
        top_p=0.8,
        top_k=20,
        repetition_penalty=1.0,
        presence_penalty=1.5,
        stop_token_ids=[],
    )

    outputs = llm.generate(inputs, sampling_params=sampling_params)
    for i, output in enumerate(outputs):
        generated_text = output.outputs[0].text

        output_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(image_path_list[i]))),
            "captions",
            os.path.basename(os.path.dirname(image_path_list[i])),
        )
        os.makedirs(output_dir, exist_ok=True)
        output_file_path = os.path.join(
            output_dir,
            os.path.splitext(os.path.basename(image_path_list[i]))[0] + "_cap.txt",
        )

        with open(output_file_path, "w", encoding="utf-8") as f:
            f.write(generated_text)


if __name__ == "__main__":
    main()