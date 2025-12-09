import os
import json
import argparse

import torch
from transformers import AutoProcessor
from vllm import LLM, SamplingParams

os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'


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
        default="data/JSSODa/prompts",
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

    messages_list = []
    src_list = []
    for data in data_list[start_idx:end_idx]:
        for el in data["elements"]:
            if el["type"] == "text":
                continue

            paragraph = data["elements"][el["text_index"]]["content"]
            messages_list.append(
                [
                    {
                        "role": "user",
                        "content": f"この段落のテキストに関連した画像を、画像生成モデルによって生成したいです。その画像を描写する、適切なプロンプトを日本語の文章として出力してください。出力はプロンプトのみとしてください。\n\n対象となる段落:\n{paragraph}",
                    }
                ]
            )
            src_list.append(el["src"])

    checkpoint_path = "Qwen/Qwen3-30B-A3B-Instruct-2507"
    processor = AutoProcessor.from_pretrained(checkpoint_path)
    inputs = [
        processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        for messages in messages_list
    ]
    print(messages_list)
    print(inputs)

    llm = LLM(
        model=checkpoint_path,
        tensor_parallel_size=torch.cuda.device_count(),
        seed=0,
    )

    sampling_params = SamplingParams(
        temperature=0.7,
        max_tokens=1024,
        top_p=0.8,
        top_k=20,
        min_p=0,
        repetition_penalty=1.0,
        presence_penalty=1.0,
        stop_token_ids=[],
    )

    outputs = llm.generate(inputs, sampling_params=sampling_params)
    for i, output in enumerate(outputs):
        generated_text = output.outputs[0].text

        output_file_path = os.path.join(
            output_path,
            os.path.splitext(src_list[i])[0] + "_prompt.txt"
        )
        os.makedirs(os.path.dirname(output_file_path), exist_ok=True)

        with open(output_file_path, "w", encoding="utf-8") as f:
            f.write(generated_text)


if __name__ == "__main__":
    main()