import os
import json
import argparse

from transformers import AutoTokenizer
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
        default="data/JSSODa/titles",
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
        text = ""
        for el in data["elements"]:
            if el["type"] == "text":
                text += el["content"]

        messages_list.append(
            [
                {
                    "role": "user",
                    "content": f"このテキストに関して、適切な日本語タイトルを出力してください。出力はタイトルのみとしてください。\n\n対象となるテキスト:\n```\n{text}\n```\n",
                }
            ]
        )
        src_list.append(f"{data['id'][:3]}/{data['id']}")

    checkpoint_path = "llm-jp/llm-jp-3.1-13b-instruct4"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    inputs = [
        tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            return_tensors="pt",
        )
        for messages in messages_list
    ]

    llm = LLM(
        model=checkpoint_path,
        max_model_len=4096,
        seed=0,
    )

    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.95,
        repetition_penalty=1.05,
        max_tokens=4000
    )

    outputs = llm.generate(inputs, sampling_params=sampling_params)
    for i, output in enumerate(outputs):
        generated_text = output.outputs[0].text

        output_file_path = os.path.join(
            output_path,
            os.path.splitext(src_list[i])[0] + "_title.txt"
        )
        os.makedirs(os.path.dirname(output_file_path), exist_ok=True)

        with open(output_file_path, "w", encoding="utf-8") as f:
            f.write(generated_text)


if __name__ == "__main__":
    main()