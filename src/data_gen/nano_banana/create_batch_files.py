import os
import json

from tqdm import tqdm
from datasets import load_dataset


OUTPUT_DIR = "data/nano_banana/requests"
BATCH_SIZE = 500


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    dataset = load_dataset("llm-jp/Synth-JDoc", split="train")

    for start_idx in tqdm(range(0, len(dataset), BATCH_SIZE)):
        request_list = []

        for idx in range(start_idx, min(start_idx+BATCH_SIZE, len(dataset))):
            data = dataset[idx]
            prompt = f"""以下のテキストが書かれた日本語の文書画像を生成してください。
現代の社会に存在しそうな文書画像にしてください。
文書は{'縦書き' if data['is_vertical'] else '横書き'}で{data['num_columns']}段組みのものにしてください。
また、図のキャプションが含まれている場合はそれに対応する図を文書に挿入してください。

文書に含むべきテキスト:
```
{data['text']}
```
"""
            request_list.append({"key": data["id"], "request": {"contents": [{"parts": [{"text": prompt}]}], "generation_config": {"responseModalities": ["TEXT", "IMAGE"]}}})

        json_str = "\n".join([json.dumps(d, ensure_ascii=False, separators=(",", ":")) for d in request_list]) + "\n"
        output_file_path = os.path.join(OUTPUT_DIR, f"batch_image_requests_{start_idx // BATCH_SIZE:05d}.jsonl")

        assert not os.path.isfile(output_file_path)

        with open(output_file_path, "w", encoding="utf-8") as fout:
            fout.write(json_str)


if __name__ == "__main__":
    main()
