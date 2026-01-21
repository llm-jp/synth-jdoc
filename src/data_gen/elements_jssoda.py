import os
import json
import argparse

from datasets import load_dataset


def make_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split",
        type=str,
        default="train",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/JSSODa/",
    )
    return parser.parse_args()


def main():
    args = make_args()
    data_split = args.split
    output_dir = args.output_dir

    output_path = os.path.join(output_dir, f"jssoda_{data_split}_text_elements.jsonl")
    assert not os.path.isfile(output_path)

    ds = load_dataset("llm-jp/JSSODa", split=data_split)
    output_list = []
    for data in ds:
        text = data["text"]
        elements = []
        for paragraph in text.split("\n\n"):
            elements.append(
                {
                    "type": "text", 
                    "content": paragraph
                }
            )

        output_list.append({
            "id": data["id"],
            "title": "",
            "elements": elements,
            "is_vertical": data["is_vertical"],
            "column_count": data["num_columns"]
        })

    os.makedirs(output_dir, exist_ok=True)

    json_str = "\n".join([json.dumps(d, ensure_ascii=False, separators=(",", ":")) for d in output_list]) + "\n"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(json_str)


if __name__ == "__main__":
    main()
