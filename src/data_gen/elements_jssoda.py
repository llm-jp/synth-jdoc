import os
import json

from datasets import load_dataset


def main():
    ds = load_dataset("llm-jp/JSSODa", split="train")
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

    output_dir = "data/JSSODa/"
    os.makedirs(output_dir, exist_ok=True)

    json_str = "\n".join([json.dumps(d, ensure_ascii=False, separators=(",", ":")) for d in output_list]) + "\n"
    with open(os.path.join(output_dir, "jssoda_train_text_elements.jsonl"), "w", encoding="utf-8") as f:
        f.write(json_str)


if __name__ == "__main__":
    main()
