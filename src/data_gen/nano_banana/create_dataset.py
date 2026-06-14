import os

from tqdm import tqdm
from datasets import load_dataset, Dataset, DatasetDict, Features, Value, Image as ImageFeature


IMAGE_DIR = "data/nano_banana/images"
# エラーで生成できなかった画像ID
ERROR_IDS = ()


def process_data(example):
    image_path = example["image_path"]

    processed_example = {
        "id": example["id"],
        "image": image_path,
        "question": "この画像内のテキストを日本語の読み順に従って全て出力してください。出力は画像内のテキストのみとしてください。",
        "text": example["text"],
        "is_vertical": example["is_vertical"],
        "num_columns": example["num_columns"],
    }

    return processed_example


def main():
    html_dataset = load_dataset("llm-jp/Synth-JDoc", split="train")

    data_list = []
    error_ids = set(ERROR_IDS)
    for record in tqdm(html_dataset):
        record_id = record["id"]
        if record_id in error_ids:
            continue

        image_path = os.path.join(IMAGE_DIR, record_id[:3], f"nano_banana_{record_id}_000.jpeg")
        assert os.path.isfile(image_path)

        d = {
            "id": record_id,
            "image_path": image_path,
            "text": record["text"],
            "is_vertical": record["is_vertical"],
            "num_columns": record["num_columns"],
        }
        data_list.append(d)

    del html_dataset

    train_datasets = Dataset.from_list(data_list)

    dataset_dict = DatasetDict({
        "train": train_datasets,
    })

    print("\noriginal datasets:")
    print(dataset_dict)

    features = Features({
        'id': Value('string'),
        'image': ImageFeature(),
        'question': Value('string'),
        'text': Value('string'),
        'is_vertical': Value('bool'),
        'num_columns': Value('int32'),
    })

    processed_datasets = dataset_dict.map(
        process_data,
        remove_columns=dataset_dict["train"].column_names,
        features=features,
        num_proc=4,
    )

    print("\nProcessed datasets:")
    print(processed_datasets)
    print("\nExample from train split:")
    print(processed_datasets["train"][0])

    print("\nPushing dataset to the Hub")

    repo_id = "<user_id>/Nano-Banana-Doc-JA"
    processed_datasets.push_to_hub(
        repo_id,
        private=True,
        max_shard_size="1GB",
    )


if __name__ == "__main__":
    main()