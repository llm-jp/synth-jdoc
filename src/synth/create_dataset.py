import os
import glob
import json
import random
from concurrent.futures import ProcessPoolExecutor

from datasets import Dataset, DatasetDict, Features, Value, Image as ImageFeature


TRAIN_JSON_DIR = "data/JSSODa/export/metadata"
TRAIN_IMAGE_DIR = "data/JSSODa/export/images"
TRAIN_NOISE_IMAGE_DIR_1 = "data/JSSODa/export/images_scan"
TRAIN_NOISE_IMAGE_DIR_2 = "data/JSSODa/export/images_augraphy"
VAL_JSON_DIR = "data/JSSODa_validation/export/metadata"
VAL_IMAGE_DIR = "data/JSSODa_validation/export/images"
VAL_NOISE_IMAGE_DIR_1 = "data/JSSODa_validation/export/images_scan"
VAL_NOISE_IMAGE_DIR_2 = "data/JSSODa_validation/export/images_augraphy"
TEST_JSON_DIR = "data/JSSODa_test/export/metadata"
TEST_IMAGE_DIR = "data/JSSODa_test/export/images"
TEST_NOISE_IMAGE_DIR_1 = "data/JSSODa_test/export/images_scan"
TEST_NOISE_IMAGE_DIR_2 = "data/JSSODa_test/export/images_augraphy"


def process_data(example):
    image_path = example["image_path"]

    processed_example = {
        "id": example["id"],
        "image": image_path,
        "question": "この画像内のテキストを日本語の読み順に従って全て出力してください。出力は画像内のテキストのみとしてください。",
        "text": example["text"],
        "is_vertical": example["is_vertical"],
        "num_columns": example["num_columns"],
        "noise_type": example["noise_type"],
    }

    return processed_example


def load_record(file_path):
    with open(file_path) as f:
        record = json.load(f)
    
    text = record["title"] if record["style"]["show_title"] else ""
    for el in record["elements"]:
        if el["type"] == "text":
            text += "\n\n" + el["content"]
        elif el["type"] == "image":
            text += "\n\n" + el["caption"]

    d = {
        "id": record["id"],
        "text": text.strip(),
        "is_vertical": record["style"]["is_vertical"],
        "num_columns": record["style"]["column_count"],
    }

    return d


def load_data(json_dir_path):
    file_path_list = sorted(glob.glob(os.path.join(json_dir_path, "*/*.json"), recursive=True))

    data_list = []
    with ProcessPoolExecutor(max_workers=4) as executor:
        data_list = list(executor.map(load_record, file_path_list))
    
    return data_list


def select_noise_images(data_list):
    n = len(data_list)
    n1 = int(n * 0.3)
    n2 = int(n * 0.3)
    n0 = n - n1 - n2

    type_list = [0] * n0 + [1] * n1 + [2] * n2

    random.shuffle(type_list)

    for i, data in enumerate(data_list):
        data["noise_type"] = type_list[i]

    return data_list


def main():
    # train
    data_list = load_data(TRAIN_JSON_DIR)
    data_list = select_noise_images(data_list)
    for data in data_list:
        if data["noise_type"] == 0:
            image_dir = TRAIN_IMAGE_DIR
            file_name = data["id"] + ".png"
        elif data["noise_type"] == 1:
            image_dir = TRAIN_NOISE_IMAGE_DIR_1
            file_name = data["id"] + "_scan.png"
        elif data["noise_type"] == 2:
            image_dir = TRAIN_NOISE_IMAGE_DIR_2
            file_name = data["id"] + "_augraphy.png"
        
        data["image_path"] = os.path.join(
            image_dir,
            data["id"][:3],
            file_name,
        )
    train_datasets = Dataset.from_list(data_list)

    # validation
    data_list = load_data(VAL_JSON_DIR)
    data_list = select_noise_images(data_list)
    for data in data_list:
        if data["noise_type"] == 0:
            image_dir = VAL_IMAGE_DIR
            file_name = data["id"] + ".png"
        elif data["noise_type"] == 1:
            image_dir = VAL_NOISE_IMAGE_DIR_1
            file_name = data["id"] + "_scan.png"
        elif data["noise_type"] == 2:
            image_dir = VAL_NOISE_IMAGE_DIR_2
            file_name = data["id"] + "_augraphy.png"
        
        data["image_path"] = os.path.join(
            image_dir,
            data["id"][:3],
            file_name,
        )
    validation_datasets = Dataset.from_list(data_list)

    # test
    data_list = load_data(TEST_JSON_DIR)
    data_list = select_noise_images(data_list)
    for data in data_list:
        if data["noise_type"] == 0:
            image_dir = TEST_IMAGE_DIR
            file_name = data["id"] + ".png"
        elif data["noise_type"] == 1:
            image_dir = TEST_NOISE_IMAGE_DIR_1
            file_name = data["id"] + "_scan.png"
        elif data["noise_type"] == 2:
            image_dir = TEST_NOISE_IMAGE_DIR_2
            file_name = data["id"] + "_augraphy.png"
        
        data["image_path"] = os.path.join(
            image_dir,
            data["id"][:3],
            file_name,
        )
    test_datasets = Dataset.from_list(data_list)

    # DatasetDict
    dataset_dict = DatasetDict({
        "train": train_datasets,
        "validation": validation_datasets,
        "test": test_datasets
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
        'noise_type': Value('int32'),
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

    repo_id = "<user_id>/Synth-JDoc"
    processed_datasets.push_to_hub(
        repo_id,
        private=True,
        max_shard_size="1GB",
    )


if __name__ == "__main__":
    main()