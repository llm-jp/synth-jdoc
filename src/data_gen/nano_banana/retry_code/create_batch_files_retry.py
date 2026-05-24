import os
import json
import glob

from tqdm import tqdm


REQUEST_DIR = "data/nano_banana/requests"
IMAGE_DIR = "data/nano_banana/images"
OUTPUT_DIR = "data/nano_banana/retry/requests"
FILE_IDX = 1


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    image_ids = set([os.path.basename(file).split("_")[2] for file in glob.glob(os.path.join(IMAGE_DIR, "*", "*.*"))])

    file_list = [os.path.join(REQUEST_DIR, file) for file in sorted(os.listdir(REQUEST_DIR))]
    output_list = []
    for file in tqdm(file_list):
        with open(file, 'r', encoding='utf-8') as f:
            data_list = [json.loads(line) for line in f]

        for data in data_list:
            data_key = data["key"]
            if not data_key in image_ids:
                output_list.append(data)

    json_str = "\n".join([json.dumps(d, ensure_ascii=False, separators=(",", ":")) for d in output_list]) + "\n"
    output_file_path = os.path.join(OUTPUT_DIR, f"batch_image_requests_retry_{FILE_IDX:05d}.jsonl")

    assert not os.path.isfile(output_file_path)

    with open(output_file_path, "w", encoding="utf-8") as fout:
        fout.write(json_str)


if __name__ == "__main__":
    main()
