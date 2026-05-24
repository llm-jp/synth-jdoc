import os

from tqdm import tqdm
from google import genai
from google.genai import types


INPUT_DIR = "data/nano_banana/requests"
OUTPUT_DIR = "data/nano_banana/job_name"


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    client = genai.Client()

    file_list = [os.path.join(INPUT_DIR, file) for file in sorted(os.listdir(INPUT_DIR))]
    for file in tqdm(file_list):
        base_name = os.path.basename(file)

        output_file_path = os.path.join(
            OUTPUT_DIR,
            os.path.splitext(base_name)[0].replace("requests", "job_name") + ".txt"
        )
        assert not os.path.isfile(output_file_path)

        # Upload batch job
        uploaded_file = client.files.upload(
            file=file,
            config=types.UploadFileConfig(display_name=base_name, mime_type='jsonl')
        )
        print(f"Uploaded file: {uploaded_file.name}")

        # Create batch job
        file_batch_job = client.batches.create(
            model="gemini-3-pro-image-preview",
            src=uploaded_file.name,
            config={
                'display_name': base_name,
            },
        )
        print(f"Created batch job: {file_batch_job.name}")

        with open(output_file_path, "w", encoding="utf-8") as fout:
            fout.write(file_batch_job.name)


if __name__ == "__main__":
    main()
