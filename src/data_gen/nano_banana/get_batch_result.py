import os
import io
import json
import base64

from tqdm import tqdm
from google import genai
from google.genai import types
from PIL import Image

INPUT_DIR = "data/nano_banana/job_name"
# INPUT_DIR = "data/nano_banana/retry/job_name"
OUTPUT_DIR = "data/nano_banana/images"
OUTPUT_DIR_TEXT = "data/nano_banana/output_text"


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    client = genai.Client()

    completed_states = set([
        'JOB_STATE_SUCCEEDED',
        'JOB_STATE_FAILED',
        'JOB_STATE_CANCELLED',
        'JOB_STATE_EXPIRED',
    ])

    file_list = [os.path.join(INPUT_DIR, file) for file in sorted(os.listdir(INPUT_DIR)) if file.endswith(".txt")]
    for file in tqdm(file_list):
        with open(file) as f:
            job_id = f.read().strip()

        batch_job = client.batches.get(name=job_id)

        if batch_job.state.name not in completed_states:
            print(f"Not completed: {file}")
            continue

        # Retrieve results
        if batch_job.state.name == 'JOB_STATE_SUCCEEDED':
            result_file_name = batch_job.dest.file_name
            file_content_bytes = client.files.download(file=result_file_name)
            file_content = file_content_bytes.decode('utf-8')

            for line in tqdm(file_content.splitlines()):
                if line:
                    parsed_response = json.loads(line)
                    request_key = parsed_response["key"]
                    if 'response' in parsed_response and parsed_response['response'] and 'parts' in parsed_response['response']['candidates'][0]['content']:
                        for i, part in enumerate(parsed_response['response']['candidates'][0]['content']['parts']):
                            if part.get('text'):
                                file_path = os.path.join(OUTPUT_DIR_TEXT, request_key[:3], f"nano_banana_{request_key}_{i:03d}.txt")
                                os.makedirs(os.path.dirname(file_path), exist_ok=True)
                                with open(file_path, "w") as f:
                                    f.write(part['text'])
                            elif part.get('inlineData'):
                                mime_type = part['inlineData']['mimeType']
                                img_data = base64.b64decode(part['inlineData']['data'])
                                image = Image.open(io.BytesIO(img_data))
                                extension = mime_type.split('/')[-1] # image/png -> png
                                file_path = os.path.join(OUTPUT_DIR, request_key[:3], f"nano_banana_{request_key}_{i:03d}.{extension}")
                                os.makedirs(os.path.dirname(file_path), exist_ok=True)
                                image.save(file_path)
                    elif 'error' in parsed_response:
                        print(f"Error: {parsed_response['error']}")
                    else:
                        print(f"Unknown Error")
        elif batch_job.state.name == 'JOB_STATE_FAILED':
            print(f"Error: {batch_job.error}, {file}")


if __name__ == "__main__":
    main()
