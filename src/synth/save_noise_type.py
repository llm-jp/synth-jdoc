from datasets import load_dataset


NUM_PROC = 4


def main():
    dataset = load_dataset("ssgw-keito/HTML-Synth-OCR-JA", split="train", num_proc=NUM_PROC)

    noise_type_1 = []
    noise_type_2 = []
    for data in dataset:
        if data["noise_type"] == 1:
            noise_type_1.append(data["id"])
        if data["noise_type"] == 2:
            noise_type_2.append(data["id"])

    with open("data/JSSODa/export/ablation/noise_type_1.txt", "w") as f:
        f.write("\n".join(noise_type_1))
    with open("data/JSSODa/export/ablation/noise_type_2.txt", "w") as f:
        f.write("\n".join(noise_type_2))


if __name__ == "__main__":
    main()
