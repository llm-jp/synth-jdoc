import os
from dataclasses import dataclass

import torch
from transformers import AutoProcessor, AutoModelForImageTextToText, HfArgumentParser, set_seed
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer


@dataclass
class ModelArguments:
    model_name: str
    revision: str


def find_subsequence_positions(input_ids: torch.Tensor, target_sequence: torch.Tensor) -> torch.Tensor:
    """
    input_idsにtarget_sequenceが含まれているか判定し、
    その開始位置を返す。

    Args:
        input_ids (torch.Tensor): 検索対象のテンソル (batch_size, sequence_length)
        target_sequence (torch.Tensor): 検索したいトークンID列 (target_length)

    Returns:
        torch.Tensor: 見つかった位置を示すテンソル。
                      各行は [バッチ内のインデックス, 開始位置のインデックス] を示す。
    """
    target_len = len(target_sequence)
    # シーケンス長がターゲットより短い場合は空のテンソルを返す
    if input_ids.shape[1] < target_len:
        return torch.tensor([], dtype=torch.long)
        
    unfolded_input = input_ids.unfold(dimension=1, size=target_len, step=1)
    matches = (unfolded_input == target_sequence.view(1, 1, -1)).all(dim=2)
    positions = matches.nonzero(as_tuple=False)
    return positions


def main():
    parser = HfArgumentParser((SFTConfig, ModelArguments))
    training_args, model_args = parser.parse_args_into_dataclasses()

    training_args.model_init_kwargs = {"torch_dtype": torch.bfloat16, "attn_implementation": "eager", "revision": model_args.revision}
    training_args.gradient_checkpointing_kwargs = {"use_reentrant": False}
    training_args.dataset_kwargs = {"skip_prepare_dataset": True}
    training_args.remove_unused_columns = False

    set_seed(training_args.seed)

    dir_name = os.path.basename(training_args.output_dir)
    if "WANDB_PROJECT" not in os.environ:
        os.environ["WANDB_PROJECT"] = "HTML_Synth"
    if "WANDB_NAME" not in os.environ:
        os.environ["WANDB_NAME"] = dir_name

    model_name = model_args.model_name

    processor = AutoProcessor.from_pretrained(
        model_name,
        revision=model_args.revision,
        use_fast=False,
    )
    processor.tokenizer.padding_side = "right"


    model = AutoModelForImageTextToText.from_pretrained(model_name, **training_args.model_init_kwargs)


    def collate_fn(examples):
        texts = []
        for example in examples:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": example["image"],
                        },
                        {"type": "text", "text": example["question"]},
                    ]
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": example["text"]},
                    ]
                },
            ]
            texts.append(processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False).strip())

        images = [
            [example["image"].convert("RGB")]
            for example in examples
        ]

        batch = processor(
            images=images, text=texts, return_tensors="pt", padding=True, padding_side="right"
        )

        labels = batch["input_ids"].clone()
        image_token_ids = processor.tokenizer.convert_tokens_to_ids(["<img>", "</img>", "<IMG_CONTEXT>"])
        labels[labels == processor.tokenizer.pad_token_id] = -100
        for image_token_id in image_token_ids:
            labels[labels == image_token_id] = -100

        # ユーザープロンプト部分をマスク (single turn用)
        target_str = "<|im_end|>\n<|im_start|>assistant\n"
        target_seq = torch.tensor(processor.tokenizer.encode(target_str, add_special_tokens=False))
        found_positions = find_subsequence_positions(labels, target_seq)
        for batch_idx in range(labels.shape[0]):
            positions_in_batch = found_positions[found_positions[:, 0] == batch_idx]

            assert positions_in_batch.numel() > 0
            if positions_in_batch.numel() > 0:
                # 複数見つかった場合も考慮し、最も早い開始位置を取得
                start_index = positions_in_batch[:, 1].min()
                labels[batch_idx, :start_index+len(target_seq)] = -100

        batch["labels"] = labels
        return batch


    dataset = load_dataset("llm-jp/Synth-JDoc", split="train")

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        data_collator=collate_fn,
        train_dataset=dataset,
        processing_class=processor,
    )

    trainer.train()

    trainer.save_model()


if __name__ == "__main__":
    main()
