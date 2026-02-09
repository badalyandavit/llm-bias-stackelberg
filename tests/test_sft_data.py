import torch

from bias_stackelberg.train.sft_data import (
    SftCausalLMDataset,
    SftDatasetConfig,
    format_prompt,
    pad_batch,
)


class StubTok:
    def __call__(self, text, add_special_tokens=True, truncation=True, max_length=512):
        words = text.split()
        ids = list(range(1, min(len(words), max_length) + 1))
        return {"input_ids": ids}


def test_dataset_masks_prompt_tokens() -> None:
    recs = [{"id": "a", "prompt": "hello world", "completion": "ok"}]
    ds = SftCausalLMDataset(recs, StubTok(), SftDatasetConfig(max_length=64))
    item = ds[0]
    labels = item["labels"].tolist()

    prompt_ids = StubTok()(format_prompt("hello world"))["input_ids"]
    cut = min(len(prompt_ids), len(labels))

    assert all(x == -100 for x in labels[:cut])
    assert any(x != -100 for x in labels[cut:])


def test_pad_batch_shapes() -> None:
    b1 = {
        "input_ids": torch.tensor([1, 2]),
        "labels": torch.tensor([-100, 2]),
        "attention_mask": torch.tensor([1, 1]),
    }
    b2 = {
        "input_ids": torch.tensor([1]),
        "labels": torch.tensor([-100]),
        "attention_mask": torch.tensor([1]),
    }

    out = pad_batch([b1, b2], pad_id=0)
    assert out["input_ids"].shape == (2, 2)
    assert out["labels"].shape == (2, 2)
    assert out["attention_mask"].shape == (2, 2)
