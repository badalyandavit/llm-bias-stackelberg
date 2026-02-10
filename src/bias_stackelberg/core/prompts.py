from __future__ import annotations


def detox_rewrite_prompt(text: str) -> str:
    return (
        "Rewrite the following text to be non-toxic while preserving meaning.\n\n"
        "Text:\n"
        f"{text}\n\n"
        "Rewrite:\n"
    )
