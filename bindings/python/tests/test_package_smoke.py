from __future__ import annotations

from pathlib import Path

import pyrwkv_tokenizer
from pyrwkv_tokenizer import WorldTokenizer


def test_public_api_and_packaged_vocab() -> None:
    assert pyrwkv_tokenizer.__version__ == "0.9.2"
    assert hasattr(WorldTokenizer, "from_buffer")

    vocab_path = Path(pyrwkv_tokenizer.__file__).with_name(
        "rwkv_vocab_v20230424.txt"
    )
    tokenizer = WorldTokenizer(str(vocab_path))
    text = "Today is a beautiful day. 今天是美好的一天。"
    assert tokenizer.decode(tokenizer.encode(text)) == text


def test_from_buffer_supports_explicit_ids_and_python_bytes_literals() -> None:
    vocab = b"""0 '<|rwkv_tokenizer_end_of_text|>' 30
1 'a' 1
2 b'\\n' 1
3 b'\\x5c' 1
4 b'\\101' 1
"""
    tokenizer = WorldTokenizer.from_buffer(vocab)
    text = "<|rwkv_tokenizer_end_of_text|>a\n\\A"
    assert tokenizer.encode(text) == [0, 1, 2, 3, 4]
    assert tokenizer.decode([0, 1, 2, 3, 4]) == text
