# RWKV tokenizer for RWKV7 deployments

`pyrwkv-tokenizer-rwkv7` is the deployable Python wheel for the
[`shiroko98/rwkv-tokenizer`](https://github.com/shiroko98/rwkv-tokenizer)
fork. It keeps the Python import name stable:

```python
from pyrwkv_tokenizer import WorldTokenizer
```

The distribution name intentionally differs from the upstream
`pyrwkv-tokenizer` package, so an ordinary `pip install -U` cannot replace
this fork with the older upstream wheel.

## Fork features

In addition to the upstream Rust tokenizer, this package provides:

- `WorldTokenizer.from_buffer(vocab_bytes)` in the Python binding;
- explicit and sparse RWKV token-ID handling, including token ID `0`;
- parsing of Python bytes-literal vocabulary tokens such as `b'\\x00'`;
- the `encode`, `encode_batch`, `decode`, `vocab_size`, and `get_vocab` APIs.

## Installation

Install the wheel that matches the server's CPython ABI and CPU architecture:

```bash
python -m pip install pyrwkv-tokenizer-rwkv7==0.9.2
```

The runtime import remains `pyrwkv_tokenizer`; do **not** import the
PyPI distribution name.

## Usage

```python
from pathlib import Path
from pyrwkv_tokenizer import WorldTokenizer

# Load a vocabulary file.
tokenizer = WorldTokenizer("/path/to/rwkv_vocab_v20260603.txt")
ids = tokenizer.encode("Hello, RWKV!")
assert tokenizer.decode(ids) == "Hello, RWKV!"

# Load an augmented or generated vocabulary without writing a temporary file.
vocab_bytes = Path("/path/to/rwkv_vocab_v20260603.txt").read_bytes()
tokenizer = WorldTokenizer.from_buffer(vocab_bytes)
```

## Supported release artifacts

Release CI produces source distributions and Linux wheels for:

- CPython 3.10, 3.11, and 3.12;
- `x86_64` (`manylinux_2_17`);
- `aarch64` (`manylinux_2_17`).

Use a release wheel for deployment. Installing directly from the source tree
requires a Rust toolchain and is intended only for development.

## Development checks

```bash
cargo test --release --manifest-path rwkv-tokenizer/Cargo.toml
pytest -q bindings/python/tests/test_package_smoke.py
```
