# Building deployment wheels

The project publishes the distribution name `pyrwkv-tokenizer-rwkv7`, while
preserving the import name `pyrwkv_tokenizer` for applications.

## Supported deployment targets

Release CI builds the following Linux wheels:

| Python | x86_64 | aarch64 |
| --- | --- | --- |
| CPython 3.10 | yes | yes |
| CPython 3.11 | yes | yes |
| CPython 3.12 | yes | yes |

The CI target is `manylinux_2_17`, so the wheels can be installed on
compatible glibc-based Linux servers without a Rust compiler.

## Local development build

Use the same Python interpreter that will load the extension:

```bash
cd bindings/python
python -m pip install 'maturin>=1.5,<2.0'
python -m maturin build --release --interpreter "$(command -v python)" --out dist
```

Then install and test the built artifact rather than importing from the source
checkout:

```bash
python -m pip install --force-reinstall --no-deps dist/*.whl
python -m pytest -q tests/test_package_smoke.py
```

## Release procedure

1. Update the version in `bindings/python/pyproject.toml`,
   `bindings/python/Cargo.toml`, and `pyrwkv_tokenizer/__init__.py`.
2. Tag the matching release, for example `v0.9.2`.
3. Push the tag. GitHub Actions creates the source distribution and Linux
   wheels, verifies the x86_64 artifacts, and attaches every artifact to the
   GitHub release.
4. Upload the release artifacts to the internal Python package index used by
   vLLM servers, or install with `--find-links` from the release artifact URL.

Do not publish this fork under the upstream `pyrwkv-tokenizer` distribution
name: it would conflict with the upstream package and make deployments
non-reproducible.
