# Week 6

Reusable dataset visualization tools for AMAT5315-2026Fall.

## Setup

Install dependencies with `uv` (CPU-only; no ML framework or accelerator is used):

```sh
uv sync --extra test
uv run pytest
```

## Visualization

`scripts/visualize.py` reads CIFAR-10 binary archives and MNIST IDX gzip files.
It produces native-resolution grayscale/RGB image galleries and split-wise
class-frequency plots. Image arrays retain their original dimensions and
integer pixel values; galleries use nearest-neighbor rendering without resize
or normalization. Labels can be hidden with `--no-labels`.

From `week6/`, for both CIFAR-10 splits (outputs saved in `evidence/`):

```sh
uv run python scripts/visualize.py cifar10 \\
  --input inputs/cifar10/cifar-10-binary.tar.gz --split both --count 20
```

For MNIST:

```sh
uv run python scripts/visualize.py mnist --input inputs/mnist --split both --count 20
```

Use `--split train` or `--split test`, `--no-labels`, or `--output-dir PATH`
to customize output. The default output directory is `evidence/`. The reusable
functions `load_cifar10`, `load_mnist`, `image_gallery`, and `class_frequency`
are importable from `scripts/visualize.py`.

Dataset downloads and generated raw runs are ignored by git; generated plots
in `evidence/` are committed as visual evidence.

## Repository conventions

- Use `uv` for Python environment and package management; CPU is the default.
- Save generated figures in `evidence/`.
- Save raw runs and their settings in `artifacts/`.
- Keep downloads and raw run outputs out of git.
