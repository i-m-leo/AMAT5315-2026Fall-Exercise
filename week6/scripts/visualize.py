#!/usr/bin/env python3
"""Native-resolution image galleries and split-wise label frequencies.

Supports CIFAR-10 binary batches and MNIST IDX gzip files. Pixel values are
passed straight to Matplotlib without resizing or normalization.
"""
from __future__ import annotations

import argparse
import gzip
import struct
import tarfile
from pathlib import Path
from typing import Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CIFAR10_CLASSES = ("airplane", "automobile", "bird", "cat", "deer",
                   "dog", "frog", "horse", "ship", "truck")


def load_cifar10(archive: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Return train/test RGB images (N,32,32,3) and integer labels."""
    result: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    with tarfile.open(archive, "r:gz") as tar:
        for split, members in (("train", [f"data_batch_{i}.bin" for i in range(1, 6)]),
                               ("test", ["test_batch.bin"])):
            images, labels = [], []
            for name in members:
                matches = [m for m in tar.getmembers() if m.name.endswith("/" + name)]
                if len(matches) != 1:
                    raise ValueError(f"Expected one {name} in {archive}")
                stream = tar.extractfile(matches[0])
                if stream is None:
                    raise ValueError(f"Cannot read {name}")
                data = np.frombuffer(stream.read(), dtype=np.uint8)
                if data.size % 3073:
                    raise ValueError(f"Invalid CIFAR-10 batch size in {name}")
                records = data.reshape(-1, 3073)
                labels.append(records[:, 0].copy())
                # CIFAR stores channel planes; transpose to HWC for imshow.
                images.append(records[:, 1:].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1).copy())
            result[split] = (np.concatenate(images), np.concatenate(labels))
    return result


def load_mnist(directory: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Load standard MNIST IDX gzip files as uint8 images and integer labels."""
    result = {}
    for split, prefix in (("train", "train"), ("test", "t10k")):
        with gzip.open(directory / f"{prefix}-images-idx3-ubyte.gz", "rb") as f:
            magic, count, rows, cols = struct.unpack(">IIII", f.read(16))
            if magic != 2051:
                raise ValueError(f"Bad MNIST image magic number: {magic}")
            images = np.frombuffer(f.read(), dtype=np.uint8)
        with gzip.open(directory / f"{prefix}-labels-idx1-ubyte.gz", "rb") as f:
            label_magic, label_count = struct.unpack(">II", f.read(8))
            if label_magic != 2049:
                raise ValueError(f"Bad MNIST label magic number: {label_magic}")
            labels = np.frombuffer(f.read(), dtype=np.uint8).copy()
        if images.size != count * rows * cols or label_count != count or labels.size != count:
            raise ValueError(f"Inconsistent MNIST {split} image/label counts")
        result[split] = (images.reshape(count, rows, cols), labels)
    return result


def image_gallery(images: np.ndarray, labels: np.ndarray | None = None, *,
                  indices: Sequence[int] | None = None, class_names: Sequence[str] | None = None,
                  count: int = 25, columns: int = 5, title: str = "Image gallery",
                  output: Path) -> None:
    """Write selected images in a grid at native pixel dimensions."""
    if images.ndim not in (3, 4) or (images.ndim == 4 and images.shape[-1] not in (3, 4)):
        raise ValueError("images must have shape (N,H,W), (N,H,W,3), or (N,H,W,4)")
    n = len(images)
    chosen = np.asarray(indices if indices is not None else np.arange(min(count, n)), dtype=int)
    if chosen.size == 0 or np.any(chosen < 0) or np.any(chosen >= n):
        raise ValueError("indices must select at least one valid image")
    if labels is not None and len(labels) != n:
        raise ValueError("labels and images must have the same length")
    columns = max(1, min(columns, len(chosen)))
    rows = (len(chosen) + columns - 1) // columns
    height, width = images.shape[1:3]
    fig, axes = plt.subplots(rows, columns, figsize=(columns * 2.0, rows * 2.0),
                             squeeze=False, layout="constrained")
    for ax in axes.flat:
        ax.set_axis_off()
    for ax, index in zip(axes.flat, chosen):
        im = images[index]
        if im.ndim == 2:
            ax.imshow(im, cmap="gray", vmin=0 if im.dtype == np.uint8 else None,
                      vmax=255 if im.dtype == np.uint8 else None,
                      interpolation="nearest", origin="upper")
        else:
            ax.imshow(im, interpolation="nearest", origin="upper")
        if labels is not None:
            label = int(labels[index])
            caption = str(class_names[label]) if class_names is not None and 0 <= label < len(class_names) else str(label)
            ax.set_title(caption, fontsize=9)
    fig.suptitle(f"{title} — {width}×{height} native pixels")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    plt.close(fig)


def class_frequency(splits: dict[str, tuple[np.ndarray, np.ndarray]], *,
                    class_names: Sequence[str] | None = None, output: Path,
                    title: str = "Class frequencies by split") -> None:
    """Plot label counts for each supplied split (typically train and test)."""
    if not splits:
        raise ValueError("at least one split is required")
    labels_all = [np.asarray(labels, dtype=int) for _, labels in splits.values()]
    if any(labels.size == 0 for labels in labels_all):
        raise ValueError("each split must contain labels")
    classes = np.arange(max(int(labels.max()) for labels in labels_all) + 1)
    names = [str(class_names[i]) if class_names is not None and i < len(class_names) else str(i)
             for i in classes]
    width = 0.8 / len(splits)
    x = np.arange(len(classes))
    fig, ax = plt.subplots(figsize=(max(8, len(classes) * 0.8), 4.5), layout="constrained")
    for j, (split, (_, labels)) in enumerate(splits.items()):
        counts = np.bincount(np.asarray(labels, dtype=int), minlength=len(classes))[:len(classes)]
        ax.bar(x + (j - (len(splits) - 1) / 2) * width, counts, width, label=split)
    ax.set_xticks(x, names, rotation=30, ha="right")
    ax.set_ylabel("Image count")
    ax.set_title(title)
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    plt.close(fig)


def empirical_distribution_figure(
    splits: dict[str, tuple[np.ndarray, np.ndarray]], *,
    class_names: Sequence[str], output: Path, examples_per_class: int = 3,
    seed: int = 5315, title: str = "Empirical dataset distribution",
) -> None:
    """Combine labeled examples per class and relative split frequencies.

    Examples are sampled without replacement from the training split where
    possible, otherwise the first available split, using a fixed RNG seed.
    The chart shows proportions within each split, not raw sample counts.
    """
    if "train" not in splits or "test" not in splits:
        raise ValueError("both train and test splits are required")
    if examples_per_class < 1:
        raise ValueError("examples_per_class must be positive")
    reference = splits["train"]
    images, labels = reference
    class_ids = np.arange(len(class_names))
    if not np.isin(labels, class_ids).all():
        raise ValueError("training labels are outside class_names")

    fig = plt.figure(figsize=(14, 9), layout="constrained")
    grid = fig.add_gridspec(len(class_ids) + 1, examples_per_class + 1,
                            height_ratios=[1] * len(class_ids) + [1.5])
    rng = np.random.default_rng(seed)
    for row, class_id in enumerate(class_ids):
        candidates = np.flatnonzero(labels == class_id)
        if candidates.size == 0:
            raise ValueError(f"no training examples for class {class_names[class_id]}")
        selected = rng.choice(candidates, size=min(examples_per_class, candidates.size),
                              replace=False)
        for col in range(examples_per_class):
            ax = fig.add_subplot(grid[row, col])
            ax.set_axis_off()
            if col < len(selected):
                image = images[selected[col]]
                if image.ndim == 2:
                    ax.imshow(image, cmap="gray", vmin=0 if image.dtype == np.uint8 else None,
                              vmax=255 if image.dtype == np.uint8 else None,
                              interpolation="nearest", origin="upper")
                else:
                    ax.imshow(image, interpolation="nearest", origin="upper")
                ax.set_title(f"{class_names[class_id]} · example {col + 1}", fontsize=8)
        if examples_per_class < len(class_names) + 1:
            ax = fig.add_subplot(grid[row, examples_per_class])
            ax.text(0.98, 0.5, class_names[class_id], ha="right", va="center",
                    transform=ax.transAxes, fontsize=10, fontweight="bold")
            ax.set_axis_off()

    ax = fig.add_subplot(grid[-1, :])
    x = np.arange(len(class_ids))
    width = 0.8 / len(splits)
    for j, (split, (_, split_labels)) in enumerate(splits.items()):
        split_labels = np.asarray(split_labels, dtype=int)
        if split_labels.size == 0 or not np.isin(split_labels, class_ids).all():
            raise ValueError(f"invalid or empty labels in {split} split")
        proportions = np.bincount(split_labels, minlength=len(class_ids)) / split_labels.size
        offset = (j - (len(splits) - 1) / 2) * width
        ax.bar(x + offset, proportions, width, label=split)
    ax.set_xticks(x, class_names, rotation=25, ha="right")
    ax.set_ylabel("Relative frequency (count / split size)")
    ax.set_title("Class frequencies by split")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda value, _: f"{value:.0%}"))
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.suptitle(title)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=("cifar10", "mnist"))
    parser.add_argument("--input", type=Path, required=True,
                        help="CIFAR-10 tar.gz archive or MNIST IDX gzip directory")
    parser.add_argument("--split", choices=("train", "test", "both"), default="both")
    parser.add_argument("--count", type=int, default=25, help="gallery size per split")
    parser.add_argument("--examples-per-class", type=int, default=3,
                        help="number of labeled training examples per class in the combined figure")
    parser.add_argument("--no-labels", action="store_true", help="omit gallery captions")
    parser.add_argument("--output-dir", type=Path, default=Path("evidence"))
    args = parser.parse_args()
    splits = load_cifar10(args.input) if args.dataset == "cifar10" else load_mnist(args.input)
    names = CIFAR10_CLASSES if args.dataset == "cifar10" else tuple(str(i) for i in range(10))
    selected = splits if args.split == "both" else {args.split: splits[args.split]}
    for split, (images, labels) in selected.items():
        image_gallery(images, None if args.no_labels else labels, class_names=names,
                      count=args.count, title=f"{args.dataset.upper()} {split}",
                      output=args.output_dir / f"{args.dataset}-{split}-gallery.png")
    if args.split == "both":
        empirical_distribution_figure(
            splits, class_names=names, examples_per_class=args.examples_per_class,
            title=f"{args.dataset.upper()} empirical distribution",
            output=args.output_dir / f"{args.dataset}-empirical-distribution.png")
    else:
        class_frequency(selected, class_names=names,
                        output=args.output_dir / f"{args.dataset}-class-frequencies.png")


if __name__ == "__main__":
    main()
