import gzip
import struct
import tarfile
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
from visualize import class_frequency, image_gallery, load_cifar10, load_mnist


def test_gallery_handles_native_grayscale_and_rgb(tmp_path):
    gray = np.zeros((2, 7, 9), dtype=np.uint8)
    rgb = np.zeros((2, 8, 6, 3), dtype=np.uint8)
    image_gallery(gray, np.array([0, 1]), output=tmp_path / "gray.png")
    image_gallery(rgb, output=tmp_path / "rgb.png")
    assert (tmp_path / "gray.png").is_file()
    assert (tmp_path / "rgb.png").is_file()


def test_frequency_plot_by_splits(tmp_path):
    splits = {"train": (np.zeros((3, 2, 2)), np.array([0, 0, 1])),
              "test": (np.zeros((2, 2, 2)), np.array([0, 1]))}
    output = tmp_path / "counts.png"
    class_frequency(splits, class_names=("zero", "one"), output=output)
    assert output.is_file()


def test_load_small_mnist_idx(tmp_path):
    for prefix, labels in (("train", bytes([2])), ("t10k", bytes([7]))):
        with gzip.open(tmp_path / f"{prefix}-images-idx3-ubyte.gz", "wb") as f:
            f.write(struct.pack(">IIII", 2051, 1, 2, 3) + bytes(range(6)))
        with gzip.open(tmp_path / f"{prefix}-labels-idx1-ubyte.gz", "wb") as f:
            f.write(struct.pack(">II", 2049, 1) + labels)
    loaded = load_mnist(tmp_path)
    assert loaded["train"][0].shape == (1, 2, 3)
    assert loaded["train"][0].dtype == np.uint8
    assert loaded["test"][1].tolist() == [7]


def test_load_small_cifar_archive(tmp_path):
    archive = tmp_path / "tiny.tar.gz"
    record = bytes([3]) + bytes(i % 256 for i in range(3072))
    with tarfile.open(archive, "w:gz") as tar:
        for name in [*(f"data_batch_{i}.bin" for i in range(1, 6)), "test_batch.bin"]:
            info = tarfile.TarInfo("cifar-10-batches-bin/" + name)
            info.size = len(record)
            import io
            tar.addfile(info, io.BytesIO(record))
    result = load_cifar10(archive)
    assert result["train"][0].shape == (5, 32, 32, 3)
    assert result["train"][0].dtype == np.uint8
    assert result["train"][1].tolist() == [3] * 5
