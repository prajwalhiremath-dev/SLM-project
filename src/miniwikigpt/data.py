from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import torch


Split = Literal["train", "val"]


class BinBatcher:
    """Random batch sampler over tokenized `.bin` files.

    The `.bin` files are memory-mapped, so even large tokenized datasets can be
    sampled without loading the full file into RAM.
    """

    def __init__(
        self,
        data_dir: str | Path,
        block_size: int,
        batch_size: int,
        device: str,
        dtype: str = "uint16",
    ) -> None:
        self.data_dir = Path(data_dir)
        self.block_size = block_size
        self.batch_size = batch_size
        self.device = device
        self.dtype = np.dtype(dtype)

        train_path = self.data_dir / "train.bin"
        val_path = self.data_dir / "val.bin"
        if not train_path.exists():
            raise FileNotFoundError(f"Missing tokenized train file: {train_path}")
        if not val_path.exists():
            raise FileNotFoundError(f"Missing tokenized val file: {val_path}")

        self.train_data = np.memmap(train_path, dtype=self.dtype, mode="r")
        self.val_data = np.memmap(val_path, dtype=self.dtype, mode="r")

        if len(self.train_data) <= block_size:
            raise ValueError("train.bin is too small for the configured block_size")
        if len(self.val_data) <= block_size:
            raise ValueError("val.bin is too small for the configured block_size")

    def get_batch(self, split: Split) -> tuple[torch.Tensor, torch.Tensor]:
        data = self.train_data if split == "train" else self.val_data
        ix = torch.randint(len(data) - self.block_size, (self.batch_size,))

        x = torch.stack(
            [torch.from_numpy(np.array(data[i : i + self.block_size], dtype=np.int64)) for i in ix]
        )
        y = torch.stack(
            [
                torch.from_numpy(np.array(data[i + 1 : i + 1 + self.block_size], dtype=np.int64))
                for i in ix
            ]
        )

        if self.device.startswith("cuda"):
            x = x.pin_memory().to(self.device, non_blocking=True)
            y = y.pin_memory().to(self.device, non_blocking=True)
        else:
            x = x.to(self.device)
            y = y.to(self.device)

        return x, y
