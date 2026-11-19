"""Configuration, reproducibility, SHA-256 verification, and logging utilities.

Adapted from the ARTE 4110U course starter code (pytorch/nlp/utils.py).
"""
from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import logging
import platform
import random
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


class Params:
    """Simple JSON-backed hyperparameter / configuration container."""

    def __init__(self, json_path: str | Path):
        self.update(json_path)

    def update(self, json_path: str | Path) -> None:
        with open(json_path, encoding="utf-8") as handle:
            self.__dict__.update(json.load(handle))

    def save(self, json_path: str | Path) -> None:
        save_dict_to_json(self.dict, json_path)

    @property
    def dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


def save_dict_to_json(values: dict[str, Any], json_path: str | Path) -> None:
    path = Path(json_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(values, handle, indent=2, allow_nan=False)
        handle.write("\n")


def load_json(json_path: str | Path) -> dict[str, Any]:
    with open(json_path, encoding="utf-8") as handle:
        return json.load(handle)


def sha256_file(file_path: str | Path) -> str:
    """Compute SHA-256 hex digest of a file for integrity / non-mutation checks."""
    h = hashlib.sha256()
    with open(file_path, "rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch RNGs for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_device(request: str = "auto") -> torch.device:
    if request == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(request)


def set_logger(log_path: str | Path) -> logging.Logger:
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
        force=True,
    )
    return logging.getLogger(__name__)


def capture_environment(device: str) -> dict[str, Any]:
    packages = {
        d.metadata["Name"]: d.version
        for d in importlib.metadata.distributions()
        if d.metadata["Name"]
    }
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], stderr=subprocess.DEVNULL, text=True
            ).strip()
        )
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = None, None

    return {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": str(torch.__version__),
        "device": str(device),
        "git_commit": commit,
        "git_dirty": dirty,
        "packages": {
            k: packages[k]
            for k in sorted(packages)
            if k.lower() in {"torch", "numpy", "pandas", "scikit-learn", "matplotlib", "pytest"}
        },
    }


def save_checkpoint(state: dict[str, Any], is_best: bool, checkpoint_dir: str | Path) -> None:
    directory = Path(checkpoint_dir)
    directory.mkdir(parents=True, exist_ok=True)
    torch.save(state, directory / "last.pth.tar")
    if is_best:
        shutil.copyfile(directory / "last.pth.tar", directory / "best.pth.tar")


def load_checkpoint(
    checkpoint_path: str | Path, model: torch.nn.Module, optimizer: torch.optim.Optimizer | None = None
) -> dict[str, Any]:
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    device = next(model.parameters()).device
    state = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(state["state_dict"])
    if optimizer is not None and "optim_dict" in state:
        optimizer.load_state_dict(state["optim_dict"])
    return state


def append_history(csv_path: str | Path, row: dict[str, Any]) -> None:
    path = Path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)
