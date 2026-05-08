import json
import os
import random

import torch
import yaml


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def save_json(path: str, data: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def resolve_device(name: str) -> torch.device:
    if name == "cuda_if_available":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def mean_dict(dicts):
    out = {}
    counts = {}
    for item in dicts:
        for key, value in item.items():
            if value is None:
                continue
            out[key] = out.get(key, 0.0) + float(value)
            counts[key] = counts.get(key, 0) + 1
    return {key: out[key] / counts[key] for key in out}
