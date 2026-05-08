import argparse
import os

import torch
from torch.utils.data import DataLoader

from .datasets import build_belief_repair_datasets
from .graph import collate_graph_samples
from .tokenizer import SimpleTokenizer
from .train import GRAPH_VARIANTS, build_model, evaluate, move_batch
from .utils import load_config, resolve_device, save_json, set_seed


def sweep_steps_for_variant(variant):
    if variant == "mbs_adaptive_halting":
        return [8, 16, 32]
    if variant in GRAPH_VARIANTS:
        return [2, 4, 8, 16, 32]
    return [None]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default=None)
    parser.add_argument("--output-dir", default="results/belief_repair_debug")
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    config = dict(checkpoint.get("config") or {})
    if args.config:
        config.update(load_config(args.config))
    variant = checkpoint["variant"]
    set_seed(int(config.get("seed", 1)))
    tokenizer = SimpleTokenizer.from_state_dict(checkpoint.get("tokenizer", SimpleTokenizer().state_dict()))
    datasets = build_belief_repair_datasets(config, tokenizer)
    loaders = {
        split: DataLoader(
            dataset,
            batch_size=int(config.get("batch_size", 16)),
            shuffle=False,
            collate_fn=lambda samples: collate_graph_samples(samples, tokenizer),
        )
        for split, dataset in datasets.items()
        if split != "train"
    }
    device = resolve_device(config.get("device", "cuda_if_available"))
    model = build_model(variant, config, tokenizer).to(device)
    model.load_state_dict(checkpoint["model_state"])

    results = {"variant": variant, "config": config, "splits": {}, "sweep": []}
    for split, loader in loaders.items():
        results["splits"][split] = evaluate(model, loader, device, config, variant)

    for steps in sweep_steps_for_variant(variant):
        row = {"message_steps": steps, "splits": {}}
        for split, loader in loaders.items():
            row["splits"][split] = evaluate(model, loader, device, config, variant, message_steps=steps)
        results["sweep"].append(row)

    if results["sweep"] and len(results["sweep"]) > 1:
        first = results["sweep"][0]["splits"]["ood_mixed"]["loss"]
        last = results["sweep"][-1]["splits"]["ood_mixed"]["loss"]
        results["ood_mixed_loss_degradation"] = last - first
        results["best_ood_mixed_sweep_acc"] = max(row["splits"]["ood_mixed"]["acc"] for row in results["sweep"])

    output_path = os.path.join(args.output_dir, f"{variant}_eval_results.json")
    save_json(output_path, results)
    print(f"Saved {output_path}")
    for row in results["sweep"]:
        print("steps", row["message_steps"], "ood_mixed_acc", f"{row['splits']['ood_mixed']['acc']:.4f}", "loss", f"{row['splits']['ood_mixed']['loss']:.4f}")


if __name__ == "__main__":
    main()
