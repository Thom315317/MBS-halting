import argparse
import csv
import os
import time

import torch
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.utils.data import DataLoader

from .baselines import RelationalGCNClassifier, RelationalGCNHaltingClassifier
from .datasets import VALUES, build_belief_repair_datasets
from .graph import collate_graph_samples
from .model import MBSModel
from .tokenizer import SimpleTokenizer
from .utils import ensure_dir, load_config, resolve_device, save_json, set_seed


# v0.3 release: only the variants used in the published 3-seed campaign are kept.
# - mbs_adaptive_halting          : reference, MBS substrate + ACT-lite halting
# - rgcn_repair_stability         : strong fixed-step baseline (T=8)
# - rgcn_repair_stability_act_forced_t8 : diagnostic baseline (forced T=8 every epoch)
# - rgcn_repair_stability_act_warmup_t8 : diagnostic baseline (3-epoch warmup at T=8 then free ACT)
MBS_VARIANTS = {"mbs_adaptive_halting"}
RGCN_VARIANTS = {
    "rgcn_repair_stability",
    "rgcn_repair_stability_act_forced_t8",
    "rgcn_repair_stability_act_warmup_t8",
}
GRAPH_VARIANTS = MBS_VARIANTS | RGCN_VARIANTS
ALL_VARIANTS = GRAPH_VARIANTS


def build_model(variant, config, tokenizer):
    if variant in RGCN_VARIANTS:
        if variant in {"rgcn_repair_stability_act", "rgcn_repair_stability_act_forced_t8", "rgcn_repair_stability_act_warmup_t8"}:
            halting = config.get("halting", {}) or {}
            force_terminal_step = 8 if variant == "rgcn_repair_stability_act_forced_t8" else None
            warmup_terminal_step = 8 if variant == "rgcn_repair_stability_act_warmup_t8" else None
            return RelationalGCNHaltingClassifier(
                vocab_size=len(tokenizer.tokens),
                num_values=len(VALUES),
                d_model=config.get("d_state", 96),
                num_cell_types=config.get("num_cell_types", 8),
                num_edge_types=config.get("num_edge_types", 12),
                message_steps=int(halting.get("max_message_steps", config.get("message_steps", 16))),
                dropout=config.get("dropout", 0.1),
                halting_config=halting,
                force_terminal_step=force_terminal_step,
                warmup_terminal_step=warmup_terminal_step,
            )
        return RelationalGCNClassifier(
            vocab_size=len(tokenizer.tokens),
            num_values=len(VALUES),
            d_model=config.get("d_state", 96),
            num_cell_types=config.get("num_cell_types", 8),
            num_edge_types=config.get("num_edge_types", 12),
            message_steps=config.get("message_steps", 8),
            dropout=config.get("dropout", 0.1),
        )
    if variant in MBS_VARIANTS:
        halting = config.get("halting", {}) or {}
        message_steps = int(halting.get("max_message_steps", 16))
        return MBSModel(
            vocab_size=len(tokenizer.tokens),
            num_values=len(VALUES),
            d_state=config.get("d_state", 96),
            num_cell_types=config.get("num_cell_types", 8),
            num_edge_types=config.get("num_edge_types", 12),
            num_operation_modes=config.get("num_operation_modes", 4),
            message_steps=message_steps,
            dropout=config.get("dropout", 0.1),
            use_modes=True,
            use_gate=True,
            adaptive_halting=True,
            halting_config=halting,
        )
    raise ValueError(f"unknown variant {variant}")


def make_loaders(config, tokenizer):
    datasets = build_belief_repair_datasets(config, tokenizer)
    batch_size = int(config.get("batch_size", 16))
    loaders = {}
    for split, dataset in datasets.items():
        loaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == "train"),
            collate_fn=lambda samples: collate_graph_samples(samples, tokenizer),
        )
    return loaders


def compute_loss(outputs, batch, config, variant):
    diagnostics = outputs.get("diagnostics", {})
    answer_loss = F.cross_entropy(outputs["logits"], batch["target_value_id"])
    loss = config.get("lambda_answer", 1.0) * answer_loss
    parts = {
        "answer_loss": answer_loss.detach(),
        "conflict_loss": torch.tensor(0.0, device=answer_loss.device),
        "repair_loss": torch.tensor(0.0, device=answer_loss.device),
        "stability_loss": diagnostics.get("stability_loss", torch.tensor(0.0, device=answer_loss.device)),
        "ponder_loss": torch.tensor(0.0, device=answer_loss.device),
    }

    if uses_conflict_loss(variant, config) and outputs.get("conflict_logits") is not None and batch.get("conflict_mask") is not None:
        mask = batch["conflict_mask"]
        if mask.any():
            conflict_loss = F.binary_cross_entropy_with_logits(outputs["conflict_logits"][mask], batch["conflict_labels"][mask])
            loss = loss + config.get("lambda_conflict", 0.2) * conflict_loss
            parts["conflict_loss"] = conflict_loss.detach()

    if uses_repair_loss(variant, config) and outputs.get("repair_logits") is not None:
        repair_labels = batch["repair_labels"]
        mask = repair_labels >= 0
        if mask.any():
            repair_loss = F.cross_entropy(outputs["repair_logits"][mask], repair_labels[mask])
            loss = loss + config.get("lambda_repair", 0.5) * repair_loss
            parts["repair_loss"] = repair_loss.detach()

    if uses_stability_loss(variant, config) and "stability_loss" in diagnostics:
        loss = loss + config.get("lambda_stability", 0.01) * diagnostics["stability_loss"]

    if variant in {"mbs_adaptive_halting", "rgcn_repair_stability_act", "rgcn_repair_stability_act_warmup_t8"} and "expected_steps_mean" in diagnostics:
        ponder_signal = diagnostics.get("ponder_active_signal")
        ponder_active = True if ponder_signal is None else float(ponder_signal.item() if torch.is_tensor(ponder_signal) else ponder_signal) >= 0.5
        if ponder_active:
            lambda_ponder = float((config.get("halting", {}) or {}).get("lambda_ponder", config.get("lambda_ponder", 0.001)))
            ponder_loss = lambda_ponder * diagnostics["expected_steps_mean"]
            loss = loss + ponder_loss
            parts["ponder_loss"] = ponder_loss.detach()

    if "mode_entropy" in diagnostics:
        loss = loss - config.get("lambda_mode_entropy", 0.001) * diagnostics["mode_entropy"]

    return loss, parts


def uses_conflict_loss(variant, config):
    return float(config.get("lambda_conflict", 0.2)) != 0.0 and variant in MBS_VARIANTS


def uses_repair_loss(variant, config):
    if float(config.get("lambda_repair", 0.5)) == 0.0:
        return False
    return variant in GRAPH_VARIANTS


def uses_stability_loss(variant, config):
    if float(config.get("lambda_stability", 0.01)) == 0.0:
        return False
    return variant in GRAPH_VARIANTS


@torch.no_grad()
def evaluate(model, loader, device, config, variant, message_steps=None):
    model.eval()
    total = 0
    correct = 0
    total_loss = 0.0
    metric_sums = {}
    metric_counts = {}
    for batch in loader:
        batch = move_batch(batch, device)
        outputs = model(batch, message_steps=message_steps)
        loss, parts = compute_loss(outputs, batch, config, variant)
        preds = outputs["logits"].argmax(dim=-1)
        correct += (preds == batch["target_value_id"]).sum().item()
        total += batch["target_value_id"].numel()
        total_loss += loss.item() * batch["target_value_id"].numel()
        for key, value in {**parts, **outputs.get("diagnostics", {})}.items():
            for metric_key, metric_value in flatten_metric(key, value):
                metric_sums[metric_key] = metric_sums.get(metric_key, 0.0) + metric_value
                metric_counts[metric_key] = metric_counts.get(metric_key, 0) + 1
    metrics = {key: metric_sums[key] / metric_counts[key] for key in metric_sums}
    metrics["acc"] = correct / max(total, 1)
    metrics["loss"] = total_loss / max(total, 1)
    return metrics


def train_one(config, variant, output_dir, checkpoint_dir):
    set_seed(int(config.get("seed", 1)))
    tokenizer = SimpleTokenizer()
    loaders = make_loaders(config, tokenizer)
    device = resolve_device(config.get("device", "cuda_if_available"))
    model = build_model(variant, config, tokenizer).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    optimizer = AdamW(model.parameters(), lr=float(config.get("lr", 1e-3)), weight_decay=float(config.get("weight_decay", 0.01)))
    scaler = GradScaler("cuda", enabled=(device.type == "cuda"))
    ensure_dir(output_dir)
    ensure_dir(checkpoint_dir)
    selected_row = None
    selected_checkpoint_path = os.path.join(checkpoint_dir, f"{variant}_best.pt")
    history = []

    warmup_epochs = int((config.get("halting", {}) or {}).get("warmup_epochs", 3))
    max_epochs = int(config.get("max_epochs", 5))
    epoch_metrics_csv = os.path.join(output_dir, f"{variant}_epoch_metrics.csv")
    csv_header_written = False
    run_start = time.time()
    print(
        f"[run] variant={variant} seed={config.get('seed')} device={device.type}"
        f"{':' + str(device.index) if device.index is not None else ''} "
        f"batch_size={int(config.get('batch_size', 16))} parameter_count={parameter_count} "
        f"max_epochs={max_epochs}",
        flush=True,
    )
    for epoch in range(1, max_epochs + 1):
        epoch_start = time.time()
        if variant == "rgcn_repair_stability_act_warmup_t8" and hasattr(model, "set_warmup_active"):
            is_warmup = epoch <= warmup_epochs
            model.set_warmup_active(is_warmup)
            print(
                f"[warmup_t8] epoch {epoch}: warmup="
                f"{'ACTIVE (force step 8, ponder=0)' if is_warmup else 'INACTIVE (free ACT)'}",
                flush=True,
            )
        model.train()
        train_metrics = []
        for batch in loaders["train"]:
            batch = move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with autocast(device_type=device.type, enabled=(device.type == "cuda")):
                outputs = model(batch)
                loss, parts = compute_loss(outputs, batch, config, variant)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(config.get("grad_clip", 1.0)))
            scaler.step(optimizer)
            scaler.update()
            with torch.no_grad():
                preds = outputs["logits"].argmax(dim=-1)
                train_metrics.append({"acc": (preds == batch["target_value_id"]).float().mean().item(), "loss": loss.item()})

        val = evaluate(model, loaders["val"], device, config, variant)
        ood = {split: evaluate(model, loaders[split], device, config, variant) for split in ["ood_entity", "ood_conflict", "ood_rule", "ood_mixed"]}
        train_acc = sum(item["acc"] for item in train_metrics) / max(len(train_metrics), 1)
        train_loss = sum(item["loss"] for item in train_metrics) / max(len(train_metrics), 1)
        row = {
            "epoch": epoch,
            "train_acc": train_acc,
            "train_loss": train_loss,
            "val_acc": val["acc"],
            "val_loss": val["loss"],
            "ood_entity_acc": ood["ood_entity"]["acc"],
            "ood_conflict_acc": ood["ood_conflict"]["acc"],
            "ood_rule_acc": ood["ood_rule"]["acc"],
            "ood_mixed_acc": ood["ood_mixed"]["acc"],
        }
        for key in [
            "answer_loss",
            "conflict_loss",
            "repair_loss",
            "stability_loss",
            "energy_mean",
            "mode_entropy",
            "update_scale_mean",
            "update_scale_std",
            "update_norm_mean",
            "state_norm_mean",
            "mode_PROPAGATE_mean",
            "mode_STABILIZE_mean",
            "mode_REPAIR_mean",
            "mode_RESOLVE_CONFLICT_mean",
            "expected_steps_mean",
            "final_step_mass_mean",
            "halt_weight_sum_mean",
            "halt_weight_sum_std",
            "ponder_loss",
        ]:
            if key in val:
                row[f"val_{key}"] = val[key]
        for key in [
            "expected_steps_mean",
            "final_step_mass_mean",
            "halt_weight_sum_mean",
            "halt_weight_sum_std",
            "ponder_loss",
        ]:
            if key in ood["ood_mixed"]:
                row[f"ood_mixed_{key}"] = ood["ood_mixed"][key]
        for key, value in val.items():
            if key.startswith("halt_prob_mean_by_step_") or key.startswith("halt_weight_mean_by_step_"):
                row[f"val_{key}"] = value
        epoch_duration = time.time() - epoch_start
        elapsed = time.time() - run_start
        eta = epoch_duration * (max_epochs - epoch)
        row["epoch_duration_seconds"] = epoch_duration
        row["elapsed_seconds"] = elapsed
        row["eta_remaining_seconds"] = eta
        if device.type == "cuda":
            row["gpu_mem_allocated_mb"] = torch.cuda.memory_allocated(device) / (1024 * 1024)
            row["gpu_mem_reserved_mb"] = torch.cuda.memory_reserved(device) / (1024 * 1024)
        history.append(row)
        write_epoch_metrics_csv(epoch_metrics_csv, row, write_header=not csv_header_written)
        csv_header_written = True
        print_epoch_block(variant, config, epoch, max_epochs, row, val, ood, device, parameter_count, history)
        if variant == "mbs_adaptive_halting":
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "variant": variant,
                    "config": config,
                    "tokenizer": tokenizer.state_dict(),
                    "epoch": epoch,
                },
                os.path.join(checkpoint_dir, f"{variant}_epoch_{epoch}.pt"),
            )
        if checkpoint_row_is_better(row, selected_row):
            selected_row = row
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "variant": variant,
                    "config": config,
                    "tokenizer": tokenizer.state_dict(),
                    "epoch": epoch,
                },
                selected_checkpoint_path,
            )
        metadata = build_training_metadata(history, selected_row, selected_checkpoint_path, variant)
        for warning in metadata["warnings"]:
            print(warning)
        save_json(
            os.path.join(output_dir, f"{variant}_train_results.json"),
            {
                "variant": variant,
                "config": config,
                "history": history,
                "parameter_count": parameter_count,
                **metadata,
            },
        )
    return history


def move_batch(batch, device):
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


def flatten_metric(key, value):
    if torch.is_tensor(value):
        value = value.detach().float()
        if value.numel() == 1:
            yield key, float(value.item())
            return
        flat = value.reshape(-1)
        for idx, item in enumerate(flat, start=1):
            yield f"{key}_{idx:02d}", float(item.item())
        return
    if isinstance(value, (list, tuple)):
        for idx, item in enumerate(value, start=1):
            yield f"{key}_{idx:02d}", float(item)
        return
    yield key, float(value)


def checkpoint_row_is_better(candidate, selected, eps=1e-12):
    if selected is None:
        return True
    candidate_val = float(candidate["val_acc"])
    selected_val = float(selected["val_acc"])
    if candidate_val > selected_val + eps:
        return True
    if abs(candidate_val - selected_val) <= eps:
        candidate_loss = float(candidate["val_loss"])
        selected_loss = float(selected["val_loss"])
        if candidate_loss < selected_loss - eps:
            return True
    return False


def build_training_metadata(history, selected_row, selected_checkpoint_path, variant):
    selected_row = selected_row or history[-1]
    best_ood_row = max(history, key=lambda row: float(row.get("ood_mixed_acc", float("-inf"))))
    selected_ood = float(selected_row.get("ood_mixed_acc", 0.0))
    best_ood = float(best_ood_row.get("ood_mixed_acc", 0.0))
    regression = best_ood - selected_ood
    checkpoint_policy = {
        "primary_metric": "val_acc",
        "mode": "max",
        "tie_breaker": ["val_loss:min", "earlier_epoch"],
        "selected_epoch": int(selected_row["epoch"]),
        "selected_metric_value": float(selected_row["val_acc"]),
        "selected_checkpoint_path": str(selected_checkpoint_path),
        "selected_epoch_val_acc": float(selected_row["val_acc"]),
        "selected_epoch_val_loss": float(selected_row["val_loss"]),
        "selected_epoch_ood_mixed_acc": selected_ood,
    }
    if variant == "mbs_adaptive_halting":
        checkpoint_policy.update(
            {
                "selected_epoch_expected_steps_val": selected_row.get("val_expected_steps_mean"),
                "selected_epoch_final_step_mass_val": selected_row.get("val_final_step_mass_mean"),
                "selected_epoch_expected_steps_ood_mixed": selected_row.get("ood_mixed_expected_steps_mean"),
                "selected_epoch_final_step_mass_ood_mixed": selected_row.get("ood_mixed_final_step_mass_mean"),
            }
        )
    best_ood_mixed_epoch = {
        "epoch": int(best_ood_row["epoch"]),
        "ood_mixed_acc": best_ood,
        "val_acc": float(best_ood_row["val_acc"]),
        "val_loss": float(best_ood_row["val_loss"]),
    }
    expected_steps = best_ood_row.get("ood_mixed_expected_steps_mean", best_ood_row.get("val_expected_steps_mean"))
    final_step_mass = best_ood_row.get("ood_mixed_final_step_mass_mean", best_ood_row.get("val_final_step_mass_mean"))
    if expected_steps is not None:
        best_ood_mixed_epoch["expected_steps_mean"] = expected_steps
    if final_step_mass is not None:
        best_ood_mixed_epoch["final_step_mass_mean"] = final_step_mass
    warning = None
    if regression > 0.03:
        warning = "WARNING: selected checkpoint is substantially worse than best OOD epoch."
    return {
        "checkpoint_policy": checkpoint_policy,
        "best_ood_mixed_epoch": best_ood_mixed_epoch,
        "ood_regression_from_best_epoch": regression,
        "warnings": [warning] if warning else [],
    }


def write_epoch_metrics_csv(path, row, write_header=False):
    flat = {key: value for key, value in row.items() if not isinstance(value, (list, tuple, dict))}
    fieldnames = sorted(flat.keys())
    mode = "w" if write_header else "a"
    with open(path, mode, encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow({key: flat.get(key) for key in fieldnames})


def _format_seconds(seconds):
    seconds = int(round(float(seconds)))
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    if h:
        return f"{h}h{m:02d}m{s:02d}s"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


def _format_step_table(row, prefix, max_steps=16):
    keys = [f"{prefix}{idx:02d}" for idx in range(1, max_steps + 1) if f"{prefix}{idx:02d}" in row]
    if not keys:
        return None
    items = [f"{key.split('_')[-1]}:{float(row[key]):.3f}" for key in keys]
    return "  ".join(items)


def print_epoch_block(variant, config, epoch, max_epochs, row, val, ood_dict, device, parameter_count, history):
    head_lines = [
        "=" * 72,
        f"Variant: {variant} | seed={config.get('seed')} | epoch {epoch}/{max_epochs}"
        + (
            f" | mode={'WARMUP_T8' if epoch <= int((config.get('halting', {}) or {}).get('warmup_epochs', 3)) else 'FREE_ACT'}"
            if variant == "rgcn_repair_stability_act_warmup_t8"
            else ""
        ),
        f"device={device.type}{':' + str(device.index) if device.index is not None else ''} "
        f"batch_size={int(config.get('batch_size', 16))} parameter_count={parameter_count}"
        + (
            f" | gpu_alloc={row.get('gpu_mem_allocated_mb', 0):.0f}MB "
            f"gpu_reserved={row.get('gpu_mem_reserved_mb', 0):.0f}MB"
            if device.type == "cuda"
            else ""
        ),
        f"duration={_format_seconds(row['epoch_duration_seconds'])}  "
        f"elapsed={_format_seconds(row['elapsed_seconds'])}  "
        f"ETA_remaining={_format_seconds(row['eta_remaining_seconds'])}",
    ]
    perf = (
        f"Performance:\n"
        f"  train_acc={row['train_acc']:.4f}  val_acc={row['val_acc']:.4f}\n"
        f"  ood_entity={row['ood_entity_acc']:.4f}  ood_conflict={row['ood_conflict_acc']:.4f}  "
        f"ood_rule={row['ood_rule_acc']:.4f}  ood_mixed={row['ood_mixed_acc']:.4f}"
    )
    losses = (
        f"Losses (val):\n"
        f"  total={row['val_loss']:.4f}  "
        f"answer={row.get('val_answer_loss', float('nan')):.4f}  "
        f"conflict={row.get('val_conflict_loss', float('nan')):.4f}  "
        f"repair={row.get('val_repair_loss', float('nan')):.4f}  "
        f"stability={row.get('val_stability_loss', float('nan')):.4f}  "
        f"ponder={row.get('val_ponder_loss', float('nan')):.6f}"
    )
    halting_lines = []
    if "val_expected_steps_mean" in row:
        halting_lines.append(
            f"Halting:\n"
            f"  E[steps] val={row['val_expected_steps_mean']:.4f}  "
            f"ood_mixed={row.get('ood_mixed_expected_steps_mean', float('nan')):.4f}\n"
            f"  final_mass val={row.get('val_final_step_mass_mean', float('nan')):.4f}  "
            f"ood_mixed={row.get('ood_mixed_final_step_mass_mean', float('nan')):.4f}\n"
            f"  halt_weight_sum: mean={row.get('val_halt_weight_sum_mean', float('nan')):.4f} "
            f"std={row.get('val_halt_weight_sum_std', float('nan')):.4f}"
        )
        prob_table = _format_step_table(row, "val_halt_prob_mean_by_step_")
        weight_table = _format_step_table(row, "val_halt_weight_mean_by_step_")
        if prob_table:
            halting_lines.append("  halt_prob_mean_by_step (val):  " + prob_table)
        if weight_table:
            halting_lines.append("  halt_weight_mean_by_step (val):  " + weight_table)
    dynamics = (
        f"Dynamics:\n"
        f"  update_norm_mean={row.get('val_update_norm_mean', float('nan')):.4f}  "
        f"state_norm_mean={row.get('val_state_norm_mean', float('nan')):.4f}  "
        f"update_scale_mean={row.get('val_update_scale_mean', float('nan')):.4f} "
        f"(std={row.get('val_update_scale_std', float('nan')):.4f})"
    )
    best_ood_row = max(history, key=lambda r: float(r.get("ood_mixed_acc", float("-inf"))))
    selected = max(history, key=lambda r: float(r.get("val_acc", float("-inf"))))
    regression = float(best_ood_row.get("ood_mixed_acc", 0.0)) - float(selected.get("ood_mixed_acc", 0.0))
    ckpt_lines = [
        f"Checkpoint policy (running):",
        f"  selected_epoch={selected['epoch']} (val_acc={selected['val_acc']:.4f}, "
        f"ood_mixed={selected['ood_mixed_acc']:.4f})",
        f"  best_ood_epoch={best_ood_row['epoch']} (ood_mixed={best_ood_row['ood_mixed_acc']:.4f})",
        f"  ood_regression_from_best={regression:.4f}"
        + ("  WARNING: selected ckpt worse than best OOD epoch" if regression > 0.03 else ""),
    ]
    block = "\n".join(head_lines + [perf, losses] + halting_lines + [dynamics] + ckpt_lines + ["=" * 72])
    print(block, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--variant",
        required=True,
        choices=sorted(ALL_VARIANTS),
    )
    parser.add_argument("--output-dir", default="results/belief_repair_debug")
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    args = parser.parse_args()
    config = load_config(args.config)
    train_one(config, args.variant, args.output_dir, args.checkpoint_dir)


if __name__ == "__main__":
    main()
