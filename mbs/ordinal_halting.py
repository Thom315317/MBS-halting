"""H7 ordinal-calibration helpers.

This module is **only imported by `mbs/train.py`** and only inside code
paths gated by `halting_ordinal.enabled` / `checkpoint_gate.enabled`.
No other module depends on it. H6 configs that do not declare these
config keys exercise none of this code.

Contains :

1. `ordinal_pairwise_loss`              — adjacent-balanced pairwise
                                           ranking loss on E[step].
2. `derive_required_hops_v1_from_metadata` — single-sample helper.
3. `compute_gate_metrics`               — val-only metric set
                                           consumed by the checkpoint
                                           gate.
4. `classify_collapse_flags`            — taxonomy
                                           (hard_floor / hard_final /
                                           soft_middle_step /
                                           binary_h9_shortcut /
                                           ordinal_healthy).
5. `gate_eligible`                      — apply the val-only gate
                                           thresholds to a per-epoch
                                           metric dict.

Reviewer-proof invariants :
  - the gate operates on validation data only (never OOD).
  - thresholds are never relaxed silently ; a failing run gets
    `no_eligible_checkpoint=True` and the highest-val_acc fallback,
    with the reasons recorded.
  - the ordinal loss uses generator-metadata `required_hops` from the
    **training** split only ; it is supervised calibration, not
    emergent.
"""
from __future__ import annotations

import math
import statistics
from collections import Counter, defaultdict
from typing import Dict, List, Tuple

import torch


ADJACENT_BOUNDARIES: List[Tuple[int, int]] = [(5, 6), (6, 7), (7, 8), (8, 9)]
AUC_THRESHOLDS: List[int] = [6, 7, 8, 9]


# ----------------------------------------------------------------------
# 1. Required-hops derivation (single sample)
# ----------------------------------------------------------------------

def derive_required_hops_v1_from_metadata(meta) -> int | None:
    """Return required_hops for a v1 sample, or None if the metadata
    is missing / malformed.

    Formula : (max(candidate_ranks_used) - winner_rank) + 2.
    See `scripts/audit_rgcn_h6_two_stage_controller_vs_required_hops.py`.
    """
    if not meta:
        return None
    crs = meta.get("candidate_ranks_used")
    wr = meta.get("winner_rank")
    if crs is None or wr is None:
        return None
    try:
        return (max(crs) - int(wr)) + 2
    except (TypeError, ValueError):
        return None


# ----------------------------------------------------------------------
# 2. Ordinal pairwise ranking loss
# ----------------------------------------------------------------------

def _sample_pairs_adjacent_balanced(
    required_hops: torch.Tensor,
    boundaries: List[Tuple[int, int]],
    max_pairs_per_batch: int,
    generator: torch.Generator | None = None,
) -> Dict[Tuple[int, int], List[Tuple[int, int]]]:
    """Sample roughly-equal-count pairs across the listed boundaries.

    Returns a dict mapping (h_low, h_high) → list of (i, j) index pairs
    where `required_hops[i] == h_low` and `required_hops[j] == h_high`.

    Missing buckets in this batch silently produce empty lists for the
    affected boundaries.
    """
    h_np = required_hops.detach().cpu().tolist()
    bucket_to_indices: Dict[int, List[int]] = defaultdict(list)
    for k, h in enumerate(h_np):
        bucket_to_indices[int(h)].append(k)

    # How many pairs each boundary gets (roughly equal).
    n_b = len(boundaries)
    if n_b == 0:
        return {}
    per_boundary_budget = max(1, max_pairs_per_batch // n_b)

    out: Dict[Tuple[int, int], List[Tuple[int, int]]] = {}
    for (h_low, h_high) in boundaries:
        lo = bucket_to_indices.get(h_low, [])
        hi = bucket_to_indices.get(h_high, [])
        if not lo or not hi:
            out[(h_low, h_high)] = []
            continue
        # Maximum number of pairs feasible at this boundary.
        feasible = min(len(lo) * len(hi), per_boundary_budget)
        pairs: List[Tuple[int, int]] = []
        # Deterministic random : we use torch.randint via the supplied
        # generator so this is reproducible across seeds.
        if generator is None:
            generator = torch.Generator(device="cpu")
        i_idx = torch.randint(0, len(lo), (feasible,), generator=generator).tolist()
        j_idx = torch.randint(0, len(hi), (feasible,), generator=generator).tolist()
        for ii, jj in zip(i_idx, j_idx):
            pairs.append((lo[ii], hi[jj]))
        out[(h_low, h_high)] = pairs
    return out


def ordinal_pairwise_loss(
    expected_steps: torch.Tensor,
    required_hops: torch.Tensor,
    *,
    margin: float = 0.15,
    pair_sampling: str = "adjacent_balanced",
    max_pairs_per_batch: int = 512,
    stop_gradient_expected_step: bool = False,
    boundaries: List[Tuple[int, int]] = None,
) -> Tuple[torch.Tensor, Dict[str, int], Dict[str, float]]:
    """Adjacent-balanced pairwise ranking loss on E[step].

    For pairs (i, j) with required_hops_i < required_hops_j, penalty :
        max(0, margin + E_i - E_j).

    Args :
      expected_steps  : (B,) float tensor with requires_grad.
      required_hops   : (B,) int tensor, 5..9 on v1 (NOT differentiable).
      margin          : scalar ≥ 0.
      pair_sampling   : currently only "adjacent_balanced" is supported.
      max_pairs_per_batch : total cap across boundaries.
      stop_gradient_expected_step : if True, the loss is computed on a
        detached copy. Useful for ablations.
      boundaries      : list of (h_low, h_high) pairs. Default :
        ADJACENT_BOUNDARIES.

    Returns :
      (loss_scalar, per_boundary_pair_counts, per_boundary_mean_loss).
      The loss scalar requires gradient when stop_gradient_expected_step
      is False AND at least one valid pair was sampled.
      If no pairs are sampled, returns a zero scalar (no grad).
    """
    if pair_sampling != "adjacent_balanced":
        raise NotImplementedError(
            f"ordinal_pairwise_loss : pair_sampling={pair_sampling!r} not "
            "implemented ; only 'adjacent_balanced' is supported at V2."
        )
    if boundaries is None:
        boundaries = ADJACENT_BOUNDARIES

    if expected_steps.dim() != 1 or required_hops.dim() != 1:
        raise ValueError(
            f"ordinal_pairwise_loss : expected 1-D tensors, got "
            f"expected_steps.shape={tuple(expected_steps.shape)}, "
            f"required_hops.shape={tuple(required_hops.shape)}"
        )
    if expected_steps.size(0) != required_hops.size(0):
        raise ValueError(
            "ordinal_pairwise_loss : batch sizes differ between "
            f"expected_steps ({expected_steps.size(0)}) and "
            f"required_hops ({required_hops.size(0)})"
        )

    E = expected_steps.detach() if stop_gradient_expected_step else expected_steps

    # Sample pairs (CPU-side bookkeeping).
    pairs_by_boundary = _sample_pairs_adjacent_balanced(
        required_hops, boundaries, max_pairs_per_batch,
    )

    total_pairs = sum(len(v) for v in pairs_by_boundary.values())
    per_b_counts: Dict[str, int] = {f"{a}_{b}": len(pairs_by_boundary.get((a, b), []))
                                    for (a, b) in boundaries}
    per_b_loss: Dict[str, float] = {f"{a}_{b}": 0.0 for (a, b) in boundaries}

    if total_pairs == 0:
        # No usable pairs in this batch — return zero scalar (no grad).
        zero = expected_steps.new_zeros(())
        return zero, per_b_counts, per_b_loss

    # Build (lo_idx, hi_idx) tensors on the right device.
    lo_list: List[int] = []
    hi_list: List[int] = []
    boundary_marks: List[Tuple[int, int]] = []
    for (a, b), pairs in pairs_by_boundary.items():
        for (i, j) in pairs:
            lo_list.append(i)
            hi_list.append(j)
            boundary_marks.append((a, b))

    lo_idx = torch.tensor(lo_list, dtype=torch.long, device=E.device)
    hi_idx = torch.tensor(hi_list, dtype=torch.long, device=E.device)

    # margin + E_lo - E_hi  (we want E_lo < E_hi by margin).
    raw = margin + E[lo_idx] - E[hi_idx]
    hinge = torch.clamp(raw, min=0.0)
    loss = hinge.mean()

    # Per-boundary mean (for logging only ; non-grad).
    if boundary_marks:
        with torch.no_grad():
            tag = torch.tensor(
                [(a * 10 + b) for (a, b) in boundary_marks],
                dtype=torch.long, device=E.device,
            )
            for (a, b) in boundaries:
                key = a * 10 + b
                mask = (tag == key)
                if mask.any():
                    per_b_loss[f"{a}_{b}"] = float(hinge[mask].mean().item())
                else:
                    per_b_loss[f"{a}_{b}"] = 0.0

    return loss, per_b_counts, per_b_loss


# ----------------------------------------------------------------------
# 3. Validation-only ordinal metrics + collapse taxonomy
# ----------------------------------------------------------------------

def _spearman(xs: List[float], ys: List[float]) -> float | None:
    if len(xs) < 2 or len(set(xs)) < 2 or len(set(ys)) < 2:
        return None
    def rank(seq):
        pairs = sorted(enumerate(seq), key=lambda p: p[1])
        ranks = [0.0] * len(seq); i = 0
        while i < len(pairs):
            j = i
            while j + 1 < len(pairs) and pairs[j + 1][1] == pairs[i][1]:
                j += 1
            avg = (i + j) / 2 + 1
            for k in range(i, j + 1):
                ranks[pairs[k][0]] = avg
            i = j + 1
        return ranks
    rx = rank(xs); ry = rank(ys)
    mx = statistics.fmean(rx); my = statistics.fmean(ry)
    sx = math.sqrt(sum((x - mx) ** 2 for x in rx))
    sy = math.sqrt(sum((y - my) ** 2 for y in ry))
    if sx == 0 or sy == 0:
        return None
    return sum((x - mx) * (y - my) for x, y in zip(rx, ry)) / (sx * sy)


def _auc_binary(scores: List[float], labels: List[int]) -> float | None:
    n_pos = sum(1 for l in labels if l == 1)
    n_neg = sum(1 for l in labels if l == 0)
    if n_pos == 0 or n_neg == 0:
        return None
    # Rank tally with ties handled (mean rank within tie groups).
    pairs = sorted(enumerate(scores), key=lambda p: p[1])
    ranks = [0.0] * len(scores); i = 0
    while i < len(pairs):
        j = i
        while j + 1 < len(pairs) and pairs[j + 1][1] == pairs[i][1]:
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[pairs[k][0]] = avg
        i = j + 1
    pos_rank_sum = sum(r for r, l in zip(ranks, labels) if l == 1)
    return (pos_rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def _shannon_entropy_bits(values: List[float]) -> float:
    c = Counter(values)
    total = sum(c.values())
    if total == 0:
        return 0.0
    h = 0.0
    for v in c.values():
        p = v / total
        if p > 0:
            h -= p * math.log2(p)
    return h


def compute_gate_metrics(
    expected_step_per_sample: List[float],
    chosen_step_per_sample: List[float],
    required_hops_per_sample: List[int],
    floor_mass_mean: float | None,
    final_mass_mean: float | None,
    floor_mass_max: float | None = None,
    final_mass_max: float | None = None,
) -> Dict[str, float | None]:
    """Compute the H7 gate metric set on a list of per-sample numbers.

    All inputs are plain Python lists / floats (no torch). The caller
    is responsible for converting batched tensors → flat lists.

    Returns a dict with all metrics used by the gate AND the
    `flags` / `notes` taxonomy.
    """
    E = expected_step_per_sample
    chosen = chosen_step_per_sample
    hops = required_hops_per_sample
    n = len(E)
    if n != len(chosen) or n != len(hops):
        raise ValueError(
            f"compute_gate_metrics : length mismatch "
            f"E={len(E)}, chosen={len(chosen)}, hops={len(hops)}"
        )

    # Filter samples with valid hops (allow -1 sentinel produced by the
    # collate when metadata is absent).
    keep = [i for i, h in enumerate(hops) if h >= 0]
    E = [E[i] for i in keep]
    chosen = [chosen[i] for i in keep]
    hops = [hops[i] for i in keep]

    s_all = _spearman(E, hops)
    mask_easy = [i for i, h in enumerate(hops) if h <= 8]
    E_easy = [E[i] for i in mask_easy]
    hops_easy = [hops[i] for i in mask_easy]
    s_easy = _spearman(E_easy, hops_easy) if len(set(hops_easy)) > 1 else None

    auc_by_t = {}
    for t in AUC_THRESHOLDS:
        labels = [1 if h >= t else 0 for h in hops]
        auc_by_t[t] = _auc_binary(E, labels)
    auc9 = auc_by_t[9]
    valid_aucs = [v for v in auc_by_t.values() if v is not None]
    macro_auc = statistics.fmean(valid_aucs) if valid_aucs else None

    bucket_means: Dict[int, float] = {}
    for h in sorted(set(hops)):
        vs = [E[i] for i, hh in enumerate(hops) if hh == h]
        bucket_means[h] = statistics.fmean(vs)
    spread = (max(bucket_means.values()) - min(bucket_means.values())
              if bucket_means else None)

    adjacent_margins: Dict[str, float | None] = {}
    for a, b in ADJACENT_BOUNDARIES:
        if a in bucket_means and b in bucket_means:
            adjacent_margins[f"m_{a}{b}"] = bucket_means[b] - bucket_means[a]
        else:
            adjacent_margins[f"m_{a}{b}"] = None
    valid_margins = [m for m in adjacent_margins.values() if m is not None]
    adj_mean = statistics.fmean(valid_margins) if valid_margins else None
    adj_min = min(valid_margins) if valid_margins else None

    chosen_entropy_bits = _shannon_entropy_bits(chosen)
    dom_mass = (max(Counter(chosen).values()) / max(len(chosen), 1)) if chosen else 0.0

    out: Dict[str, float | None] = {
        "n": n,
        "n_valid": len(E),
        "S_all": s_all,
        "S_easy": s_easy,
        "AUC9": auc9,
        "MACRO_AUC": macro_auc,
        "bucket_means": bucket_means,
        "bucket_spread": spread,
        **adjacent_margins,
        "adjacent_margin_mean": adj_mean,
        "adjacent_margin_min": adj_min,
        "chosen_step_entropy_bits": chosen_entropy_bits,
        "dominant_chosen_step_mass": dom_mass,
        "floor_mass_mean": floor_mass_mean,
        "final_mass_mean": final_mass_mean,
        "floor_mass_max": floor_mass_max,
        "final_mass_max": final_mass_max,
    }
    out["flags"], out["notes"] = classify_collapse_flags(out)
    return out


def classify_collapse_flags(m: Dict[str, float | None]) -> Tuple[List[str], List[str]]:
    """Apply the H7 collapse taxonomy. Returns (flags, notes)."""
    flags: List[str] = []
    notes: List[str] = []

    floor_mean = m.get("floor_mass_mean")
    final_mean = m.get("final_mass_mean")
    floor_max = m.get("floor_mass_max")
    final_max = m.get("final_mass_max")
    if floor_mean is not None and (
        (floor_mean >= 0.5) or (floor_max is not None and floor_max >= 0.8)
    ):
        flags.append("hard_floor")
    if final_mean is not None and (
        (final_mean >= 0.5) or (final_max is not None and final_max >= 0.8)
    ):
        flags.append("hard_final")

    spread = m.get("bucket_spread")
    entropy = m.get("chosen_step_entropy_bits", 0.0)
    dom = m.get("dominant_chosen_step_mass", 0.0)
    soft = False
    if spread is not None and spread <= 0.20 and entropy < 1.0:
        soft = True
        notes.append(
            f"soft_middle_step:spread<=0.20+entropy<1.0 "
            f"(spread={spread:.3f}, entropy={entropy:.3f})"
        )
    if dom >= 0.80:
        soft = True
        notes.append(f"soft_middle_step:dominant_mass>=0.80 (mass={dom:.3f})")
    if soft:
        flags.append("soft_middle_step")

    auc9 = m.get("AUC9")
    s_easy = m.get("S_easy")
    if (auc9 is not None and auc9 >= 0.95
            and s_easy is not None and abs(s_easy) <= 0.10):
        flags.append("binary_h9_shortcut")

    return flags, notes


# ----------------------------------------------------------------------
# 4. Checkpoint-gate eligibility
# ----------------------------------------------------------------------

def gate_eligible(
    metrics: Dict[str, float | None],
    gate_cfg: Dict,
    best_val_acc_so_far: float,
    val_acc: float,
) -> Tuple[bool, List[str]]:
    """Apply the validation-only gate. Returns (eligible, reasons_failed).

    The caller has already established the val_acc envelope :
        eligible_acc = val_acc >= best_val_acc_so_far - min_acc_within_best

    `metrics` is the output of `compute_gate_metrics` for the val
    split at this epoch.
    """
    reasons: List[str] = []
    min_acc_within = float(gate_cfg.get("min_acc_within_best", 0.02))
    if val_acc < best_val_acc_so_far - min_acc_within:
        reasons.append(
            f"val_acc {val_acc:.4f} below best - {min_acc_within} "
            f"(= {best_val_acc_so_far - min_acc_within:.4f})"
        )

    flags = metrics.get("flags", [])
    if gate_cfg.get("reject_hard_collapse", True):
        if "hard_floor" in flags:
            reasons.append("hard_floor flagged")
        if "hard_final" in flags:
            reasons.append("hard_final flagged")
    if gate_cfg.get("reject_soft_middle_step", True):
        if "soft_middle_step" in flags:
            reasons.append("soft_middle_step flagged")

    s_easy = metrics.get("S_easy")
    min_s_easy = float(gate_cfg.get("min_s_easy", 0.15))
    if s_easy is None or s_easy < min_s_easy:
        reasons.append(f"S_easy {s_easy} < {min_s_easy}")

    macro_auc = metrics.get("MACRO_AUC")
    min_macro_auc = float(gate_cfg.get("min_macro_auc", 0.70))
    if macro_auc is None or macro_auc < min_macro_auc:
        reasons.append(f"MACRO_AUC {macro_auc} < {min_macro_auc}")

    adj_mean = metrics.get("adjacent_margin_mean")
    min_adj_mean = float(gate_cfg.get("min_adjacent_margin_mean", 0.0))
    if adj_mean is None or adj_mean <= min_adj_mean:
        reasons.append(
            f"adjacent_margin_mean {adj_mean} <= {min_adj_mean}"
        )

    adj_min = metrics.get("adjacent_margin_min")
    min_adj_min = float(gate_cfg.get("min_adjacent_margin_min", -0.10))
    if adj_min is None or adj_min < min_adj_min:
        reasons.append(
            f"adjacent_margin_min {adj_min} < {min_adj_min}"
        )

    return (len(reasons) == 0), reasons
