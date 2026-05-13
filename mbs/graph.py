from dataclasses import dataclass
from copy import deepcopy
import random

import torch


CELL_TYPES = {
    "ENTITY": 0,
    "ATTRIBUTE": 1,
    "VALUE": 2,
    "SOURCE": 3,
    "TIME": 4,
    "CLAIM": 5,
    "RULE": 6,
    "QUERY": 7,
}

EDGE_TYPES = {
    "HAS_ATTRIBUTE": 0,
    "HAS_VALUE": 1,
    "FROM_SOURCE": 2,
    "AT_TIME": 3,
    "CLAIM_ENTITY": 4,
    "CLAIM_ATTRIBUTE": 5,
    "CLAIM_VALUE": 6,
    "CONTRADICTS": 7,
    "RULE_APPLIES": 8,
    "QUERY_ENTITY": 9,
    "QUERY_ATTRIBUTE": 10,
    "TEMPORAL_NEXT": 11,
    # Used by the depth_controlled_latent_halting_probe (v1) task only.
    # Bidirectional "more reliable than" relation. Lifted from the
    # main-repo dirty mbs/graph.py as part of the H6 reproducibility
    # dependency patch ; see results/.../H6_REPRO_DEPENDENCY_PATCH_PLAN.md.
    "MORE_RELIABLE_THAN": 12,
}

OPERATION_MODES = {
    "PROPAGATE": 0,
    "STABILIZE": 1,
    "REPAIR": 2,
    "RESOLVE_CONFLICT": 3,
}

MODE_FOR_EDGE_TYPE = {
    EDGE_TYPES["HAS_ATTRIBUTE"]: OPERATION_MODES["PROPAGATE"],
    EDGE_TYPES["HAS_VALUE"]: OPERATION_MODES["PROPAGATE"],
    EDGE_TYPES["FROM_SOURCE"]: OPERATION_MODES["STABILIZE"],
    EDGE_TYPES["AT_TIME"]: OPERATION_MODES["STABILIZE"],
    EDGE_TYPES["CLAIM_ENTITY"]: OPERATION_MODES["PROPAGATE"],
    EDGE_TYPES["CLAIM_ATTRIBUTE"]: OPERATION_MODES["PROPAGATE"],
    EDGE_TYPES["CLAIM_VALUE"]: OPERATION_MODES["PROPAGATE"],
    EDGE_TYPES["CONTRADICTS"]: OPERATION_MODES["RESOLVE_CONFLICT"],
    EDGE_TYPES["RULE_APPLIES"]: OPERATION_MODES["REPAIR"],
    EDGE_TYPES["QUERY_ENTITY"]: OPERATION_MODES["PROPAGATE"],
    EDGE_TYPES["QUERY_ATTRIBUTE"]: OPERATION_MODES["PROPAGATE"],
    EDGE_TYPES["TEMPORAL_NEXT"]: OPERATION_MODES["STABILIZE"],
    EDGE_TYPES["MORE_RELIABLE_THAN"]: OPERATION_MODES["RESOLVE_CONFLICT"],
}


@dataclass
class Batch:
    tensors: dict

    def to(self, device):
        return {key: value.to(device) if torch.is_tensor(value) else value for key, value in self.tensors.items()}


def collate_graph_samples(samples, tokenizer, max_text_len=160):
    batch_size = len(samples)
    max_nodes = max(len(sample["belief_cells"]) for sample in samples)
    max_edges = max(max(1, len(sample["edges"])) for sample in samples)
    max_text = max(min(max_text_len, len(tokenizer.encode(sample["text"], max_text_len))) for sample in samples)

    cell_type_ids = torch.zeros(batch_size, max_nodes, dtype=torch.long)
    cell_token_ids = torch.zeros(batch_size, max_nodes, dtype=torch.long)
    node_mask = torch.zeros(batch_size, max_nodes, dtype=torch.bool)
    edge_index = torch.zeros(batch_size, max_edges, 2, dtype=torch.long)
    edge_type_ids = torch.zeros(batch_size, max_edges, dtype=torch.long)
    edge_mask = torch.zeros(batch_size, max_edges, dtype=torch.bool)
    query_node_idx = torch.zeros(batch_size, dtype=torch.long)
    target_value_id = torch.zeros(batch_size, dtype=torch.long)
    conflict_labels = torch.zeros(batch_size, max_nodes, dtype=torch.float)
    conflict_mask = torch.zeros(batch_size, max_nodes, dtype=torch.bool)
    repair_labels = torch.full((batch_size, max_nodes), -100, dtype=torch.long)
    # v1 enriched-halting fields. Default to all-False / -1 sentinel for
    # samples that do not carry per-node lists (back-compat with all
    # non-v1 generators).
    is_query_claim_node = torch.zeros(batch_size, max_nodes, dtype=torch.bool)
    claim_value_ids = torch.full((batch_size, max_nodes), -1, dtype=torch.long)
    mode_labels = torch.full((batch_size, max_edges), -100, dtype=torch.long)
    text_ids = torch.zeros(batch_size, max_text, dtype=torch.long)
    text_mask = torch.zeros(batch_size, max_text, dtype=torch.bool)

    for batch_idx, sample in enumerate(samples):
        cells = sample["belief_cells"]
        edges = sample["edges"]
        node_mask[batch_idx, : len(cells)] = True
        target_value_id[batch_idx] = sample["answer_class"]
        query_node_idx[batch_idx] = sample["metadata"]["query_node_idx"]
        sample_repair_labels = sample.get("repair_labels")
        sample_is_query_claim_node = sample.get("is_query_claim_node")
        sample_claim_value_ids = sample.get("claim_value_ids")
        for node_idx, cell in enumerate(cells):
            cell_type_ids[batch_idx, node_idx] = CELL_TYPES[cell["type"]]
            cell_token_ids[batch_idx, node_idx] = tokenizer.encode_cell(cell["text"])
            if cell["type"] == "CLAIM":
                conflict_mask[batch_idx, node_idx] = True
                conflict_labels[batch_idx, node_idx] = float(cell.get("conflict", 0))
            if sample_repair_labels is not None and node_idx < len(sample_repair_labels):
                repair_labels[batch_idx, node_idx] = int(sample_repair_labels[node_idx])
            if sample_is_query_claim_node is not None and node_idx < len(sample_is_query_claim_node):
                is_query_claim_node[batch_idx, node_idx] = bool(sample_is_query_claim_node[node_idx])
            if sample_claim_value_ids is not None and node_idx < len(sample_claim_value_ids):
                claim_value_ids[batch_idx, node_idx] = int(sample_claim_value_ids[node_idx])

        edge_mask[batch_idx, : len(edges)] = True
        for edge_idx, edge in enumerate(edges):
            edge_index[batch_idx, edge_idx, 0] = edge["src"]
            edge_index[batch_idx, edge_idx, 1] = edge["dst"]
            edge_type = EDGE_TYPES[edge["type"]]
            edge_type_ids[batch_idx, edge_idx] = edge_type
            mode_labels[batch_idx, edge_idx] = MODE_FOR_EDGE_TYPE[edge_type]

        encoded = tokenizer.encode(sample["text"], max_text_len)
        text_ids[batch_idx, : len(encoded)] = torch.tensor(encoded, dtype=torch.long)
        text_mask[batch_idx, : len(encoded)] = True

    return {
        "cell_type_ids": cell_type_ids,
        "cell_token_ids": cell_token_ids,
        "node_mask": node_mask,
        "edge_index": edge_index,
        "edge_type_ids": edge_type_ids,
        "edge_mask": edge_mask,
        "query_node_idx": query_node_idx,
        "target_value_id": target_value_id,
        "conflict_labels": conflict_labels,
        "conflict_mask": conflict_mask,
        "repair_labels": repair_labels,
        "is_query_claim_node": is_query_claim_node,
        "claim_value_ids": claim_value_ids,
        "mode_labels": mode_labels,
        "text_ids": text_ids,
        "text_mask": text_mask,
    }


def randomize_sample_edges(sample, seed=1, mode="preserve_types"):
    randomized = deepcopy(sample)
    rng = random.Random(seed)
    num_nodes = len(randomized["belief_cells"])
    if num_nodes <= 0:
        return randomized

    edges = randomized["edges"]
    if mode == "shuffle_destinations":
        destinations = [edge["dst"] for edge in edges]
        rng.shuffle(destinations)
        randomized["edges"] = [
            {**edge, "dst": int(destinations[idx]), "provenance": "randomized_edges"}
            for idx, edge in enumerate(edges)
        ]
        return randomized

    randomized_edges = []
    for edge in edges:
        if mode in {"preserve_types", "full_random"}:
            src = rng.randrange(num_nodes)
            dst = rng.randrange(num_nodes)
        else:
            raise ValueError(f"unknown edge randomization mode {mode}")
        randomized_edges.append(
            {
                "src": int(src),
                "dst": int(dst),
                "type": edge["type"],
                "provenance": "randomized_edges",
            }
        )
    randomized["edges"] = randomized_edges
    return randomized


def permute_sample_nodes(sample, seed=1):
    permuted = deepcopy(sample)
    rng = random.Random(seed)
    num_nodes = len(permuted["belief_cells"])
    permutation = list(range(num_nodes))
    rng.shuffle(permutation)
    old_to_new = {old_idx: new_idx for new_idx, old_idx in enumerate(permutation)}

    for key in ("belief_cells", "corrupted_cells", "target_cells"):
        if key in permuted and isinstance(permuted[key], list) and len(permuted[key]) == num_nodes:
            permuted[key] = [permuted[key][old_idx] for old_idx in permutation]

    for key in (
        "repair_labels",
        "conflict_labels",
        "is_query_claim_node",
        "claim_value_ids",
    ):
        if key in permuted and isinstance(permuted[key], list) and len(permuted[key]) == num_nodes:
            permuted[key] = [permuted[key][old_idx] for old_idx in permutation]

    permuted["edges"] = [
        {
            **edge,
            "src": int(old_to_new[int(edge["src"])]),
            "dst": int(old_to_new[int(edge["dst"])]),
        }
        for edge in permuted["edges"]
    ]
    query_idx = int(permuted["metadata"]["query_node_idx"])
    permuted["metadata"] = dict(permuted["metadata"])
    permuted["metadata"]["query_node_idx"] = int(old_to_new[query_idx])
    permuted["metadata"]["node_order_permuted"] = True
    return permuted
