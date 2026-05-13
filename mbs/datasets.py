import random
from torch.utils.data import Dataset

from .graph import EDGE_TYPES, permute_sample_nodes, randomize_sample_edges


ENTITIES = [
    "Alice",
    "Bob",
    "Carol",
    "Dave",
    "Eve",
    "Frank",
    "Grace",
    "Heidi",
    "Ivan",
    "Judy",
    "Mallory",
    "Niaj",
    "Olivia",
    "Peggy",
    "Rupert",
    "Sybil",
    "Trent",
    "Uma",
    "Victor",
    "Wendy",
    "Xavier",
    "Yvonne",
    "Zara",
    "Quinn",
    "Ruth",
]
ATTRIBUTES = ["location", "status", "color"]
VALUES = ["Paris", "Berlin", "Rome", "London", "Madrid", "Oslo", "Tokyo", "Cairo"]
SOURCES = ["A", "B", "C", "D", "E", "F", "G", "H"]
RULES = ["latest_wins", "trusted_source_wins", "rollback_source", "source_priority_chain"]
HARD_RULES = [
    "trusted_source_wins",
    "source_priority_chain",
    "rollback_source",
    "earliest_wins",
    "second_latest_wins",
    "least_reliable_source_wins",
]


class BeliefRepairDataset(Dataset):
    def __init__(self, size, split="train", seed=1, hard=False):
        self.size = int(size)
        self.split = split
        self.seed = int(seed)
        generator = generate_belief_repair_hard_sample if hard else generate_belief_repair_sample
        self.samples = [generator(random.Random(self.seed * 100_000 + idx), split) for idx in range(self.size)]

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.samples[idx]


class AdaptiveHaltingProbeDataset(Dataset):
    def __init__(self, size, split="train", seed=1, difficulty_level=None, difficulty_mode="size_controlled"):
        self.size = int(size)
        self.split = split
        self.seed = int(seed)
        self.difficulty_level = difficulty_level
        self.difficulty_mode = difficulty_mode
        self.samples = [
            generate_adaptive_halting_probe_sample(
                random.Random(self.seed * 100_000 + idx),
                split=split,
                difficulty_level=difficulty_level,
                difficulty_mode=difficulty_mode,
            )
            for idx in range(self.size)
        ]

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.samples[idx]


class TransformedDataset(Dataset):
    def __init__(self, dataset, split, config):
        self.dataset = dataset
        self.split = split
        self.config = dict(config)
        self.seed = int(self.config.get("seed", 1))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        transform_seed = self.seed * 1_000_000 + split_offset(self.split) * 10_000 + int(idx)

        if self.config.get("randomize_answer_labels", False):
            rng = random.Random(transform_seed + 17)
            sample = dict_deepcopy(sample)
            original_answer = sample["answer_class"]
            sample["answer_class"] = rng.randrange(len(VALUES))
            sample["metadata"] = dict(sample["metadata"])
            sample["metadata"]["original_answer_class"] = original_answer
            sample["metadata"]["randomized_answer_class"] = sample["answer_class"]
            if self.config.get("randomize_answer_disable_repair", False):
                sample["repair_labels"] = [-100 for _ in sample["belief_cells"]]
            if self.config.get("randomize_answer_disable_conflict", False):
                sample["conflict_labels"] = [0 for _ in sample["belief_cells"]]

        edge_random = (
            self.config.get("randomize_edges_train", False) and self.split == "train"
        ) or (
            self.config.get("randomize_edges_eval", False) and self.split != "train"
        ) or self.config.get("randomize_edges_all", False)
        if edge_random:
            sample = randomize_sample_edges(
                sample,
                seed=transform_seed + 31,
                mode=self.config.get("edge_random_mode", "preserve_types"),
            )

        permute_nodes = (
            self.config.get("permute_node_order_train", False) and self.split == "train"
        ) or (
            self.config.get("permute_node_order_eval", False) and self.split != "train"
        ) or self.config.get("permute_node_order_all", False)
        if permute_nodes:
            sample = permute_sample_nodes(sample, seed=transform_seed + 43)

        return sample


def split_profile(split):
    if split in {"train", "val", "id"}:
        return dict(entities=(2, 5), sources=(1, 3), conflicts=(0, 2), distractors=(1, 4), rules=["latest_wins", "trusted_source_wins"])
    if split == "ood_entity":
        return dict(entities=(10, 25), sources=(1, 3), conflicts=(0, 2), distractors=(10, 28), rules=["latest_wins", "trusted_source_wins"])
    if split == "ood_conflict":
        return dict(entities=(3, 8), sources=(3, 6), conflicts=(3, 8), distractors=(5, 14), rules=["latest_wins", "trusted_source_wins"])
    if split == "ood_rule":
        return dict(entities=(3, 8), sources=(3, 6), conflicts=(1, 4), distractors=(4, 12), rules=RULES)
    if split == "ood_mixed":
        return dict(entities=(10, 25), sources=(4, 8), conflicts=(3, 8), distractors=(15, 36), rules=RULES)
    raise ValueError(f"unknown split {split}")


def hard_split_profile(split):
    if split in {"train", "val", "id"}:
        return dict(entities=(3, 6), sources=(4, 6), conflicts=(2, 4), distractors=(4, 8), rules=HARD_RULES)
    if split == "ood_entity":
        return dict(entities=(10, 25), sources=(4, 6), conflicts=(2, 4), distractors=(12, 28), rules=HARD_RULES)
    if split == "ood_conflict":
        return dict(entities=(5, 10), sources=(5, 8), conflicts=(5, 10), distractors=(8, 18), rules=HARD_RULES)
    if split == "ood_rule":
        return dict(entities=(5, 10), sources=(5, 8), conflicts=(4, 8), distractors=(8, 18), rules=HARD_RULES)
    if split == "ood_mixed":
        return dict(entities=(10, 25), sources=(5, 8), conflicts=(5, 10), distractors=(18, 36), rules=HARD_RULES)
    raise ValueError(f"unknown split {split}")


DIFFICULTY_LEVELS = ["easy", "medium", "hard", "xhard"]


def probe_profile(level, difficulty_mode="size_controlled"):
    if difficulty_mode == "size_controlled":
        base = {
            "easy": dict(entities=10, sources=6, target_claims=1, conflict_pairs=0, distractors=29, rules=["latest_wins"]),
            "medium": dict(entities=10, sources=6, target_claims=3, conflict_pairs=1, distractors=25, rules=["trusted_source_wins", "latest_wins"]),
            "hard": dict(entities=10, sources=6, target_claims=4, conflict_pairs=3, distractors=20, rules=["source_priority_chain", "rollback_source"]),
            "xhard": dict(entities=10, sources=6, target_claims=5, conflict_pairs=5, distractors=15, rules=["source_priority_chain", "rollback_source", "second_latest_wins", "least_reliable_source_wins"]),
        }
    elif difficulty_mode == "size_correlated":
        base = {
            "easy": dict(entities=3, sources=2, target_claims=1, conflict_pairs=0, distractors=2, rules=["latest_wins"]),
            "medium": dict(entities=5, sources=4, target_claims=3, conflict_pairs=1, distractors=8, rules=["trusted_source_wins", "latest_wins"]),
            "hard": dict(entities=8, sources=6, target_claims=4, conflict_pairs=3, distractors=16, rules=["source_priority_chain", "rollback_source"]),
            "xhard": dict(entities=12, sources=8, target_claims=5, conflict_pairs=5, distractors=28, rules=["source_priority_chain", "rollback_source", "second_latest_wins", "least_reliable_source_wins"]),
        }
    else:
        raise ValueError(f"unknown difficulty_mode {difficulty_mode}")
    if level not in base:
        raise ValueError(f"unknown difficulty_level {level}")
    return base[level]


def generate_adaptive_halting_probe_sample(rng, split="train", difficulty_level=None, difficulty_mode="size_controlled"):
    if difficulty_level is None:
        if split in DIFFICULTY_LEVELS:
            difficulty_level = split
        else:
            difficulty_level = rng.choice(DIFFICULTY_LEVELS)
    profile = probe_profile(difficulty_level, difficulty_mode)
    num_entities = profile["entities"]
    num_sources = profile["sources"]
    entities = ENTITIES[:num_entities]
    sources = SOURCES[:num_sources]
    attrs = ATTRIBUTES[:]
    rule = rng.choice(profile["rules"])
    query_entity = rng.choice(entities)
    query_attr = rng.choice(attrs)
    source_priority = sources[:]
    rng.shuffle(source_priority)
    trusted_source = source_priority[0]
    rollback_source = source_priority[-1]
    claims = []
    decisive_claim_indices = []
    time = 1

    def add_claim(claim):
        nonlocal time
        created = make_claim(claim["entity"], claim["attr"], claim["value"], claim["source"], time)
        created["rolled_back"] = bool(claim.get("rolled_back", False))
        claims.append(created)
        time += 1
        return len(claims) - 1

    answer_value, plan = make_probe_query_plan(rng, rule, source_priority, trusted_source, rollback_source, profile["target_claims"])
    for item in plan:
        idx = add_claim({"entity": query_entity, "attr": query_attr, **item})
        if item["value"] == answer_value and not item.get("rolled_back", False):
            decisive_claim_indices.append(idx)

    for _ in range(profile["conflict_pairs"]):
        entity = rng.choice([entity for entity in entities if entity != query_entity] or entities)
        attr = rng.choice(attrs)
        value_a, value_b = rng.sample(VALUES, 2)
        source_a = rng.choice(sources)
        source_b = rng.choice([source for source in sources if source != source_a] or sources)
        add_claim({"entity": entity, "attr": attr, "value": value_a, "source": source_a})
        add_claim({"entity": entity, "attr": attr, "value": value_b, "source": source_b})

    for _ in range(profile["distractors"]):
        entity = rng.choice(entities)
        attr = rng.choice(attrs)
        if entity == query_entity and attr == query_attr:
            entity = rng.choice([candidate for candidate in entities if candidate != query_entity] or entities)
        add_claim({"entity": entity, "attr": attr, "value": rng.choice(VALUES), "source": rng.choice(sources)})

    conflicts = mark_conflicts(claims)
    resolved = resolve_answer(claims, query_entity, query_attr, rule, source_priority, trusted_source, rollback_source)
    if resolved != answer_value:
        raise RuntimeError(f"adaptive_halting_probe construction failed for {difficulty_level}/{rule}: expected {answer_value}, got {resolved}")

    text = render_text(claims, rule, query_entity, query_attr, trusted_source, source_priority, rollback_source)
    cells, edges, query_node_idx, repair_labels = build_cells_and_edges(
        claims, rule, query_entity, query_attr, answer_value, source_priority, trusted_source, rollback_source, conflicts
    )
    decisive_node_indices = find_claim_nodes(cells, claims, decisive_claim_indices)
    distance = shortest_query_evidence_distance(query_node_idx, decisive_node_indices, edges)
    corrupted_cells = [dict(cell) for cell in cells]
    target_cells = [dict(cell) for cell in cells]
    for node_idx, target_value_id in enumerate(repair_labels):
        if target_value_id >= 0:
            target_cells[node_idx]["target_value"] = VALUES[target_value_id]
    for cell in corrupted_cells:
        if cell["type"] == "CLAIM" and rng.random() < 0.12:
            cell["corrupted"] = True

    metadata = {
        "split": split,
        "task": "adaptive_halting_probe",
        "difficulty_level": difficulty_level,
        "difficulty_score": DIFFICULTY_LEVELS.index(difficulty_level) + 1,
        "difficulty_mode": difficulty_mode,
        "num_entities": num_entities,
        "num_sources": num_sources,
        "num_claims": len(claims),
        "num_distractors": profile["distractors"],
        "num_conflicts": sum(1 for claim in claims if claim.get("conflict", 0)),
        "num_rules": 1,
        "has_rollback": bool(rule == "rollback_source" or any(claim.get("rolled_back", False) for claim in claims)),
        "priority_chain_length": len(source_priority) if rule in {"source_priority_chain", "least_reliable_source_wins"} else 0,
        "graph_num_nodes": len(cells),
        "graph_num_edges": len(edges),
        "query_evidence_distance": distance,
        "rule": rule,
        "query_node_idx": query_node_idx,
        "source_priority": source_priority,
        "trusted_source": trusted_source,
        "rollback_source": rollback_source,
        "claims": [dict(claim) for claim in claims],
        "decisive_claim_indices": decisive_claim_indices,
        "decisive_node_indices": decisive_node_indices,
    }
    return {
        "text": text,
        "answer_class": VALUES.index(answer_value),
        "query_entity": query_entity,
        "query_attr": query_attr,
        "target_value": answer_value,
        "belief_cells": cells,
        "edges": edges,
        "corrupted_cells": corrupted_cells,
        "target_cells": target_cells,
        "repair_labels": repair_labels,
        "conflict_labels": [int(cell.get("conflict", 0)) for cell in cells],
        "mode_labels": None,
        "metadata": metadata,
    }


def make_probe_query_plan(rng, rule, source_priority, trusted_source, rollback_source, target_claims):
    if rule in HARD_RULES:
        answer, plan = make_hard_query_plan(rng, rule, source_priority, trusted_source, rollback_source)
    else:
        values = rng.sample(VALUES, max(3, target_claims + 1))
        answer = values[0]
        wrong_values = values[1:]
        if rule == "latest_wins":
            plan = [{"value": wrong_values[idx % len(wrong_values)], "source": source_priority[(idx + 1) % len(source_priority)]} for idx in range(max(0, target_claims - 1))]
            plan.append({"value": answer, "source": source_priority[0]})
        else:
            raise ValueError(f"unsupported probe rule {rule}")
    if rule not in HARD_RULES and len(plan) < target_claims:
        used = {item["value"] for item in plan}
        extras = [value for value in VALUES if value not in used]
        for idx in range(target_claims - len(plan)):
            plan.append({"value": extras[idx % len(extras)], "source": source_priority[(idx + 2) % len(source_priority)]})
    return answer, plan[:target_claims]


def find_claim_nodes(cells, claims, claim_indices):
    wanted = {
        f"{claims[idx]['entity']} {claims[idx]['attr']} {claims[idx]['value']} {claims[idx]['source']} t{claims[idx]['time']}"
        for idx in claim_indices
        if 0 <= idx < len(claims)
    }
    return [idx for idx, cell in enumerate(cells) if cell["type"] == "CLAIM" and cell["text"] in wanted]


def shortest_query_evidence_distance(query_node_idx, decisive_node_indices, edges):
    if not decisive_node_indices:
        return None
    targets = set(decisive_node_indices)
    adjacency = {}
    for edge in edges:
        adjacency.setdefault(edge["src"], []).append(edge["dst"])
    frontier = [(query_node_idx, 0)]
    seen = {query_node_idx}
    for node, dist in frontier:
        if node in targets:
            return dist
        for nxt in adjacency.get(node, []):
            if nxt not in seen:
                seen.add(nxt)
                frontier.append((nxt, dist + 1))
    return None


def generate_belief_repair_sample(rng, split="train"):
    profile = split_profile(split)
    num_entities = rng.randint(*profile["entities"])
    num_sources = rng.randint(*profile["sources"])
    requested_conflicts = rng.randint(*profile["conflicts"])
    num_distractors = rng.randint(*profile["distractors"])
    entities = ENTITIES[:num_entities]
    sources = SOURCES[:num_sources]
    attrs = ATTRIBUTES[: max(1, min(len(ATTRIBUTES), 1 + num_entities % len(ATTRIBUTES)))]
    rule = rng.choice(profile["rules"])
    query_entity = rng.choice(entities)
    query_attr = rng.choice(attrs)
    query_value_options = rng.sample(VALUES, 2)
    source_priority = sources[:]
    rng.shuffle(source_priority)
    trusted_source = source_priority[0]
    rollback_source = source_priority[-1]

    claims = []
    t = 1
    first_source = rng.choice(sources)
    second_source = rng.choice([src for src in sources if src != first_source] or sources)
    claims.append(make_claim(query_entity, query_attr, query_value_options[0], first_source, t))
    t += 1
    claims.append(make_claim(query_entity, query_attr, query_value_options[1], second_source, t))
    t += 1

    for _ in range(max(0, requested_conflicts - 1)):
        entity = rng.choice(entities)
        attr = rng.choice(attrs)
        value_a, value_b = rng.sample(VALUES, 2)
        src_a = rng.choice(sources)
        src_b = rng.choice([src for src in sources if src != src_a] or sources)
        claims.append(make_claim(entity, attr, value_a, src_a, t))
        t += 1
        claims.append(make_claim(entity, attr, value_b, src_b, t))
        t += 1

    for _ in range(num_distractors):
        claims.append(make_claim(rng.choice(entities), rng.choice(attrs), rng.choice(VALUES), rng.choice(sources), t))
        t += 1

    if rule == "rollback_source":
        for claim in claims:
            if claim["source"] == rollback_source and rng.random() < 0.6:
                claim["rolled_back"] = True

    conflicts = mark_conflicts(claims)
    answer_value = resolve_answer(claims, query_entity, query_attr, rule, source_priority, trusted_source, rollback_source)
    if answer_value is None:
        answer_value = claims[0]["value"]

    text = render_text(claims, rule, query_entity, query_attr, trusted_source, source_priority, rollback_source)
    cells, edges, query_node_idx, repair_labels = build_cells_and_edges(
        claims, rule, query_entity, query_attr, answer_value, source_priority, trusted_source, rollback_source, conflicts
    )
    corrupted_cells = [dict(cell) for cell in cells]
    target_cells = [dict(cell) for cell in cells]
    for node_idx, target_value_id in enumerate(repair_labels):
        if target_value_id >= 0:
            target_cells[node_idx]["target_value"] = VALUES[target_value_id]
    for cell in corrupted_cells:
        if cell["type"] == "CLAIM" and rng.random() < 0.1:
            cell["corrupted"] = True

    return {
        "text": text,
        "answer_class": VALUES.index(answer_value),
        "query_entity": query_entity,
        "query_attr": query_attr,
        "target_value": answer_value,
        "belief_cells": cells,
        "edges": edges,
        "corrupted_cells": corrupted_cells,
        "target_cells": target_cells,
        "repair_labels": repair_labels,
        "conflict_labels": [int(cell.get("conflict", 0)) for cell in cells],
        "mode_labels": None,
        "metadata": {
            "split": split,
            "num_entities": num_entities,
            "num_sources": num_sources,
            "num_conflicts": sum(1 for claim in claims if claim.get("conflict", 0)),
            "rule": rule,
            "query_node_idx": query_node_idx,
            "source_priority": source_priority,
            "trusted_source": trusted_source,
            "rollback_source": rollback_source,
            "claims": [dict(claim) for claim in claims],
        },
    }


def generate_belief_repair_hard_sample(rng, split="train"):
    profile = hard_split_profile(split)
    num_entities = rng.randint(*profile["entities"])
    num_sources = rng.randint(*profile["sources"])
    requested_conflicts = rng.randint(*profile["conflicts"])
    num_distractors = rng.randint(*profile["distractors"])
    entities = ENTITIES[:num_entities]
    sources = SOURCES[:num_sources]
    attrs = ATTRIBUTES[:]
    rule = rng.choice(profile["rules"])
    query_entity = rng.choice(entities)
    query_attr = rng.choice(attrs)
    source_priority = sources[:]
    rng.shuffle(source_priority)
    trusted_source = source_priority[0]
    rollback_source = source_priority[-1]

    claims = []
    t = 1

    def add_query_claim(value, source, rolled_back=False):
        nonlocal t
        claim = make_claim(query_entity, query_attr, value, source, t)
        claim["rolled_back"] = bool(rolled_back)
        claims.append(claim)
        t += 1
        return claim

    answer_value, query_plan = make_hard_query_plan(rng, rule, source_priority, trusted_source, rollback_source)
    for item in query_plan:
        add_query_claim(item["value"], item["source"], item.get("rolled_back", False))

    for _ in range(max(0, requested_conflicts)):
        entity = rng.choice([entity for entity in entities if entity != query_entity] or entities)
        attr = rng.choice(attrs)
        value_a, value_b = rng.sample(VALUES, 2)
        src_a = rng.choice(sources)
        src_b = rng.choice([src for src in sources if src != src_a] or sources)
        claims.append(make_claim(entity, attr, value_a, src_a, t))
        t += 1
        claims.append(make_claim(entity, attr, value_b, src_b, t))
        t += 1

    for _ in range(num_distractors):
        entity = rng.choice(entities)
        attr = rng.choice(attrs)
        if entity == query_entity and attr == query_attr:
            entity = rng.choice([candidate for candidate in entities if candidate != query_entity] or entities)
        claims.append(make_claim(entity, attr, rng.choice(VALUES), rng.choice(sources), t))
        t += 1

    conflicts = mark_conflicts(claims)
    resolved = resolve_answer(claims, query_entity, query_attr, rule, source_priority, trusted_source, rollback_source)
    if resolved != answer_value:
        raise RuntimeError(f"hard sample construction failed for rule {rule}: expected {answer_value}, got {resolved}")

    text = render_text(claims, rule, query_entity, query_attr, trusted_source, source_priority, rollback_source)
    cells, edges, query_node_idx, repair_labels = build_cells_and_edges(
        claims, rule, query_entity, query_attr, answer_value, source_priority, trusted_source, rollback_source, conflicts
    )
    corrupted_cells = [dict(cell) for cell in cells]
    target_cells = [dict(cell) for cell in cells]
    for node_idx, target_value_id in enumerate(repair_labels):
        if target_value_id >= 0:
            target_cells[node_idx]["target_value"] = VALUES[target_value_id]
    for cell in corrupted_cells:
        if cell["type"] == "CLAIM" and rng.random() < 0.12:
            cell["corrupted"] = True

    return {
        "text": text,
        "answer_class": VALUES.index(answer_value),
        "query_entity": query_entity,
        "query_attr": query_attr,
        "target_value": answer_value,
        "belief_cells": cells,
        "edges": edges,
        "corrupted_cells": corrupted_cells,
        "target_cells": target_cells,
        "repair_labels": repair_labels,
        "conflict_labels": [int(cell.get("conflict", 0)) for cell in cells],
        "mode_labels": None,
        "metadata": {
            "split": split,
            "hard": True,
            "num_entities": num_entities,
            "num_sources": num_sources,
            "num_conflicts": sum(1 for claim in claims if claim.get("conflict", 0)),
            "rule": rule,
            "query_node_idx": query_node_idx,
            "source_priority": source_priority,
            "trusted_source": trusted_source,
            "rollback_source": rollback_source,
            "claims": [dict(claim) for claim in claims],
        },
    }


def make_hard_query_plan(rng, rule, source_priority, trusted_source, rollback_source):
    values = rng.sample(VALUES, 5)
    answer = values[0]
    wrong_values = values[1:]
    top_source = source_priority[0]
    second_source = source_priority[1 % len(source_priority)]
    middle_source = source_priority[len(source_priority) // 2]
    least_source = source_priority[-1]

    if rule == "trusted_source_wins":
        return answer, [
            {"value": wrong_values[0], "source": second_source},
            {"value": answer, "source": trusted_source},
            {"value": wrong_values[1], "source": middle_source},
            {"value": wrong_values[2], "source": least_source},
        ]

    if rule == "source_priority_chain":
        return answer, [
            {"value": wrong_values[0], "source": second_source},
            {"value": answer, "source": top_source},
            {"value": wrong_values[1], "source": middle_source},
            {"value": wrong_values[2], "source": least_source},
        ]

    if rule == "rollback_source":
        return answer, [
            {"value": wrong_values[0], "source": second_source},
            {"value": wrong_values[1], "source": middle_source},
            {"value": answer, "source": top_source},
            {"value": wrong_values[2], "source": rollback_source, "rolled_back": True},
        ]

    if rule == "earliest_wins":
        return answer, [
            {"value": answer, "source": second_source},
            {"value": wrong_values[0], "source": top_source},
            {"value": wrong_values[1], "source": middle_source},
            {"value": wrong_values[2], "source": least_source},
        ]

    if rule == "second_latest_wins":
        return answer, [
            {"value": wrong_values[0], "source": second_source},
            {"value": wrong_values[1], "source": top_source},
            {"value": answer, "source": middle_source},
            {"value": wrong_values[2], "source": least_source},
        ]

    if rule == "least_reliable_source_wins":
        return answer, [
            {"value": wrong_values[0], "source": top_source},
            {"value": answer, "source": least_source},
            {"value": wrong_values[1], "source": second_source},
            {"value": wrong_values[2], "source": top_source},
        ]

    raise ValueError(f"unknown hard rule {rule}")


def split_offset(split):
    return {
        "train": 1,
        "val": 2,
        "id": 2,
        "ood_entity": 3,
        "ood_conflict": 4,
        "ood_rule": 5,
        "ood_mixed": 6,
        "easy": 7,
        "medium": 8,
        "hard": 9,
        "xhard": 10,
        "mixed": 11,
    }.get(split, 99)


def dict_deepcopy(value):
    if isinstance(value, dict):
        return {key: dict_deepcopy(item) for key, item in value.items()}
    if isinstance(value, list):
        return [dict_deepcopy(item) for item in value]
    return value


def make_claim(entity, attr, value, source, time):
    return {
        "entity": entity,
        "attr": attr,
        "value": value,
        "source": source,
        "time": time,
        "rolled_back": False,
        "conflict": 0,
    }


def mark_conflicts(claims):
    conflicts = set()
    for idx, left in enumerate(claims):
        for jdx, right in enumerate(claims):
            if idx >= jdx:
                continue
            if left["entity"] == right["entity"] and left["attr"] == right["attr"] and left["value"] != right["value"]:
                conflicts.add(idx)
                conflicts.add(jdx)
    for idx in conflicts:
        claims[idx]["conflict"] = 1
    return conflicts


def resolve_answer(claims, entity, attr, rule, source_priority, trusted_source, rollback_source):
    relevant = [claim for claim in claims if claim["entity"] == entity and claim["attr"] == attr]
    if rule == "rollback_source":
        relevant = [claim for claim in relevant if not (claim["source"] == rollback_source and claim.get("rolled_back", False))]
    if not relevant:
        return None
    if rule == "trusted_source_wins":
        trusted = [claim for claim in relevant if claim["source"] == trusted_source]
        if trusted:
            return max(trusted, key=lambda item: item["time"])["value"]
    if rule == "source_priority_chain":
        rank = {source: idx for idx, source in enumerate(source_priority)}
        return min(relevant, key=lambda item: (rank.get(item["source"], 999), -item["time"]))["value"]
    if rule == "earliest_wins":
        return min(relevant, key=lambda item: item["time"])["value"]
    if rule == "second_latest_wins":
        ordered = sorted(relevant, key=lambda item: item["time"], reverse=True)
        return ordered[1 if len(ordered) > 1 else 0]["value"]
    if rule == "least_reliable_source_wins":
        rank = {source: idx for idx, source in enumerate(source_priority)}
        return max(relevant, key=lambda item: (rank.get(item["source"], -1), item["time"]))["value"]
    return max(relevant, key=lambda item: item["time"])["value"]


def render_text(claims, rule, query_entity, query_attr, trusted_source, source_priority, rollback_source):
    parts = []
    for claim in claims:
        parts.append(
            f"source {claim['source']} says {claim['entity']} {claim['attr']} {claim['value']} at t{claim['time']}."
        )
        if claim.get("rolled_back", False):
            parts.append(f"rollback source {claim['source']} at t{claim['time']}.")
    if rule == "latest_wins":
        parts.append("rule latest wins.")
    elif rule == "trusted_source_wins":
        parts.append(f"source {trusted_source} is more reliable than source {source_priority[-1]}.")
    elif rule == "rollback_source":
        parts.append(f"rollback source {rollback_source}.")
    elif rule == "source_priority_chain":
        parts.append("source priority chain " + " ".join(source_priority) + ".")
    elif rule == "earliest_wins":
        parts.append("rule earliest wins.")
    elif rule == "second_latest_wins":
        parts.append("rule second latest wins.")
    elif rule == "least_reliable_source_wins":
        parts.append("rule least reliable source wins.")
    else:
        parts.append(f"rule {rule}.")
    parts.append(f"query {query_entity} {query_attr}.")
    return " ".join(parts)


def build_cells_and_edges(claims, rule, query_entity, query_attr, answer_value, source_priority, trusted_source, rollback_source, conflicts):
    cells = []
    index = {}
    repair_label_by_node = {}

    def add_cell(cell_type, text, **extra):
        key = (cell_type, text)
        if cell_type != "CLAIM" and key in index:
            return index[key]
        idx = len(cells)
        index[key] = idx
        cells.append({"type": cell_type, "text": text, **extra})
        return idx

    entity_nodes = {entity: add_cell("ENTITY", entity) for entity in sorted({claim["entity"] for claim in claims} | {query_entity})}
    attr_nodes = {attr: add_cell("ATTRIBUTE", attr) for attr in sorted({claim["attr"] for claim in claims} | {query_attr})}
    value_nodes = {value: add_cell("VALUE", value) for value in sorted({claim["value"] for claim in claims} | {answer_value})}
    source_nodes = {source: add_cell("SOURCE", source) for source in sorted({claim["source"] for claim in claims} | set(source_priority))}
    time_nodes = {claim["time"]: add_cell("TIME", f"t{claim['time']}") for claim in claims}
    rule_node = add_cell("RULE", rule)
    query_node = add_cell("QUERY", f"query {query_entity} {query_attr}")
    repair_label_by_node[query_node] = VALUES.index(answer_value)

    edges = []

    def add_edge(src, dst, edge_type, bidirectional=True, provenance="observable_facts"):
        edges.append({"src": int(src), "dst": int(dst), "type": edge_type, "provenance": provenance})
        if bidirectional:
            edges.append({"src": int(dst), "dst": int(src), "type": edge_type, "provenance": provenance})

    for entity, ent_idx in entity_nodes.items():
        for attr, attr_idx in attr_nodes.items():
            add_edge(ent_idx, attr_idx, "HAS_ATTRIBUTE")
    for attr_idx in attr_nodes.values():
        for value_idx in value_nodes.values():
            add_edge(attr_idx, value_idx, "HAS_VALUE")

    claim_nodes = []
    for claim_idx, claim in enumerate(claims):
        claim_node = add_cell(
            "CLAIM",
            f"{claim['entity']} {claim['attr']} {claim['value']} {claim['source']} t{claim['time']}",
            conflict=int(claim_idx in conflicts),
        )
        repair_label_by_node[claim_node] = (
            VALUES.index(answer_value)
            if claim["entity"] == query_entity and claim["attr"] == query_attr
            else VALUES.index(claim["value"])
        )
        claim_nodes.append(claim_node)
        add_edge(claim_node, entity_nodes[claim["entity"]], "CLAIM_ENTITY")
        add_edge(claim_node, attr_nodes[claim["attr"]], "CLAIM_ATTRIBUTE")
        add_edge(claim_node, value_nodes[claim["value"]], "CLAIM_VALUE")
        add_edge(claim_node, source_nodes[claim["source"]], "FROM_SOURCE")
        add_edge(claim_node, time_nodes[claim["time"]], "AT_TIME")
        add_edge(rule_node, claim_node, "RULE_APPLIES")

    sorted_times = sorted(time_nodes)
    for left, right in zip(sorted_times, sorted_times[1:]):
        add_edge(time_nodes[left], time_nodes[right], "TEMPORAL_NEXT")

    for idx, left in enumerate(claims):
        for jdx, right in enumerate(claims):
            if idx >= jdx:
                continue
            if left["entity"] == right["entity"] and left["attr"] == right["attr"] and left["value"] != right["value"]:
                add_edge(claim_nodes[idx], claim_nodes[jdx], "CONTRADICTS")
                add_edge(claim_nodes[jdx], claim_nodes[idx], "CONTRADICTS")

    add_edge(query_node, entity_nodes[query_entity], "QUERY_ENTITY", provenance="observable_query")
    add_edge(query_node, attr_nodes[query_attr], "QUERY_ATTRIBUTE", provenance="observable_query")
    repair_labels = [repair_label_by_node.get(node_idx, -100) for node_idx in range(len(cells))]
    return cells, edges, query_node, repair_labels


def build_belief_repair_datasets(config, tokenizer=None):
    # H6 reproducibility dependency : the v1 task generator was missing
    # from the dfb99b0 commit but the committed model + audit + CSV
    # artefacts all depend on it. See
    # results/.../H6_REPRO_DEPENDENCY_PATCH_PLAN.md.
    if config.get("task") == "depth_controlled_latent_halting_probe":
        return build_depth_controlled_latent_halting_datasets(config, tokenizer)
    if config.get("task") in {"adaptive_halting_probe", "belief_repair_difficulty_ladder"}:
        return build_adaptive_halting_probe_datasets(config, tokenizer)
    seed = int(config.get("seed", 1))
    hard = config.get("task") == "belief_repair_hard" or bool(config.get("hard_dataset", False))
    datasets = {
        "train": BeliefRepairDataset(config.get("train_size", 5000), "train", seed, hard=hard),
        "val": BeliefRepairDataset(config.get("val_size", 512), "val", seed + 1, hard=hard),
        "ood_entity": BeliefRepairDataset(config.get("ood_size", 512), "ood_entity", seed + 2, hard=hard),
        "ood_conflict": BeliefRepairDataset(config.get("ood_size", 512), "ood_conflict", seed + 3, hard=hard),
        "ood_rule": BeliefRepairDataset(config.get("ood_size", 512), "ood_rule", seed + 4, hard=hard),
        "ood_mixed": BeliefRepairDataset(config.get("ood_size", 512), "ood_mixed", seed + 5, hard=hard),
    }
    transform_keys = {
        "randomize_answer_labels",
        "randomize_answer_disable_repair",
        "randomize_answer_disable_conflict",
        "randomize_edges_train",
        "randomize_edges_eval",
        "randomize_edges_all",
        "edge_random_mode",
        "permute_node_order_train",
        "permute_node_order_eval",
        "permute_node_order_all",
    }
    if any(key in config for key in transform_keys):
        datasets = {split: TransformedDataset(dataset, split, config) for split, dataset in datasets.items()}
    return datasets


def build_adaptive_halting_probe_datasets(config, tokenizer=None):
    seed = int(config.get("seed", 1))
    difficulty_mode = config.get("difficulty_mode", "size_controlled")
    test_size = int(config.get("test_size_per_difficulty", config.get("ood_size", 512)))
    datasets = {
        "train": AdaptiveHaltingProbeDataset(config.get("train_size", 5000), "train", seed, difficulty_mode=difficulty_mode),
        "val": AdaptiveHaltingProbeDataset(config.get("val_size", 512), "val", seed + 1, difficulty_mode=difficulty_mode),
        "easy": AdaptiveHaltingProbeDataset(test_size, "easy", seed + 2, difficulty_level="easy", difficulty_mode=difficulty_mode),
        "medium": AdaptiveHaltingProbeDataset(test_size, "medium", seed + 3, difficulty_level="medium", difficulty_mode=difficulty_mode),
        "hard": AdaptiveHaltingProbeDataset(test_size, "hard", seed + 4, difficulty_level="hard", difficulty_mode=difficulty_mode),
        "xhard": AdaptiveHaltingProbeDataset(test_size, "xhard", seed + 5, difficulty_level="xhard", difficulty_mode=difficulty_mode),
        "mixed": AdaptiveHaltingProbeDataset(test_size, "mixed", seed + 6, difficulty_mode=difficulty_mode),
    }
    transform_keys = {
        "randomize_answer_labels",
        "randomize_answer_disable_repair",
        "randomize_answer_disable_conflict",
        "randomize_edges_train",
        "randomize_edges_eval",
        "randomize_edges_all",
        "edge_random_mode",
        "permute_node_order_train",
        "permute_node_order_eval",
        "permute_node_order_all",
    }
    if any(key in config for key in transform_keys):
        datasets = {split: TransformedDataset(dataset, split, config) for split, dataset in datasets.items()}
    return datasets


# =============================================================================
# v1 depth-controlled latent halting probe generator
# =============================================================================
# Lifted from main repo dirty mbs/datasets.py:1182..1520 as part of the
# H6 reproducibility dependency patch. The committed dfb99b0 state did
# not include this generator, even though the committed H6 audit script,
# CSVs, configs, and model expect samples produced by it. See
# results/claim_strengthening/h7_ordinal_halting/H6_REPRO_DEPENDENCY_PATCH_PLAN.md
# for the scope analysis.
#
# This is NOT an H7 method change. v2 / v3 / v3.1 generators are NOT
# included here (out of scope).


DEPTH_PROBE_RULE_NAME = "trusted_chain_top_wins"

# Depth buckets default to {2,4,6,8} so the trust chain fits inside the
# existing 8-element SOURCE pool (A..H) without expanding the tokenizer
# beyond the alphabet it already covers. Users can override with config
# `depth_buckets` plus a matching `k_max` if they extend the pool.
DEFAULT_DEPTH_BUCKETS = [2, 4, 6, 8]
DEFAULT_K_MAX = 8


def _build_depth_probe_sample(rng, depths, k_max, value_pool, source_pool,
                              entity_pool, attribute_pool,
                              randomize_query_claim_order=True):
    """Build a single sample for the depth-controlled probe.

    Args:
        rng: random.Random instance (sample-deterministic).
        depths: list of allowed depth buckets.
        k_max: total chain length (constant per dataset).
        value_pool, source_pool, entity_pool, attribute_pool: token pools.
        randomize_query_claim_order: shuffle the 4 CLAIM cells in the cells list.
    """
    if k_max < max(depths):
        raise ValueError("k_max must be >= max(depths)")
    if len(value_pool) < 4:
        raise ValueError("value_pool must have >= 4 entries")
    if len(source_pool) < k_max:
        raise ValueError(f"source_pool must have >= k_max={k_max} entries")

    D = rng.choice(depths)
    # 4 distinct ranks in {1..D} with min=1, max=D
    if D >= 4:
        intermediates = rng.sample(range(2, D), 2)  # 2 picks in [2, D-1]
    elif D == 2:
        # only ranks 1 and 2 available — degrade to 2 candidates? No, we still
        # need 4 distinct CLAIMs, so use the chain up to D=2 plus pad with
        # later ranks that DON'T determine the winner. We achieve depth=2 by
        # placing the winner at rank 1 and runner-up at rank 2; the other 2
        # CLAIMs go to higher ranks (which the model still has to traverse to
        # rule out, but the winning comparison only requires depth 2).
        # For simplicity, we drop D=2 from depths if k_max=12 and we want at
        # least 4 candidates within {1..D}. Here we sample 2 ranks beyond D.
        higher = rng.sample(range(3, k_max + 1), 2)
        intermediates = sorted(higher)  # treated as candidates whose ranks > D
    else:
        # D = 3: ranks 2 and any of {3..K_MAX}
        higher_pool = list(range(2, k_max + 1))
        higher_pool.remove(D)  # ensure D is reserved for runner-up
        higher = rng.sample(higher_pool, 2)
        intermediates = sorted(higher)
    candidate_ranks = [1, D] + list(intermediates)
    # Re-sort and dedupe (just in case for D=2)
    candidate_ranks = sorted(set(candidate_ranks))
    while len(candidate_ranks) < 4:
        # fall back: sample more ranks from {2..k_max} that aren't already used
        avail = [r for r in range(2, k_max + 1) if r not in candidate_ranks]
        candidate_ranks.append(rng.choice(avail))
        candidate_ranks = sorted(set(candidate_ranks))
    # Trim if accidentally longer than 4 (shouldn't happen for D >= 4 normally)
    candidate_ranks = candidate_ranks[:4]

    # 4 distinct values
    values = rng.sample(value_pool, 4)
    answer_value = values[0]  # the winner's value (rank=1 candidate)
    # Pair (rank, value): rank=1 -> values[0], the other ranks -> values[1..3]
    rank_to_value = {candidate_ranks[0]: values[0]}
    for i, r in enumerate(candidate_ranks[1:], start=1):
        rank_to_value[r] = values[i]

    # Build the chain by assigning K_MAX source IDs to ranks 1..K_MAX
    # uniformly at random, so source identity is decoupled from rank.
    sources_in_chain = rng.sample(source_pool, k_max)
    # rank -> source name
    rank_to_source = {rank: sources_in_chain[rank - 1] for rank in range(1, k_max + 1)}
    # source -> rank (for the gold-oracle audit only)
    source_to_rank = {s: r for r, s in rank_to_source.items()}

    # Choose a query (entity, attribute) — any will do since the rule
    # ignores them; values differ across CLAIMs.
    query_entity = rng.choice(entity_pool)
    query_attr = rng.choice(attribute_pool)

    # Build cells in a deterministic order:
    #   ENTITY (1), ATTRIBUTE (1), VALUE (4), SOURCE (k_max), RULE (1),
    #   QUERY (1), CLAIM (4)
    cells = []

    def add_cell(cell_type, text, **extra):
        idx = len(cells)
        cells.append({"type": cell_type, "text": text, **extra})
        return idx

    entity_idx = add_cell("ENTITY", query_entity)
    attribute_idx = add_cell("ATTRIBUTE", query_attr)
    value_idx_by_value = {v: add_cell("VALUE", v) for v in values}
    source_idx_by_name = {s: add_cell("SOURCE", s) for s in sources_in_chain}
    rule_idx = add_cell("RULE", DEPTH_PROBE_RULE_NAME)
    query_node_idx = add_cell("QUERY", f"query {query_entity} {query_attr}")

    # CLAIM cells — text mirrors the legacy generator format so the
    # tokenizer encodes them unambiguously.
    claim_specs = []  # list of (rank, value, source_name)
    for rank in candidate_ranks:
        claim_specs.append((rank, rank_to_value[rank], rank_to_source[rank]))
    if randomize_query_claim_order:
        rng.shuffle(claim_specs)

    claim_node_idx_list = []
    for rank, value, source_name in claim_specs:
        text = f"{query_entity} {query_attr} {value} {source_name} t1"
        idx = add_cell("CLAIM", text, conflict=0)
        claim_node_idx_list.append((idx, rank, value, source_name))

    # Edges
    edges = []

    def add_edge(src, dst, edge_type, bidirectional=True):
        edges.append({"src": int(src), "dst": int(dst), "type": edge_type, "provenance": "depth_probe"})
        if bidirectional:
            edges.append({"src": int(dst), "dst": int(src), "type": edge_type, "provenance": "depth_probe"})

    # Schema edges
    add_edge(entity_idx, attribute_idx, "HAS_ATTRIBUTE")
    for v_idx in value_idx_by_value.values():
        add_edge(attribute_idx, v_idx, "HAS_VALUE")

    # CLAIM-side edges
    for claim_node, rank, value, source_name in claim_node_idx_list:
        add_edge(claim_node, entity_idx, "CLAIM_ENTITY")
        add_edge(claim_node, attribute_idx, "CLAIM_ATTRIBUTE")
        add_edge(claim_node, value_idx_by_value[value], "CLAIM_VALUE")
        add_edge(claim_node, source_idx_by_name[source_name], "FROM_SOURCE")
        add_edge(rule_idx, claim_node, "RULE_APPLIES")

    # Trust chain edges (the depth-controlled core)
    for r in range(1, k_max):
        more_idx = source_idx_by_name[rank_to_source[r]]      # higher rank (more reliable)
        less_idx = source_idx_by_name[rank_to_source[r + 1]]  # lower rank
        add_edge(more_idx, less_idx, "MORE_RELIABLE_THAN", bidirectional=True)

    # Query edges
    add_edge(query_node_idx, entity_idx, "QUERY_ENTITY")
    add_edge(query_node_idx, attribute_idx, "QUERY_ATTRIBUTE")

    # Answer label
    answer_class = VALUES.index(answer_value)

    # Per-node fields
    n_nodes = len(cells)
    is_query_claim_node = [False] * n_nodes
    is_winner_query_claim_node = [False] * n_nodes
    claim_source_is_trusted = [False] * n_nodes
    claim_is_rolled_back = [False] * n_nodes
    claim_value_ids = [-1] * n_nodes
    repair_labels = [-100] * n_nodes
    conflict_labels = [0] * n_nodes
    for claim_node, rank, value, source_name in claim_node_idx_list:
        is_query_claim_node[claim_node] = True
        if rank == 1:
            is_winner_query_claim_node[claim_node] = True
        if rank == 1:
            claim_source_is_trusted[claim_node] = True
        # claim_value_ids = the asserted value (NOT the answer)
        claim_value_ids[claim_node] = VALUES.index(value)
        # repair_label is left at -100 — the latent task does NOT use it.
        # We still set query CLAIMs to the answer for legacy compatibility
        # IF and only if a downstream task switches repair_loss_mode away
        # from "none" — which our config forbids.
        repair_labels[claim_node] = answer_class
    repair_labels[query_node_idx] = answer_class

    text = " ".join(c["text"] for c in cells if c["type"] == "CLAIM")

    metadata = {
        "task": "depth_controlled_latent_halting_probe",
        "rule": DEPTH_PROBE_RULE_NAME,
        "oracle_depth": int(D),
        "depth_bucket": int(D),
        "chain_length": int(k_max - 1),
        "k_max": int(k_max),
        "graph_num_nodes": n_nodes,
        "graph_num_edges": len(edges),
        "winner_rank": 1,
        "runner_up_rank": int(D),
        "candidate_ranks_used": list(candidate_ranks),
        "rank_to_source": {int(k): str(v) for k, v in rank_to_source.items()},
        "source_to_rank": {str(k): int(v) for k, v in source_to_rank.items()},
        "query_entity": query_entity,
        "query_attr": query_attr,
        "query_node_idx": query_node_idx,
        "claims": [
            {"entity": query_entity, "attr": query_attr, "value": value,
             "source": source_name, "time": 1, "rank": rank,
             "is_winner_audit": (rank == 1)}
            for _, rank, value, source_name in claim_node_idx_list
        ],
    }
    return {
        "text": text,
        "answer_class": answer_class,
        "query_entity": query_entity,
        "query_attr": query_attr,
        "target_value": answer_value,
        "belief_cells": cells,
        "edges": edges,
        "corrupted_cells": [dict(c) for c in cells],
        "target_cells": [dict(c) for c in cells],
        "repair_labels": repair_labels,
        "is_query_claim_node": is_query_claim_node,
        "is_winner_query_claim_node": is_winner_query_claim_node,
        "claim_source_is_trusted": claim_source_is_trusted,
        "claim_is_rolled_back": claim_is_rolled_back,
        "claim_value_ids": claim_value_ids,
        "conflict_labels": conflict_labels,
        "mode_labels": None,
        "metadata": metadata,
    }


def _depth_probe_split_profile(split):
    """Per-split allowed depth-bucket lists. We keep depths uniform across all
    splits so OOD differences come from the standard split shifts (entity /
    rule pools), not from depth distribution shifts.

    For now all splits use the same bucket list; future variants could OOD-shift
    by holding out depths.
    """
    return DEFAULT_DEPTH_BUCKETS


class DepthControlledLatentHaltingProbeDataset(Dataset):
    def __init__(self, size, split="train", seed=1, depths=None, k_max=DEFAULT_K_MAX,
                 randomize_query_claim_order=True):
        self.size = int(size)
        self.split = split
        self.seed = int(seed)
        self.depths = list(depths) if depths is not None else _depth_probe_split_profile(split)
        self.k_max = int(k_max)
        self.randomize_query_claim_order = bool(randomize_query_claim_order)
        # Pools — the SOURCE pool stays at the existing 8-element alphabet so
        # the existing tokenizer covers it without modification. k_max <= 8.
        if self.k_max > len(SOURCES):
            raise ValueError(
                f"k_max={self.k_max} > len(SOURCES)={len(SOURCES)}. "
                "Extend the tokenizer pool before raising k_max."
            )
        self.source_pool = SOURCES[: self.k_max]
        self.entity_pool = ENTITIES[:]
        self.attribute_pool = ATTRIBUTES[:]
        self.value_pool = VALUES[:]
        self.samples = [
            _build_depth_probe_sample(
                random.Random(self.seed * 100_000 + idx),
                self.depths, self.k_max,
                self.value_pool, self.source_pool,
                self.entity_pool, self.attribute_pool,
                randomize_query_claim_order=self.randomize_query_claim_order,
            )
            for idx in range(self.size)
        ]

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.samples[idx]


def build_depth_controlled_latent_halting_datasets(config, tokenizer=None):
    seed = int(config.get("seed", 1))
    depths = list(config.get("depth_buckets", DEFAULT_DEPTH_BUCKETS))
    k_max = int(config.get("k_max", DEFAULT_K_MAX))
    randomize_qc = bool(config.get("randomize_query_claim_order", True))
    datasets = {
        "train": DepthControlledLatentHaltingProbeDataset(
            config.get("train_size", 5000), "train", seed,
            depths=depths, k_max=k_max,
            randomize_query_claim_order=randomize_qc,
        ),
        "val": DepthControlledLatentHaltingProbeDataset(
            config.get("val_size", 512), "val", seed + 1,
            depths=depths, k_max=k_max,
            randomize_query_claim_order=randomize_qc,
        ),
        # OOD splits are currently same-distribution; differentiated only by seed.
        # OOD-shifted depth distributions can be added later via depth_buckets_ood_*.
        "ood_entity": DepthControlledLatentHaltingProbeDataset(
            config.get("ood_size", 512), "ood_entity", seed + 2,
            depths=depths, k_max=k_max,
            randomize_query_claim_order=randomize_qc,
        ),
        "ood_conflict": DepthControlledLatentHaltingProbeDataset(
            config.get("ood_size", 512), "ood_conflict", seed + 3,
            depths=depths, k_max=k_max,
            randomize_query_claim_order=randomize_qc,
        ),
        "ood_rule": DepthControlledLatentHaltingProbeDataset(
            config.get("ood_size", 512), "ood_rule", seed + 4,
            depths=depths, k_max=k_max,
            randomize_query_claim_order=randomize_qc,
        ),
        "ood_mixed": DepthControlledLatentHaltingProbeDataset(
            config.get("ood_size", 512), "ood_mixed", seed + 5,
            depths=depths, k_max=k_max,
            randomize_query_claim_order=randomize_qc,
        ),
    }
    return datasets


def depth_probe_gold_oracle(sample):
    """Non-neural oracle: read the rank-1 source from metadata, return its value.
    Used by the smoke test to verify the generator produces consistent samples.
    """
    meta = sample["metadata"]
    rank_to_source = meta["rank_to_source"]
    top_source = rank_to_source[1] if 1 in rank_to_source else rank_to_source["1"]
    for c in meta["claims"]:
        if c["source"] == top_source:
            return c["value"]
    return None


