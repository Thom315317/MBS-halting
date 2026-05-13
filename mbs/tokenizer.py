import re


class SimpleTokenizer:
    def __init__(self):
        base_tokens = [
            "<pad>",
            "<unk>",
            "source",
            "says",
            "is",
            "more",
            "reliable",
            "than",
            "latest",
            "wins",
            "trusted",
            "rollback",
            "priority",
            "chain",
            "query",
            "location",
            "status",
            "color",
            "at",
            "t1",
            "t2",
            "t3",
            "t4",
            "t5",
            "t6",
            "active",
            "inactive",
            "earliest",
            "second",
            "least",
            "latest_wins",
            "trusted_source_wins",
            "rollback_source",
            "source_priority_chain",
            "earliest_wins",
            "second_latest_wins",
            "least_reliable_source_wins",
            # Used by the depth_controlled_latent_halting_probe task only.
            "trusted_chain_top_wins",
        ]
        entities = [
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
        # SOURCES extended from 8 (A..H) to 20 (A..T) for the
        # depth_controlled_latent_halting_probe_v2 task, which needs k_max up
        # to 20. The first 8 IDs are unchanged so v1 / v0.4 / belief_repair_*
        # checkpoints reload bit-identically (they only ever saw A..H).
        sources = ["A", "B", "C", "D", "E", "F", "G", "H",
                   "I", "J", "K", "L", "M", "N", "O", "P",
                   "Q", "R", "S", "T"]
        values = ["Paris", "Berlin", "Rome", "London", "Madrid", "Oslo", "Tokyo", "Cairo"]
        self.tokens = []
        self.token_to_id = {}
        for token in base_tokens + entities + sources + values:
            self.add(token)

    @property
    def pad_id(self):
        return self.token_to_id["<pad>"]

    @property
    def unk_id(self):
        return self.token_to_id["<unk>"]

    def add(self, token: str) -> int:
        if token not in self.token_to_id:
            self.token_to_id[token] = len(self.tokens)
            self.tokens.append(token)
        return self.token_to_id[token]

    def encode(self, text: str, max_len: int = 160):
        pieces = re.findall(r"[A-Za-z0-9_]+", text)
        ids = [self.token_to_id.get(piece, self.unk_id) for piece in pieces[:max_len]]
        return ids or [self.unk_id]

    def encode_cell(self, text: str) -> int:
        ids = self.encode(text, max_len=1)
        return ids[0]

    def state_dict(self):
        return {"tokens": self.tokens}

    @classmethod
    def from_state_dict(cls, state):
        obj = cls()
        obj.tokens = []
        obj.token_to_id = {}
        for token in state["tokens"]:
            obj.add(token)
        return obj
