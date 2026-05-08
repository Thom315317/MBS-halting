from mbs.datasets import BeliefRepairDataset
from mbs.graph import collate_graph_samples
from mbs.model import MBSModel
from mbs.tokenizer import SimpleTokenizer
from mbs.train import build_model


def test_t_sweep_runs_without_shape_errors():
    tokenizer = SimpleTokenizer()
    batch = collate_graph_samples([BeliefRepairDataset(1, split="ood_mixed", seed=7)[0]], tokenizer)
    model = MBSModel(vocab_size=len(tokenizer.tokens), num_values=8, d_state=32, message_steps=2, dropout=0.0)
    for steps in [2, 4, 8]:
        outputs = model(batch, message_steps=steps)
        assert outputs["logits"].shape == (1, 8)


def test_v0_3_variants_run():
    tokenizer = SimpleTokenizer()
    batch = collate_graph_samples([BeliefRepairDataset(1, split="train", seed=8)[0]], tokenizer)
    config = {
        "d_state": 32,
        "dropout": 0.0,
        "message_steps": 2,
        "halting": {"max_message_steps": 4, "min_message_steps": 2, "init_halt_prob": 0.05, "lambda_ponder": 0.001},
    }
    for variant in ["mbs_adaptive_halting", "rgcn_repair_stability", "rgcn_repair_stability_act_warmup_t8"]:
        model = build_model(variant, config, tokenizer)
        outputs = model(batch)
        assert outputs["logits"].shape == (1, 8)
