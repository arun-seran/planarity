"""Torch-only code paths: HF backend, adapter interventions. Skipped without torch/peft."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("peft")
transformers = pytest.importorskip("transformers")

from iuw.config import COUNTERFACTUAL, PROFILE_A  # noqa: E402
from iuw.inspection import adapters  # noqa: E402
from iuw.llm.backend import LossConfig  # noqa: E402
from iuw.llm.hf_backend import HFBackend  # noqa: E402
from iuw.llm.policy import EpisodeSpec, SamplingConfig, rollout  # noqa: E402
from iuw.llm.tokenizer import ApproxQwenTokenizer  # noqa: E402
from iuw.render import make_rendering  # noqa: E402
from iuw.train.advantages import compute_advantages  # noqa: E402
from iuw.train.inner import make_train_items  # noqa: E402


class SmallVocab(ApproxQwenTokenizer):
    """Folds approximate token ids into a tiny vocabulary for a random-init model."""

    V = 512

    def encode(self, text):
        return [i % self.V for i in super().encode(text)]


@pytest.fixture(scope="module")
def backend():
    cfg = transformers.Qwen2Config(vocab_size=SmallVocab.V, hidden_size=32, intermediate_size=64,
                                   num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2)
    model = transformers.AutoModelForCausalLM.from_config(cfg)
    be = HFBackend("tiny", rank=4, model=model, tokenizer=SmallVocab(), device="cpu")
    for name, _, mod in adapters.lora_modules(be.model):  # make adapters non-trivial
        torch.nn.init.normal_(mod.lora_B["default"].weight, std=0.02)
    return be


def test_hf_rollout_and_first_pass_ratio(backend):
    r = make_rendering("named")
    spec = EpisodeSpec(COUNTERFACTUAL, PROFILE_A, 3, r)
    for loss in ("builtin-ppo", "renormalised"):
        trajs = rollout([spec], backend.scorer(), backend.tokenizer, SamplingConfig())
        assert not trajs[0].violations
        items = make_train_items(trajs, compute_advantages(trajs))
        m = backend.train(items, LossConfig(loss=loss, learning_rate=1e-4))
        assert m["ratio_first_pass"] == pytest.approx(1.0, abs=1e-3)


def test_adapter_interventions_restore_weights(backend):
    before = {k: v.clone() for k, v in backend.lora_state().items()}
    with adapters.ablate_layers(backend.model, [0, 1]) as n:
        assert n > 0
    with adapters.keep_top_directions(backend.model, k=1):
        pass
    with adapters.keep_top_directions(backend.model, k=1, random_control=True):
        pass
    after = backend.lora_state()
    assert all(torch.equal(before[k], after[k]) for k in before)
    spec = adapters.adapter_spectrum(backend.model)
    assert spec and all(len(v) >= 1 for v in spec.values())
