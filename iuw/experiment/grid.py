"""The MVP run grid and its codebook-familiarised extension.

Two renderings x two worlds x three seeds = 12 runs, 9,600 training
episodes. Each run's seed fixes its Masked mapping. The familiarised arm
(6 runs) gives the Masked rendering supervised label-function training first,
with no world transitions, then trains in both worlds.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Callable, Iterable, Optional

from ..llm.backend import Backend
from ..render.quiz import codebook_pairs
from ..render.rendering import make_rendering
from ..render.transcript import ChatFrame
from ..scoring import Scorer as ScoreNormaliser
from ..train.inner import RunConfig, RunResult, run_training, sealed_test

RENDERINGS = ("named", "masked")
WORLDS = ("reference", "counterfactual")
SEEDS = (0, 1, 2)

BackendFactory = Callable[[RunConfig], Backend]


def mvp_grid(template: Optional[RunConfig] = None, renderings=RENDERINGS, worlds=WORLDS,
             seeds=SEEDS) -> list[RunConfig]:
    template = template or RunConfig(run_id="template")
    out = []
    for r in renderings:
        for w in worlds:
            for s in seeds:
                out.append(replace(template, run_id=f"{r}-{w}-s{s}", rendering=r, train_world=w,
                                   seed_index=s))
    return out


def familiarised_grid(template: Optional[RunConfig] = None, seeds=SEEDS) -> list[RunConfig]:
    return [replace(c, run_id=f"masked-fam-{c.train_world}-s{c.seed_index}")
            for c in mvp_grid(template, renderings=("masked",), seeds=seeds)]


def familiarise_codebook(backend: Backend, mapping_index: int, steps: int = 20,
                         learning_rate: float = 1e-5) -> dict:
    """Supervised label-function pairs alone; then fix the result as the reference policy."""
    r = make_rendering("masked", mapping_index)
    frame = ChatFrame()
    system = r.system_prompt()
    tok = backend.tokenizer
    examples = []
    for prompt, completion in codebook_pairs(r):
        p = tok.encode(frame.question(system, prompt))
        c = tok.encode(completion + "<|im_end|>")
        examples.append((p, c))
    metrics = {}
    for _ in range(steps):
        metrics = backend.supervised(examples, learning_rate)
    backend.snapshot_reference()
    return {"steps": steps, "pairs": len(examples), **metrics}


def run_grid(configs: Iterable[RunConfig], backend_factory: BackendFactory,
             normaliser: ScoreNormaliser, out_root: Path, open_sealed_test: bool = False,
             familiarise: bool = False, progress=None) -> dict[str, RunResult]:
    results = {}
    for cfg in configs:
        run_dir = out_root / cfg.run_id
        if (run_dir / "result.json").exists():
            if progress:
                progress(f"[{cfg.run_id}] already complete; skipping")
            continue
        backend = backend_factory(cfg)
        if familiarise:
            fam = familiarise_codebook(backend, cfg.mapping_index)
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "familiarisation.json").write_text(json.dumps(fam, indent=2, default=str))
        res = run_training(cfg, backend, normaliser, run_dir, progress, resume=True)
        if open_sealed_test:
            sealed_test(cfg, backend, normaliser, run_dir)
        results[cfg.run_id] = res
    return results


def load_learning_costs(out_root: Path) -> dict[str, dict]:
    """Read each run's learning cost from disk: {run_id: {rendering, world, seed, cost}}."""
    out = {}
    for d in sorted(Path(out_root).iterdir()):
        res, cfg = d / "result.json", d / "config.json"
        if not res.exists() or not cfg.exists():
            continue
        r, c = json.loads(res.read_text()), json.loads(cfg.read_text())
        out[c["run_id"]] = {"rendering": c["rendering"], "world": c["train_world"],
                            "seed": c["seed_index"], "cost": r["learning_cost"],
                            "familiarised": (d / "familiarisation.json").exists()}
    return out
