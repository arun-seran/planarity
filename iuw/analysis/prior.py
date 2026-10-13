"""Prior conflict: the interaction term in learning cost.

    Delta_prior = (C_Named^CF - C_Named^Ref) - (C_Masked^CF - C_Masked^Ref)

C is cumulative regret against the reference return over a run's training
episodes. The first difference mixes priors with the counterfactual world's
own difficulty; the second measures that difficulty alone. A positive value
is *label-associated interference*: not a direct measure of unlearning, and
a zero gap does not show the model lacks relevant priors.

The interval is clustered by training run: runs are resampled within each
cell. It must be positive under every Masked baseline that was run
(standard and, if gated in, codebook-familiarised).
"""

from __future__ import annotations

from typing import Mapping, Optional, Sequence

import numpy as np

CELLS = (("named", "counterfactual"), ("named", "reference"),
         ("masked", "counterfactual"), ("masked", "reference"))


def delta_prior_from_cells(cells: Mapping[tuple[str, str], Sequence[float]]) -> float:
    m = {k: float(np.mean(v)) for k, v in cells.items()}
    return ((m[("named", "counterfactual")] - m[("named", "reference")])
            - (m[("masked", "counterfactual")] - m[("masked", "reference")]))


def delta_prior(costs: Mapping[str, Mapping], masked_familiarised: Optional[bool] = False,
                n_boot: int = 10_000, seed: int = 0) -> dict:
    """``costs``: {run_id: {"rendering", "world", "cost", "familiarised"}} (see grid.load_learning_costs).

    ``masked_familiarised`` selects which Masked runs form the baseline:
    False = standard Masked runs, True = codebook-familiarised runs.
    """
    cells: dict[tuple[str, str], list[float]] = {k: [] for k in CELLS}
    for r in costs.values():
        if r["rendering"] == "masked" and bool(r.get("familiarised")) != bool(masked_familiarised):
            continue
        key = (r["rendering"], r["world"])
        if key in cells:
            cells[key].append(float(r["cost"]))
    missing = [k for k, v in cells.items() if not v]
    if missing:
        return {"delta_prior": None, "missing_cells": missing}
    point = delta_prior_from_cells(cells)
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        res = {k: rng.choice(v, size=len(v), replace=True) for k, v in cells.items()}
        boots.append(delta_prior_from_cells(res))
    lo, hi = np.quantile(boots, [0.025, 0.975])
    return {
        "delta_prior": point, "ci95": [float(lo), float(hi)],
        "cell_means": {f"{r}/{w}": float(np.mean(v)) for (r, w), v in cells.items()},
        "runs_per_cell": {f"{r}/{w}": len(v) for (r, w), v in cells.items()},
        "positive": bool(lo > 0),
        "baseline": "codebook-familiarised" if masked_familiarised else "standard",
        "reading": "label-associated interference" if lo > 0 else "no interval-supported interference",
    }


def prior_conflict_gate(results: Sequence[dict]) -> dict:
    """Passes only if Delta_prior > 0 with a positive interval under every baseline run."""
    ran = [r for r in results if r.get("delta_prior") is not None]
    return {"passes": bool(ran) and all(r["positive"] for r in ran),
            "baselines": [r["baseline"] for r in ran]}
