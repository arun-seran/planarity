"""Causal tests on LoRA adapters (run on the rented inspection GPU).

Weight distances, probes and activation differences only nominate
candidates. These interventions test them, preferably within one checkpoint:

* switch off the adapter layer by layer and in blocks;
* keep only the top singular directions of each adapter (how few carry it?);
* add the mean activation difference to the base model as a steering vector;
* restore ablated components (rescue), and compare with random components
  matched for layer and scale.

All interventions are context managers that restore the weights on exit.
They work on a PEFT model, such as ``HFBackend.model`` loaded with a
downloaded Tinker adapter (``HFBackend(base, init_adapter=path)``).
"""

from __future__ import annotations

import re
from contextlib import contextmanager
from typing import Iterable, Optional, Sequence

import numpy as np

_LAYER_RE = re.compile(r"\.layers\.(\d+)\.")


def lora_modules(model) -> list[tuple[str, int, object]]:
    """(name, layer index, module) for every module carrying LoRA weights."""
    out = []
    for name, mod in model.named_modules():
        if hasattr(mod, "lora_A") and hasattr(mod, "lora_B") and len(getattr(mod, "lora_A", {})):
            m = _LAYER_RE.search(name)
            out.append((name, int(m.group(1)) if m else -1, mod))
    return out


def _ab(mod, adapter: str = "default"):
    return mod.lora_A[adapter].weight, mod.lora_B[adapter].weight, mod.scaling[adapter]


@contextmanager
def ablate_layers(model, layers: Iterable[int], adapter: str = "default"):
    """Zero the adapter in the given decoder layers (B = 0 removes the update)."""
    import torch

    layers = set(layers)
    saved = []
    with torch.no_grad():
        for name, layer, mod in lora_modules(model):
            if layer in layers:
                B = mod.lora_B[adapter].weight
                saved.append((B, B.detach().clone()))
                B.zero_()
    try:
        yield len(saved)
    finally:
        with torch.no_grad():
            for B, val in saved:
                B.copy_(val)


def layer_blocks(n_layers: int, block: int) -> list[list[int]]:
    return [list(range(i, min(i + block, n_layers))) for i in range(0, n_layers, block)]


@contextmanager
def keep_top_directions(model, k: int, adapter: str = "default", random_control: bool = False,
                        seed: int = 0):
    """Replace each adapter update dW = scale * B @ A by its rank-k truncation.

    With ``random_control`` keep k random orthonormal directions of the same
    adapter instead, rescaled to the top-k singular values (matched for layer
    and scale).
    """
    import torch

    gen = torch.Generator().manual_seed(seed)
    saved = []
    with torch.no_grad():
        for name, layer, mod in lora_modules(model):
            A, B, scale = _ab(mod, adapter)
            saved.append((A, A.detach().clone(), B, B.detach().clone()))
            dW = (B.float() @ A.float())
            U, S, Vh = torch.linalg.svd(dW, full_matrices=False)
            r = A.shape[0]
            kk = min(k, r, len(S))
            if random_control:
                Q1, _ = torch.linalg.qr(torch.randn(U.shape[0], kk, generator=gen))
                Q2, _ = torch.linalg.qr(torch.randn(Vh.shape[1], kk, generator=gen))
                Uk, Vk = Q1.to(dW.device), Q2.T.to(dW.device)  # CPU generator keeps the draw device-independent
            else:
                Uk, Vk = U[:, :kk], Vh[:kk]
            Sk = S[:kk]
            newB = torch.zeros_like(B, dtype=torch.float32)
            newA = torch.zeros_like(A, dtype=torch.float32)
            newB[:, :kk] = Uk * Sk.sqrt()
            newA[:kk] = Sk.sqrt()[:, None] * Vk
            B.copy_(newB.to(B.dtype))
            A.copy_(newA.to(A.dtype))
    try:
        yield
    finally:
        with torch.no_grad():
            for A, a, B, b in saved:
                A.copy_(a)
                B.copy_(b)


def adapter_spectrum(model, adapter: str = "default") -> dict[str, list[float]]:
    """Singular values of each adapter update (nominates candidates; proves nothing)."""
    import torch

    out = {}
    with torch.no_grad():
        for name, layer, mod in lora_modules(model):
            A, B, scale = _ab(mod, adapter)
            out[name] = (torch.linalg.svdvals(B.float() @ A.float()) * scale).tolist()
    return out


def _decoder_layers(model):
    for attr in ("model.model.layers", "base_model.model.model.layers", "model.layers"):
        obj = model
        try:
            for part in attr.split("."):
                obj = getattr(obj, part)
            return obj
        except AttributeError:
            continue
    raise AttributeError("could not find decoder layers")


def capture_residual(model, token_batches: Sequence[Sequence[int]], positions: Sequence[Sequence[int]],
                     layer: int, device: Optional[str] = None) -> np.ndarray:
    """Residual-stream activations (output of decoder layer ``layer``) at the given positions."""
    import torch

    layers = _decoder_layers(model)
    grabbed = []

    def hook(_, __, out):
        grabbed.append((out[0] if isinstance(out, tuple) else out).detach().float().cpu())

    h = layers[layer].register_forward_hook(hook)
    acts = []
    try:
        with torch.no_grad():
            for toks, pos in zip(token_batches, positions):
                grabbed.clear()
                ids = torch.as_tensor(list(toks), device=device or next(model.parameters()).device)[None]
                model(input_ids=ids)
                acts.append(grabbed[0][0, [p - 1 for p in pos]].numpy())
    finally:
        h.remove()
    return np.concatenate(acts) if acts else np.zeros((0,))


def mean_activation_difference(model, token_batches, positions, layer: int, adapter_off_ctx) -> np.ndarray:
    """Adapted minus base mean activation at decision positions (same model, adapter toggled)."""
    adapted = capture_residual(model, token_batches, positions, layer)
    with adapter_off_ctx():
        base = capture_residual(model, token_batches, positions, layer)
    return adapted.mean(0) - base.mean(0)


@contextmanager
def steer(model, layer: int, vector: np.ndarray, alpha: float = 1.0):
    """Add ``alpha * vector`` to the output of decoder layer ``layer`` at every position."""
    import torch

    layers = _decoder_layers(model)
    v = torch.as_tensor(vector)

    def hook(_, __, out):
        if isinstance(out, tuple):
            return (out[0] + alpha * v.to(out[0].dtype).to(out[0].device),) + tuple(out[1:])
        return out + alpha * v.to(out.dtype).to(out.device)

    h = layers[layer].register_forward_hook(hook)
    try:
        yield
    finally:
        h.remove()
