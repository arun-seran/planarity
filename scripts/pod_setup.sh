#!/usr/bin/env bash
# One-time setup on a Runpod GPU pod (runpod/pytorch CUDA 12.8 image), run from the repo root
# after the code is copied over:  bash scripts/pod_setup.sh
# Installs the package with the local-model extras plus the fused kernels that Qwen3.5/3.6's
# linear-attention layers use (without them transformers falls back to a slow torch path),
# then downloads the main model into the container disk's Hugging Face cache.
set -euo pipefail
export HF_HOME=${HF_HOME:-/root/hf}
# The image's system Python is "externally managed" (PEP 668); a venv that can still see the
# image's CUDA build of torch avoids reinstalling it.
python -m venv --system-site-packages /root/venv
source /root/venv/bin/activate
pip install -q --upgrade pip
pip install -q -e ".[test,local]"
pip install -q flash-linear-attention || echo "WARN: flash-linear-attention not installed (slow fallback)"
pip install -q causal-conv1d --no-build-isolation || echo "WARN: causal-conv1d not installed (slow fallback)"
pip install -q tilelang || echo "WARN: tilelang not installed (fla needs it on Hopper with Triton < 3.7.1)"
python - <<'PY'
import torch, transformers, peft
print("torch", torch.__version__, "cuda", torch.version.cuda, "gpu", torch.cuda.get_device_name(0),
      f"{torch.cuda.get_device_properties(0).total_memory / 2**30:.0f} GiB")
print("transformers", transformers.__version__, "peft", peft.__version__)
PY
MODEL=${MODEL:-Qwen/Qwen3.6-35B-A3B}
python -c "import sys; from huggingface_hub import snapshot_download as d; print(d(sys.argv[1]))" "$MODEL"
du -sh "$HF_HOME"
pytest -q -m "not slow"
