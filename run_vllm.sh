#!/usr/bin/env bash
# Launch vLLM serving Llama 3.3 70B (FP8) on an NVIDIA DGX Spark (128 GB unified memory).
#
# Why these settings (see README, "Server setup"):
#   * 70B in BF16 is ~140 GB and cannot fit in 128 GB -> use an FP8 checkpoint (~70 GB).
#   * CPU and GPU share one memory pool on the Spark, so an OOM can freeze the whole machine.
#     Cap vLLM at 75% of memory and keep the context / concurrency small.
#
# Usage:
#   export HF_TOKEN=hf_xxx        # Hugging Face token (accept the Llama license first)
#   ./run_vllm.sh
#
# Override anything with env vars, e.g.:  GPU_UTIL=0.70 ./run_vllm.sh
set -euo pipefail

: "${HF_TOKEN:?Set HF_TOKEN to your Hugging Face token first}"

IMAGE="${VLLM_IMAGE:-nvcr.io/nvidia/vllm:26.03.post1-py3}"   # check NGC for the newest tag
MODEL="${MODEL:-RedHatAI/Llama-3.3-70B-Instruct-FP8-dynamic}"
SERVED_NAME="${SERVED_NAME:-Llama-3.3-70B-Instruct}"          # must match LLM_MODEL / --model in the client
GPU_UTIL="${GPU_UTIL:-0.75}"
MAX_LEN="${MAX_LEN:-4096}"
MAX_SEQS="${MAX_SEQS:-4}"
PORT="${PORT:-8000}"

# Optional: free page cache before loading ~70 GB of weights into unified memory.
sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches' || echo "(skipped dropping page cache)"

docker run -d --name llama33-70b \
  --gpus all --ipc=host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  --restart unless-stopped \
  -p "${PORT}:8000" \
  -v "${HOME}/.cache/huggingface:/root/.cache/huggingface" \
  -e HF_TOKEN="${HF_TOKEN}" \
  "${IMAGE}" \
  vllm serve "${MODEL}" \
    --served-model-name "${SERVED_NAME}" \
    --dtype auto \
    --max-model-len "${MAX_LEN}" \
    --max-num-seqs "${MAX_SEQS}" \
    --gpu-memory-utilization "${GPU_UTIL}" \
    --host 0.0.0.0 --port 8000

echo "Started. Loading a 70B model can take several minutes. Follow progress with:"
echo "  docker logs -f llama33-70b"
echo "Ready when this returns the model list:"
echo "  curl http://localhost:${PORT}/v1/models"
