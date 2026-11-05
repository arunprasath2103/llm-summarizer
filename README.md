# PubMed Abstract Summarizer

Summarizes usable PubMed-style abstracts with a locally hosted **Llama 3.3 70B** model served by vLLM on an NVIDIA DGX Spark. The prompt asks the model to *“Summarize the clinical outcome of this study in one sentence.”*

- Handles missing fields, null-like values, inconsistent key names, and corrupted entries.
- Uses vLLM’s OpenAI-compatible `/v1/chat/completions` endpoint.
- Retries timeouts, dropped connections, HTTP 429 responses, and 5xx responses with exponential backoff and jitter.
- Fails fast with a clear message if the server is unavailable or the model name is wrong.
- Requires Python 3.8+ and one dependency: `requests`.

## Repository layout

```text
pubmed_summarizer.py     Main script
sample_data.json         Messy sample records from the brief
run_vllm.sh              Docker and vLLM launch command for the DGX Spark
requirements.txt         Python dependencies
tests/
  test_cleaning.py       Parsing, cleaning, and payload unit tests (no network)
  mock_llm_server.py     Fake vLLM server for testing timeouts, drops, and 503s
```

## Quick start

```bash
pip install -r requirements.txt

# 1. See what the cleaner does. No network needed.
python pubmed_summarizer.py --dry-run

# 2. Summarize against the local server.
# The default server is http://192.168.1.100:8000
python pubmed_summarizer.py --input sample_data.json --output summaries.json
```

With no `--input`, the embedded sample data is used. Use `--input -` to read from standard input.

Without `--output`, the JSON report is printed to standard output and logs go to standard error.

## Configuration

| Setting | Environment variable | CLI flag | Default |
|---|---|---|---|
| Endpoint | `LLM_URL` | `--url` | `http://192.168.1.100:8000/v1/chat/completions` |
| Model name | `LLM_MODEL` | `--model` | `Llama-3.3-70B-Instruct` |
| API key (optional) | `LLM_API_KEY` | — | None |
| Connect timeout | — | `--connect-timeout` | 5 s |
| Read timeout | — | `--read-timeout` | 120 s |
| Attempts per record | — | `--max-attempts` | 3 |
| Circuit breaker | — | `--max-consecutive-failures` | 3 |
| Skip `/v1/models` check | — | `--skip-preflight` | Off |

## Exit codes

| Code | Meaning |
|---:|---|
| `0` | Every usable record was summarized. |
| `1` | At least one record failed or was not attempted. |
| `2` | Fatal error: unreadable input, unreachable server, or model not served. |
| `130` | Interrupted with Ctrl-C. Partial results are still written. |

## Output

The script produces a JSON report with `stats`, per-record `results`, and `skipped` entries. Each skipped entry includes a reason.

Example output shape:

```json
{
  "stats": {
    "entries_read": 3,
    "usable": 2,
    "skipped": 1,
    "ok": 2,
    "failed": 0,
    "not_attempted": 0
  },
  "results": [
    {
      "id": "PMID:349281",
      "title": "Efficacy of Drug X in severe cardiac failure.",
      "status": "ok",
      "summary": "<one sentence from the model>",
      "attempts": 1,
      "latency_s": 12.4,
      "note": null
    }
  ],
  "skipped": [
    {
      "index": 3,
      "id": "ERR_NOT_FOUND",
      "reason": "no usable abstract (missing, null or empty)"
    }
  ]
}
```

## How the data is cleaned

| Problem in the legacy data | Handling |
|---|---|
| Different key names, such as `abstract_text` / `Abstract`, `ArticleTitle` / `title`, and `authorList` / `authors` | Keys are matched case-insensitively, with underscores ignored, against alias lists. |
| `null`, `""`, `"Unknown"`, `"N/A"`, or whitespace-only values | Treated as missing. |
| No abstract (record 3) | Skipped and reported with a reason. The model is not asked to guess. |
| Missing title (record 2) | Kept. The prompt uses `Title: (not provided)`. |
| `authors: "Unknown"` | Converted to an empty list. |
| IDs such as `"PMID:349281"`, `349281` (integer), or `"ERR_NOT_FOUND"` | Normalized to `PMID:<digits>`. A record with a valid abstract but unusable ID is kept as `record_<n>` with a warning. |
| Duplicate PMIDs or non-object entries | Skipped with a reason. |
| XML/HTML remnants such as `<AbstractText>` and `<i>`, or control characters | Stripped. Only known tags are removed, so clinical text such as `p<0.05` and `HR>1.2` is preserved. |
| Trailing commas, JSON Lines, or `{"records": [...]}` envelopes | Repaired or unwrapped. |
| Very long abstracts | Truncated to 8,000 characters (about 2,000 tokens) so prompts fit the server’s context window. |

The abstract for record 2 contradicts itself (“null hypothesis rejected” alongside “no significant improvement”). The prompt tells the model to point out the contradiction instead of guessing.

## Failure handling

| Situation | Behavior |
|---|---|
| Server is down at startup | The `/v1/models` preflight fails fast with exit code `2` and a hint that a 70B model can take minutes to load. |
| Wrong model name | The preflight lists the models offered by the server and exits with code `2`. |
| Connection refused, dropped, or reset | Retried with exponential backoff (2 s, 4 s, and so on, capped at 30 s) plus jitter. |
| HTTP `408`, `425`, `429`, `500`, `502`, `503`, or `504` | Retried, honoring `Retry-After`. |
| HTTP `400`, `401`, `404`, or `422` | Not retried. The record is marked as failed with the server’s message. |
| HTTP 200 with a truncated or invalid JSON body | Retried. |
| Three consecutive record failures | The circuit breaker marks the remaining records `not_attempted`, so a dead server does not cause a long wait for every record. |
| Ctrl-C | Partial results are written and the script exits with code `130`. |

### What if the server takes 45 seconds to respond?

The script waits and can still succeed. The default read timeout is 120 seconds, and responses slower than 30 seconds generate a warning. This is deliberate: a 70B model on a DGX Spark may decode at only a few tokens per second. Community reports suggest roughly 3–4 tokens per second, so measure your own system. A 120-token answer can therefore take 30–60 seconds, and a shorter timeout could trigger retries while the model is still working.

If no response arrives within 120 seconds, the request times out and is retried. For a non-streaming request, the read timeout effectively covers the total generation time. When the client disconnects, vLLM aborts the abandoned request, so retries do not pile up behind it.

Requests are sent one at a time to avoid building a queue behind a slow model. To raise throughput, increase `--max-num-seqs` on the server and add a small worker pool in the client.

## Server setup (DGX Spark, Docker, and vLLM)

```bash
export HF_TOKEN=hf_xxx  # Accept the Llama license on Hugging Face first
./run_vllm.sh
curl http://localhost:8000/v1/models
```

The `run_vllm.sh` script runs:

```bash
docker run -d --name llama33-70b \
  --gpus all --ipc=host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  --restart unless-stopped \
  -p 8000:8000 \
  -v $HOME/.cache/huggingface:/root/.cache/huggingface \
  -e HF_TOKEN=$HF_TOKEN \
  nvcr.io/nvidia/vllm:26.03.post1-py3 \
  vllm serve RedHatAI/Llama-3.3-70B-Instruct-FP8-dynamic \
    --served-model-name Llama-3.3-70B-Instruct \
    --dtype auto \
    --max-model-len 4096 \
    --max-num-seqs 4 \
    --gpu-memory-utilization 0.75 \
    --host 0.0.0.0 --port 8000
```

### Why these arguments?

- **FP8 checkpoint:** Llama 3.3 70B in 16-bit precision uses about 140 GB, exceeding the Spark’s 128 GB of unified memory. FP8 uses about 70 GB. NVFP4 (about 40 GB) is smaller, but reports describe issues with it on the Spark’s GB10 GPU, so FP8 is the safer default.
- **`--gpu-memory-utilization 0.75`:** CPU and GPU share one memory pool on the Spark. A GPU out-of-memory error can become system-wide memory exhaustion. At roughly 120 GB usable, `0.75` allocates about 90 GB to vLLM for weights, activations, and KV cache, leaving about 30 GB for the OS and Docker. Community guidance suggests 0.7 or lower. If memory is tight, run `GPU_UTIL=0.70 ./run_vllm.sh`.
- **`--max-model-len 4096` and `--max-num-seqs 4`:** Llama 3.3 70B needs about 320 KiB of KV cache per token in 16-bit precision (80 layers × 8 KV heads × 128 dimensions × K and V × 2 bytes). Four sequences of 4,096 tokens use about 5 GiB. Abstracts are short, so a longer context would use memory without much benefit.
- **`--served-model-name`:** Must match the client’s `--model` option or `LLM_MODEL` environment variable.
- **`--ipc=host` and the `--ulimit` flags:** Standard options for the NGC vLLM container.
- **Dropping the page cache:** Frees memory that would otherwise compete with the weights in unified memory.

If `docker pull` is denied, run `docker login nvcr.io` with username `$oauthtoken` and your NGC API key as the password. Check the NGC catalog for the newest `nvcr.io/nvidia/vllm` tag.

## Testing

```bash
# Unit tests (no network)
python -m unittest discover -s tests -v

# Simulate failure modes against a mock server.
# Behaviors are consumed one per POST: ok | slow:<seconds> | drop | 503 | 400
python tests/mock_llm_server.py 8801 drop 503 ok ok &
python pubmed_summarizer.py --url http://127.0.0.1:8801/v1/chat/completions

# Test the 45-second response case with default timeouts.
python tests/mock_llm_server.py 8802 slow:45 ok &
python pubmed_summarizer.py --url http://127.0.0.1:8802/v1/chat/completions
```

The script was exercised against the mock server with a dropped connection, 503 response, read timeout, 45-second response, dead server, wrong model name, and circuit breaker. It has not been benchmarked on real DGX Spark hardware, so validate the Docker settings on your machine and adjust `GPU_UTIL`, `MAX_LEN`, and `MAX_SEQS` if needed.
