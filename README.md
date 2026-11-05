# PubMed Abstract Summarizer (local Llama 3.3 70B via vLLM)

Parses messy, legacy PubMed-style JSON, cleans it, and asks a locally hosted
**Llama 3.3 70B** (served by vLLM on an NVIDIA DGX Spark) to
*"Summarize the clinical outcome of this study in one sentence."* for every usable abstract.

- Handles missing fields, null-like values, inconsistent key names and corrupted entries
- Talks to vLLM's OpenAI-compatible `/v1/chat/completions` endpoint
- Retries with exponential backoff + jitter on timeouts, dropped connections, 429 and 5xx
- Fails fast, with a clear message, if the server is down or the model name is wrong
- One dependency (`requests`), Python 3.8+
- 
## Repository layout

```
pubmed_summarizer.py     the script
sample_data.json         the messy sample records from the brief
run_vllm.sh              Docker + vLLM launch command for the DGX Spark
requirements.txt
tests/
  test_cleaning.py       unit tests for parsing / cleaning / payload (no network)
  mock_llm_server.py     fake vLLM server for testing timeouts, drops, 503s
```

## Quick start

```bash
pip install -r requirements.txt

# 1. See what the cleaner does. No network needed.
python pubmed_summarizer.py --dry-run

# 2. Summarize against the local server (default http://192.168.1.100:8000)
python pubmed_summarizer.py --input sample_data.json --output summaries.json
```

With no `--input` the embedded sample data is used. `--input -` reads from stdin.
Without `--output` the JSON report is printed to stdout and logs go to stderr.

### Configuration

| Setting | Env var | CLI flag | Default |
|---|---|---|---|
| Endpoint | `LLM_URL` | `--url` | `http://192.168.1.100:8000/v1/chat/completions` |
| Model name | `LLM_MODEL` | `--model` | `Llama-3.3-70B-Instruct` |
| API key (optional) | `LLM_API_KEY` | - | none |
| Connect timeout | - | `--connect-timeout` | 5 s |
| Read timeout | - | `--read-timeout` | 120 s |
| Attempts per record | - | `--max-attempts` | 3 |
| Circuit breaker | - | `--max-consecutive-failures` | 3 |
| Skip `/v1/models` check | - | `--skip-preflight` | off |

### Exit codes

| Code | Meaning |
|---|---|
| 0 | every usable record was summarized |
| 1 | at least one record failed or was not attempted |
| 2 | fatal: unreadable input, server unreachable, or model not served |
| 130 | interrupted with Ctrl-C (partial results are still written) |

### Output

A JSON report with `stats`, per-record `results` and the `skipped` entries (each with a reason).
The shape is illustrative:

```json
{
  "stats": {"entries_read": 3, "usable": 2, "skipped": 1, "ok": 2, "failed": 0, "not_attempted": 0},
  "results": [
    {"id": "PMID:349281", "title": "Efficacy of Drug X in severe cardiac failure.",
     "status": "ok", "summary": "<one sentence from the model>", "attempts": 1, "latency_s": 12.4, "note": null}
  ],
  "skipped": [
    {"index": 3, "id": "ERR_NOT_FOUND", "reason": "no usable abstract (missing, null or empty)"}
  ]
}
```

## How the data is cleaned

| Problem in the legacy data | Handling |
|---|---|
| Different key names (`abstract_text` / `Abstract`, `ArticleTitle` / `title`, `authorList` / `authors`) | Keys are matched case-insensitively, ignoring `_`, against alias lists |
| `null`, `""`, `"Unknown"`, `"N/A"`, whitespace-only | Treated as missing |
| No abstract (record 3) | Skipped and reported with a reason. There is nothing to summarize, and the model is never asked to guess |
| Missing title (record 2) | Kept. The prompt says `Title: (not provided)` |
| `authors: "Unknown"` | Becomes an empty list |
| IDs as `"PMID:349281"`, `349281` (int) or `"ERR_NOT_FOUND"` | Normalized to `PMID:<digits>`. An unusable ID with a valid abstract is kept as `record_<n>` with a warning |
| Duplicate PMIDs, non-object entries | Skipped with a reason |
| XML/HTML remnants (`<AbstractText>`, `<i>`) and control characters | Stripped. Only known tags are removed, so clinical text like `p<0.05` and `HR>1.2` survives |
| Trailing commas, JSON Lines, `{"records": [...]}` envelopes | Repaired / unwrapped |
| Very long abstracts | Truncated to 8,000 characters (about 2k tokens) so prompts fit the server's context window |

Note that the abstract for record 2 contradicts itself ("null hypothesis rejected" next to
"no significant improvement"). The prompt tells the model to say so instead of guessing.

## Failure handling

| Situation | Behaviour |
|---|---|
| Server down at startup | `/v1/models` preflight fails fast, exit 2, with a hint that a 70B model can take minutes to load |
| Wrong model name | Preflight lists what the server offers and exits 2 |
| Connection refused / dropped / reset | Retry with exponential backoff (2 s, 4 s ... capped at 30 s) plus jitter |
| HTTP 408, 425, 429, 500, 502, 503, 504 | Retry (honours `Retry-After`) |
| HTTP 400, 401, 404, 422 | No retry, because retrying cannot help. The record is marked failed with the server's message |
| 200 with a truncated / invalid JSON body | Retry |
| 3 records in a row fail | Circuit breaker. The remaining records are marked `not_attempted`, so a dead server does not cost a long hang per record |
| Ctrl-C | Partial results are written, exit 130 |

### What happens if the server takes 45 seconds to respond?

The script waits and succeeds. The default read timeout is 120 s, and a response slower than 30 s
logs a warning. This is deliberate: a 70B model on a DGX Spark decodes at only a few tokens per
second (community reports are roughly 3-4 tok/s, so measure your own box). A 120-token answer
can therefore legitimately take 30-60 s, and a short timeout would just trigger pointless retries
while the model is still working.

If nothing arrives within 120 s, the request is treated as a timeout and retried. For a
non-streaming request the read timeout is effectively the total generation time. When the client
disconnects, vLLM aborts the abandoned request, so retries do not pile up behind it.

Requests are sent one at a time on purpose, so a slow model does not build a queue. To raise
throughput, increase `--max-num-seqs` on the server and add a small worker pool in the client.

## Server setup (DGX Spark, Docker + vLLM)

```bash
export HF_TOKEN=hf_xxx      # accept the Llama license on Hugging Face first
./run_vllm.sh
curl http://localhost:8000/v1/models
```

`run_vllm.sh` runs:

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

### Why these arguments (avoiding an OOM)

- **FP8 checkpoint.** Llama 3.3 70B in 16-bit is about 140 GB, which exceeds the Spark's 128 GB of
  unified memory. FP8 is about 70 GB. NVFP4 (about 40 GB) is smaller, but there are reports of
  problems with it on the Spark's GB10 GPU, so FP8 is the safer default.
- **`--gpu-memory-utilization 0.75`.** CPU and GPU share one memory pool on the Spark, so a GPU
  out-of-memory error can become system-wide memory exhaustion. 0.75 of roughly 120 GB usable is
  about 90 GB for vLLM: about 70 GB for weights, the rest for activations and KV cache, leaving
  about 30 GB for the OS and Docker. Community guidance is 0.7 or lower, so use `GPU_UTIL=0.70
  ./run_vllm.sh` if memory is tight.
- **`--max-model-len 4096` and `--max-num-seqs 4`.** Llama 3.3 70B needs about 320 KiB of KV cache
  per token in 16-bit (80 layers x 8 KV heads x 128 dims x K and V x 2 bytes). 4 sequences x
  4,096 tokens is about 5 GiB. Abstracts are short, so a long context would only waste memory.
- **`--served-model-name`.** Must match the client's `--model` / `LLM_MODEL`.
- **`--ipc=host` and the `--ulimit` flags.** Standard for the NGC vLLM container.
- **Dropping the page cache** before launch frees memory that would otherwise compete with the
  weights in unified memory.

If `docker pull` is denied, run `docker login nvcr.io` (username `$oauthtoken`, password your NGC API key)
and check the NGC catalog for the newest `nvcr.io/nvidia/vllm` tag.

## Testing

```bash
# unit tests (no network)
python -m unittest discover -s tests -v

# simulate failure modes against a mock server:
#   behaviours are consumed one per POST: ok | slow:<seconds> | drop | 503 | 400
python tests/mock_llm_server.py 8801 drop 503 ok ok &
python pubmed_summarizer.py --url http://127.0.0.1:8801/v1/chat/completions

# the 45-second case, with default timeouts
python tests/mock_llm_server.py 8802 slow:45 ok &
python pubmed_summarizer.py --url http://127.0.0.1:8802/v1/chat/completions
```

The script was exercised against this mock server (dropped connection, 503, read timeout,
45 s response, dead server, wrong model name, circuit breaker). It has not been benchmarked on
real DGX Spark hardware, so validate the Docker settings on your machine and adjust
`GPU_UTIL`, `MAX_LEN` and `MAX_SEQS` if needed.
#   l l m - s u m m a r i z e r 
 
 
