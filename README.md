# Reproducing vLLM KV-cache saturation and request throttling

This repository demonstrates a specific vLLM scheduler behavior on a small model:

1. Constrain the GPU KV cache to exactly one 2,048-token sequence.
2. Submit four distinct, near-context-limit requests simultaneously.
3. Observe one running request, three waiting requests, and 100% KV-block use.
4. Observe queued requests completing serially as the active request releases blocks.

This is scheduler admission control—not HTTP rate limiting. Every request is accepted and eventually returns HTTP 200; requests without available KV blocks remain in vLLM's internal waiting queue.

## Verified result

The experiment was verified with vLLM 0.11.2 on an NVIDIA RTX 4060 Ti (16 GiB):

```text
GPU KV cache size: 2,048 tokens
Maximum concurrency for 2,048 tokens per request: 1.00x

0.168s  running=1  waiting=3  KV=86.61%
2.541s  running=1  waiting=3  KV=100.00%
2.804s  running=1  waiting=2  KV=86.61%
5.178s  running=1  waiting=2  KV=100.00%
```

Client completion times were 2.670, 5.291, 7.921, and 10.564 seconds. See [`PROOF.md`](PROOF.md) for the full representative trace.

## Repository contents

| Path | Purpose |
|---|---|
| `download_benchmark.py` | Downloads and verifies the pinned benchmark file |
| `prepare_prompt.py` | Selects the longest QA record and trims it by tokenizer |
| `run_server.sh` | Starts vLLM with the exact KV-cache allocation |
| `load_test.py` | Sends synchronized concurrent requests |
| `proof_test.py` | Sends requests while sampling vLLM scheduler metrics |
| `requirements.txt` | Exact direct dependency versions from the verified run |
| `Dockerfile` | Reproducible NVIDIA-container execution path |
| `PROOF.md` | Captured evidence and interpretation |
## Benchmark source and provenance

The input comes from the official [LooGLE long-context benchmark](https://github.com/bigai-nlco/LooGLE), pinned to commit `6734382215bea3a63f055ffd7873b2967b6a2477`. `download_benchmark.py` retrieves `LooGLE-testdata/longdep_qa.jsonl` from GitHub and requires this SHA-256 digest:

```text
596af3daf28053ff8c29c7c88eb132b9053497d8021db8c4f84af446363a3927
```

The selected record, **Urban planning of Barcelona**, contains 115,286 characters and 18,544 whitespace-delimited words. It tokenizes to about 29,167 TinyLlama tokens before trimming. The prepared raw prompt is 1,750 tokens; vLLM reports 1,751 prompt tokens after adding its beginning-of-sequence token.

## Why these exact settings?

### Model and context

`TinyLlama/TinyLlama-1.1B-Chat-v1.0` is small enough for commodity GPUs and has a native 2,048-token context. We explicitly set:

```text
--max-model-len 2048
```

This makes the test boundary obvious and avoids testing an artificially extended context.

### Exact KV-cache size

For TinyLlama in FP16, KV bytes per token are:

```text
2 (K and V) × 22 layers × 4 KV heads × 64 head dimensions × 2 bytes
= 22,528 bytes/token
```

The configured allocation is therefore:

```text
22,528 bytes/token × 2,048 tokens = 46,137,344 bytes
```

`run_server.sh` passes `--kv-cache-memory-bytes 46137344`. vLLM confirms the result at startup as exactly 2,048 GPU KV tokens and 1.00x maximum concurrency for 2,048-token requests. The explicit byte setting overrides indirect cache sizing through `--gpu-memory-utilization`.

### Why 1,751 input + 282 output tokens?

Each response uses 2,033 logical tokens. vLLM allocates KV blocks in groups of 16 tokens, so 2,033 tokens occupy all 128 blocks (`2,048 / 16`) and report 100% cache use. Fifteen positions remain inside the final allocated block, allowing completion. A 297-token output would use all 2,048 logical positions and caused a scheduler stall during boundary testing, so 282 is the proven safe setting.

`ignore_eos=true` forces every request to generate the requested output length. A unique marker appears at the beginning of each prompt, preventing automatic prefix caching from sharing the long document prefix between requests.

`--max-num-seqs 16` is intentionally higher than the four test clients. Therefore, the observed queue is caused by KV capacity rather than the sequence-count limit.
## Recommended reproduction: Docker

### Prerequisites

- Linux with an NVIDIA GPU
- A compatible NVIDIA driver
- Docker Engine
- NVIDIA Container Toolkit configured so `docker run --gpus all ...` works
- Internet access while building (benchmark/tokenizer) and on first run (model weights)

Build the image from the repository root:

```bash
docker build --pull -t vllm-kv-throttle:v0.11.2 .
```

Start the server. Mounting the Hugging Face cache avoids downloading the 2.2 GB model again after the container is removed:

```bash
docker run --rm --name vllm-kv-proof \
  --gpus all \
  --ipc=host \
  -p 8000:8000 \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  vllm-kv-throttle:v0.11.2
```

Wait until the logs contain `Application startup complete`, then execute the proof inside the running container:

```bash
docker exec vllm-kv-proof python3 proof_test.py
```

In another terminal, inspect the raw metrics if desired:

```bash
curl -s http://127.0.0.1:8000/metrics
```

Stop and remove the container with `Ctrl+C` in the server terminal, or:

```bash
docker stop vllm-kv-proof
```

The image is based on the official `vllm/vllm-openai:v0.11.2` image. It installs the exact requirements, downloads the commit-pinned benchmark file, validates its digest, and prepares the 1,750-token prompt during the build.
## Native Python reproduction

Python 3.11 and a CUDA environment supported by vLLM 0.11.2 are required. Use a clean virtual environment; mixing vLLM with unrelated globally installed FastAPI or Transformers versions can break startup.

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python download_benchmark.py
.venv/bin/python prepare_prompt.py
```

Start the long-running server in terminal 1:

```bash
PYTHON=.venv/bin/python ./run_server.sh
```

Run the integrated proof in terminal 2:

```bash
.venv/bin/python proof_test.py
```

Or run only the clients without metric sampling:

```bash
.venv/bin/python load_test.py --concurrency 4
```

## Reading the evidence

The decisive metric combination is:

```text
num_requests_running = 1
num_requests_waiting = 3
kv_cache_usage_perc  = 1.0
```

This proves that requests reached vLLM and were admitted to its scheduler, but only one had enough KV capacity to execute. Completion times increasing in approximately one-request increments independently confirm serialization.

This is not a CUDA OOM: the server remains healthy and all requests return HTTP 200. It is also not an HTTP throttle: there are no HTTP 429 responses. When the active sequence finishes, vLLM releases its blocks and admits the next waiting sequence.

## Expected variations and troubleshooting

- **Server refuses to start:** verify that the startup log reports at least 2,048 KV tokens. Keep the exact model, dtype, and cache bytes together.
- **Requests are rejected:** confirm `prompt_tokens + max_tokens <= max_model_len`.
- **Cache does not reach 100%:** confirm `max_tokens=282`, `ignore_eos=true`, and that the generated prompt metadata reports 1,750 tokenizer tokens.
- **More than one request runs:** verify the startup log says `GPU KV cache size: 2,048 tokens`; an ignored `--kv-cache-memory-bytes` changes the experiment.
- **Prefix cache hits are nonzero:** preserve the distinct request marker at the beginning of each prompt.
- **Docker cannot see the GPU:** validate NVIDIA Container Toolkit with `docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu24.04 nvidia-smi`.
- **OOM occurs:** that is a separate physical-allocation failure. Check other GPU processes and model/runtime memory; the intended experiment limits KV blocks without exhausting total VRAM.

## Re-run data preparation

Generated and downloaded data are intentionally excluded from Git. Recreate them deterministically with:

```bash
python download_benchmark.py
python prepare_prompt.py
```

`generated/prompt.json` records source size, title, question, tokenizer, and final prepared token count.
