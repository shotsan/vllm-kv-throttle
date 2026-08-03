# Live throttling proof

Captured on an RTX 4060 Ti with vLLM 0.11.2.

## Server capacity evidence

```text
kv_cache_memory_bytes: 46137344
GPU KV cache size: 2,048 tokens
Maximum concurrency for 2,048 tokens per request: 1.00x
```

## Runtime scheduler evidence

Four requests were submitted concurrently. Representative samples from `proof_test.py`:

```text
0.168s  running=1  waiting=3  KV=86.61%
2.435s  running=1  waiting=3  KV=99.21%
2.541s  running=1  waiting=3  KV=100.00%
2.698s  running=0  waiting=3  KV=0.00%   first request released cache
2.804s  running=1  waiting=2  KV=86.61%  second request admitted
5.178s  running=1  waiting=2  KV=100.00%
5.442s  running=1  waiting=1  KV=86.61%  third request admitted
7.822s  running=1  waiting=1  KV=100.00%
8.033s  running=1  waiting=0  KV=86.61%  fourth request admitted
10.473s running=1  waiting=0  KV=100.00%
```

## Client completion evidence

```text
request 0: HTTP 200, 2.670 seconds, 1,751 prompt + 282 completion tokens
request 2: HTTP 200, 5.291 seconds, 1,751 prompt + 282 completion tokens
request 3: HTTP 200, 7.921 seconds, 1,751 prompt + 282 completion tokens
request 1: HTTP 200, 10.564 seconds, 1,751 prompt + 282 completion tokens
```

This is scheduler throttling rather than HTTP rate limiting: all requests were accepted, but only one could run. The other three stayed in vLLM's waiting queue and completed one after another as KV blocks were released.
