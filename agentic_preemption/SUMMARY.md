# Summary: forcing a KV-cache preemption livelock

## Goal
Show what happens when a vLLM server is asked to run **more concurrent agentic-coding requests
than its KV cache can actually hold** — the opposite of the sibling throttle demo. We want the
cache to be *oversubscribed*, forced to throw away in-flight work, and to collapse toward "nothing
gets done."

## The design decisions (and why)

| Decision | What it does | Why |
|---|---|---|
| **Tiny KV cache** — `--num-gpu-blocks-override 10` (≈ **11,702 tokens ≈ 1.43 full sequences**) | Shrinks the cache to hold barely more than one full request | The knob's documented purpose is *"testing preemption."* A small, exact cache makes the oversubscription boundary obvious. |
| **Optimistic admission** — `--no-scheduler-reserve-full-isl` | Admit a request if its *prompt* fits **right now**, without checking whether its full length will fit later | This is what lets a 2nd/3rd request "begin running" before the cache discovers it can't sustain them. Without it, extras would just queue cleanly. |
| **Forced long outputs** — `ignore_eos: true`, `max_tokens: 4096` | Every request generates the full 4,096 tokens instead of stopping early at EOS | Guarantees each sequence grows to its worst case (~1,786 prompt + 4,096 output ≈ **5,900 tokens** of KV), maximizing pressure. |
| **More workers than fit** — 8 concurrent, fired together | 8 long requests compete for ~1.43 sequences of cache | Creates the permanent overcommit. |
| **Fixed patience** — 180 s client timeout, 120 s run | Requests are abandoned if they don't finish in 180 s | Captures the *user's* view: "it never finished." |

### The mechanism that makes it break
vLLM **doesn't reserve KV for `max_tokens`** — it allocates block-by-block *as tokens are
generated*. So at admission time a second request looks fine (the first has barely grown), and it
gets in. Only later, as both decode and their footprints grow past 11,702 tokens, does the cache
run out — and by then it's too late to un-admit cleanly, so the scheduler must **preempt**: free a
running sequence's blocks and re-queue it (recompute from scratch = its context is thrown away).
`--no-scheduler-reserve-full-isl` removes the guard that would have prevented this; `ignore_eos`
guarantees the growth that triggers it. This optimism is normally *good* (real requests stop early
at EOS) — we broke that assumption on purpose.

## The four recipes at a glance

All use the same tiny cache and 8 workers over 120 s. **A/B/C** run on the **over-admitting**
server (guard OFF, small 1,800-token prompt) and differ only by `max_tokens`/`timeout` — they all
thrash. **D** flips two things — the **admission guard ON** (`RESERVE_ISL=1`, a server restart)
and a **large 6,000-token prompt** — to get the *opposite*: clean queueing, no preemption.

| Recipe | server / prompt | `max_tokens` | `timeout` | started → completed | preemptions (in-run) | shows | results dir |
|---|---|---|---|---|---|---|---|
| **A** | over-admit / 1800 | 4096 | 180 s | 8 → **0** | +1 | goodput collapse — "nothing gets done" | `results/preempt_20260810_184406/` |
| **B** | over-admit / 1800 | 512 | 180 s | 9 → 4 | **+173** | preemption churn — context thrown away repeatedly | `results/preempt_20260810_190411/` |
| **C** | over-admit / 1800 | 4096 | 1200 s | 8 → 2 (at 535 s & 1034 s) | +1 | **livelock, not deadlock** — oldest slowly drains | `results/preempt_20260810_190936/` |
| **D** | **guard ON / 6000** | 1500 | 180 s | 11 → **8** | **0** | **clean queue (parent throttle)** — 1 running, 7 waiting, serial drain | `results/preempt_20260811_082636/` |

Pick **B** to watch the evictions, **C** to prove progress still happens, **A** for the starkest
"nothing finished," **D** for the well-behaved contrast (admission control, no thrash).

---

## Recipe A — goodput collapse ("nothing gets done")

**Results:** `results/preempt_20260810_184406/`
**Config:** 8 workers · 120 s · `max_tokens=4096` · `timeout=180 s`

- **0 of 8 requests completed.** All 8 ran until the 180 s client timeout and were abandoned →
  **goodput 0/min.**
- **Cache pinned full:** KV usage peaks at 100%, averages ~79%, oscillating **100% ↔ 56%**.
- **Only ~2 run at once:** `running` flips between 1 and 2 (never more); the other **6–7 sit
  queued** the entire time.
- **Context thrown away:** a preemption fires early (`num_preemptions_total` → 1) — a running
  run's KV is discarded and re-queued.
- **Livelock flagged for 56.7 s** of the 120 s run (KV saturated + zero completions).

**The picture:** two ~5,900-token sequences can't both fit in an 11,702-token cache, so it
endlessly admits, saturates, drops one, and re-admits — the 100%↔56% thrash — while the queue
never drains. Under that constant disruption, not even the oldest request finishes 4,096 tokens
before its 180 s deadline. From the client's side, **nothing ever gets done.**

---

## Recipe B — smaller outputs expose the preemption churn

**Results:** `results/preempt_20260810_190411/`
**Config:** 8 workers · 120 s · **`max_tokens=512`** · `timeout=180 s`

Same tiny cache, but each run generates only 512 tokens, so requests finish faster and **new ones
keep arriving** — the admit→evict cycle repeats constantly:

- **Preemptions explode: +173 during the run** (vs. just +1 in Recipe A). This is the literal
  "throw away a run's context" happening over and over.
- **Turnover:** 9 requests started (more than 8 workers — some finished and were relaunched);
  **4 completed**, 5 timed out. Goodput is nonzero but tiny (2/min).
- Even a 512-token request took **85–177 s** to finish under the contention.
- KV still pinned (avg 79%, peak 100%), livelock flagged 52 s.

**Takeaway:** shorter outputs trade "nothing completes" for "constant preemption" — the cache
churns visibly (173 evictions) while goodput stays crippled.

---

## Recipe C — long patience proves it's a livelock, not a hang

**Results:** `results/preempt_20260810_190936/`
**Config:** 8 workers · 120 s · `max_tokens=4096` · **`timeout=1200 s`**

Identical to Recipe A except clients wait up to 20 minutes instead of 180 s:

- **Requests do eventually finish** — 2 completed, at **535 s and 1,034 s** (≈9 and ≈17 minutes)
  — draining **one at a time, oldest-first**, exactly as vLLM's forward-progress guarantee
  predicts. The other 6 ran out of even the 1,200 s patience.
- Barely any preemption during the run (+1): with long outputs there's no turnover, so it's the
  same queue-and-stall regime as Recipe A — just observed for longer.

**Takeaway:** this is the proof that it's a **livelock, not a true deadlock.** The system never
fully stops; the oldest request grinds to completion (17 minutes for what normally takes ~30–60 s)
while everything newer starves. Recipe A only *looked* total because its 180 s patience cut off
before any request could drain.

---

## Recipe D — clean queueing (the parent throttle, no preemption)

**Results:** `results/preempt_20260811_082636/`
**Server:** `RESERVE_ISL=1` (admission guard ON) · same tiny KV (11,702 tokens)
**Config:** 8 workers · 120 s · **`SEED_TOKENS=6000`** (large prompt) · `MAX_TOKENS=1500` · `timeout=180 s`

The mirror image of A/B/C. Two changes flip thrash into clean admission control — they do
**different jobs**:
1. **Large prompt** (~6,000 tokens ≈ 3 of the ~5.5 blocks) is the *physical reason* two can't
   coexist: 2 × 3 blocks = 6 > 5.58, so **only one fits at a time**. (True regardless of any flag.)
2. **Admission guard ON** (`--no-scheduler-reserve-full-isl` removed) decides what the scheduler
   *does about that* when request 2 reaches the front: it must check the **whole** prompt fits
   before admitting → it doesn't (only ~1.5 blocks free) → request 2 **waits cleanly**. With the
   guard OFF and chunked prefill on, the scheduler checks only request 2's **first chunk** (~1
   block), which *does* fit → it admits it → then both grow → preemption. So the guard doesn't
   make "only one fit" — the prompt size does; the guard makes the scheduler **refuse cleanly
   instead of admitting-then-preempting.**

(If the prompt were small enough to prefill in a *single* chunk, "check first chunk" == "check
whole prompt" and the guard would be redundant — the large prompt alone would suffice. This
server has chunked prefill ON, so the guard is the insurance that closes that loophole.)

What we observed — exactly the parent's behavior:
- **0 preemptions** the entire run (nothing's context ever thrown away).
- **1 running, 7 waiting** for essentially every sample (running=1 in 460/465 samples).
- **11 started → 8 completed** (`finish_reason=length`), **goodput 4/min** (vs 0 in Recipe A).
- **Serial drain:** completion latencies **36 s, 70 s, 105 s, 139 s, 173 s** — rising in ~34 s
  steps, i.e. requests finishing **one at a time** (the parent's "completion times increase in
  one-request increments").
- KV steady at ~74% (one request resident), never oversubscribed; watchdog: **no livelock**.

### Why 3 of the 11 "failed"
The 3 failures (`w0`, `w3`, `w5` in `requests.jsonl`) are **not errors** — they are client-side
**180 s timeouts on requests that were still WAITING in the queue**, never admitted:
`start≈0.00 s`, `latency=180.2 s`, `finish=None`, `TimeoutError('timed out')` — they never even
started generating.

It's a direct consequence of clean queueing: one request runs at a time and each takes ~34 s, so
the FCFS queue clears one slot every ~34 s — and **only ~5 fit inside a 180 s patience window**:

| completes at | 36 s | 70 s | 105 s | 139 s | 173 s | ~207 s | ~241 s | ~275 s |
|---|---|---|---|---|---|---|---|---|
| request | w4 | w1 | w7 | w6 | w2 | **w0 ✗** | **w3 ✗** | **w5 ✗** |

`w0/w3/w5` sat in positions 6–8; their turn would have come at ~207/241/275 s, but the client gave
up at 180 s. **Same failure *type* as A/B/C (a client `TIMEOUT`), opposite *cause*:** here it's
honest **queue latency**, not thrash — `num_preemptions_total = 0`, **zero wasted work**. Raise
`TIMEOUT` (as in Recipe C's `1200 s`) and all 8 drain cleanly in order.

**Takeaway:** this is scheduler **admission control**, not throttling-by-thrash. Because each
request nearly fills the cache and the guard refuses to admit a second until the first frees its
blocks, requests execute **serially with zero wasted work** — the clean counterpart to the
preemption livelock.

**Can this be done with just `WORKERS/DURATION/MAX_TOKENS/TIMEOUT`?** No. `MAX_TOKENS` is *output*,
and vLLM reserves no output KV at admission, so those four can't gate admission. Clean queueing
needs a **large prompt** (`SEED_TOKENS`) *and* the **guard ON** (`RESERVE_ISL=1`, a server flag) —
so a **server restart** is required.

---

See [`DETAILS.md`](./DETAILS.md) for the full mechanics and [`README.md`](./README.md) for how to
run each recipe.
