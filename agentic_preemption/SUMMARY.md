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

## The three recipes at a glance

All share the same tiny cache and 8 workers over 120 s; only `max_tokens` and `timeout` change.
Across all three the cache stays pinned (**~79% avg, 100% peak**) with only **~1–2 running and
6–7 queued** — what changes is the *symptom* each surfaces.

| Recipe | `max_tokens` | `timeout` | started → completed | preemptions (in-run) | shows | results dir |
|---|---|---|---|---|---|---|
| **A** | 4096 | 180 s | 8 → **0** | +1 | goodput collapse — "nothing gets done" | `results/preempt_20260810_184406/` |
| **B** | 512 | 180 s | 9 → 4 | **+173** | preemption churn — context thrown away repeatedly | `results/preempt_20260810_190411/` |
| **C** | 4096 | 1200 s | 8 → 2 (at 535 s & 1034 s) | +1 | **livelock, not deadlock** — oldest slowly drains | `results/preempt_20260810_190936/` |

Pick **B** to watch the evictions, **C** to prove progress still happens, **A** for the starkest
"nothing finished."

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

See [`DETAILS.md`](./DETAILS.md) for the full mechanics and [`README.md`](./README.md) for how to
run each recipe.
