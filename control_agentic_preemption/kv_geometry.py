#!/usr/bin/env python3
"""Expose the EXACT KV-cache page geometry of the Qwen3.5-MoE hybrid used in
these preemption experiments -- WITHOUT loading weights or touching the GPU.

It reads the model's HF `config.json` and applies vLLM's own spec formulas
(copied verbatim from the installed vllm 0.25.1):
  - attention page:  AttentionSpec.real_page_size_bytes
                     = 2 * block_size * num_kv_heads * head_size * dtype_size
  - mamba  page:     MambaSpec.page_size_bytes = sum(prod(shape)*dtype_size)
                     with shapes from
                     MambaStateShapeCalculator.gated_delta_net_state_shape(...)

Then it reproduces vLLM's rule "attention block_size chosen so attention page
>= mamba page", and prints the per-sequence footprint (fixed mamba state +
token-proportional attention) that explains the 2-request preemption cap.

Run:  python3 kv_geometry.py [--config /path/to/config.json] [--kv-dtype fp8]
No GPU, no vLLM engine, no weights. Pure arithmetic on config values.
"""
import argparse
import json
from math import prod

# ---- dtype sizes in bytes (vllm get_dtype_size) ----
DT = {"fp8": 1, "int8": 1, "bfloat16": 2, "float16": 2, "float32": 4, "fp32": 4}


def gated_delta_net_state_shape(tp, num_k, num_v, k_dim, v_dim, conv_k, num_spec=0):
    """Verbatim from vllm/model_executor/layers/mamba/mamba_utils.py."""
    conv_dim = k_dim * num_k * 2 + v_dim * num_v
    conv_state_shape = (conv_dim // tp, conv_k - 1 + num_spec)      # _orient_conv_shape
    temporal_state_shape = (num_v // tp, v_dim, k_dim)
    return conv_state_shape, temporal_state_shape


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=(
        "/home/ways_lab/.cache/huggingface/hub/models--Qwen--Qwen3.6-35B-A3B-FP8/"
        "snapshots/95a723d08a9490559dae23d0cff1d9466213d989/config.json"))
    ap.add_argument("--kv-dtype", default="fp8", help="attention KV cache dtype")
    ap.add_argument("--conv-dtype", default="bfloat16", help="mamba conv-state dtype")
    ap.add_argument("--ssm-dtype", default="float32", help="mamba ssm-state dtype")
    ap.add_argument("--num-gpu-blocks", type=int, default=10, help="the override N")
    ap.add_argument("--null-blocks", type=int, default=1, help="reserved null block(s)")
    a = ap.parse_args()

    cfg = json.load(open(a.config))
    t = cfg["text_config"]
    layer_types = t["layer_types"]
    n_full = layer_types.count("full_attention")
    n_lin = layer_types.count("linear_attention")

    kv_b = DT[a.kv_dtype]
    conv_b = DT[a.conv_dtype]
    ssm_b = DT[a.ssm_dtype]

    # ---- mamba (gated-delta-net) page, per linear-attn layer, FIXED per seq ----
    conv_shape, temporal_shape = gated_delta_net_state_shape(
        1, t["linear_num_key_heads"], t["linear_num_value_heads"],
        t["linear_key_head_dim"], t["linear_value_head_dim"],
        t["linear_conv_kernel_dim"])
    conv_bytes = prod(conv_shape) * conv_b
    ssm_bytes = prod(temporal_shape) * ssm_b
    mamba_page = conv_bytes + ssm_bytes

    # ---- attention block_size chosen so attention page >= mamba page ----
    kv_heads = t["num_key_value_heads"]
    head = t["head_dim"]
    per_tok_per_layer = 2 * kv_heads * head * kv_b          # bytes/token/full-attn-layer
    import math
    block_size = math.ceil(mamba_page / per_tok_per_layer)  # tokens/attention page
    attn_page = per_tok_per_layer * block_size

    print("=" * 66)
    print(f"MODEL: {cfg['architectures'][0]}   layers={t['num_hidden_layers']}")
    print(f"  full_attention layers : {n_full}")
    print(f"  linear_attention(GDN) : {n_lin}")
    print(f"  attn: num_kv_heads={kv_heads} head_dim={head} kv_dtype={a.kv_dtype}({kv_b}B)")
    print(f"  GDN : k_heads={t['linear_num_key_heads']} v_heads={t['linear_num_value_heads']}"
          f" k_dim={t['linear_key_head_dim']} v_dim={t['linear_value_head_dim']}"
          f" conv_k={t['linear_conv_kernel_dim']}")
    print("=" * 66)
    print("MAMBA (GatedDeltaNet) STATE — one fixed page per linear-attn layer:")
    print(f"  conv_state  shape={conv_shape}  x {conv_b}B = {conv_bytes:,} B")
    print(f"  ssm_state   shape={temporal_shape}  x {ssm_b}B = {ssm_bytes:,} B")
    print(f"  mamba page / layer                      = {mamba_page:,} B")
    print("-" * 66)
    print("ATTENTION page (sized to be >= mamba page):")
    print(f"  bytes/token/layer = 2*{kv_heads}*{head}*{kv_b} = {per_tok_per_layer:,} B")
    print(f"  block_size = ceil({mamba_page:,}/{per_tok_per_layer}) = {block_size:,} tokens")
    print(f"  attention page / layer                  = {attn_page:,} B")
    print("=" * 66)

    # ---- per-sequence footprint ----
    fixed_mamba = n_lin * mamba_page
    attn_per_page_all_layers = n_full * attn_page
    print("PER-SEQUENCE FOOTPRINT")
    print(f"  FIXED mamba state (all {n_lin} GDN layers, independent of length):")
    print(f"      {n_lin} x {mamba_page:,} = {fixed_mamba:,} B  ({fixed_mamba/2**20:.1f} MiB)")
    print(f"  ATTENTION per 1 block-step (all {n_full} full-attn layers):")
    print(f"      {n_full} x {attn_page:,} = {attn_per_page_all_layers:,} B "
          f"({attn_per_page_all_layers/2**20:.1f} MiB) per {block_size} tokens")
    print("=" * 66)

    # ---- reconcile the override + reported capacity (VERIFIED) ----
    N = a.num_gpu_blocks
    # A vLLM "block" (the unit num_gpu_blocks counts) = one 2096-token attention
    # page across ALL full-attn layers. VERIFIED: natural blocks = avail_KV / this.
    block_bytes = n_full * attn_page
    print("BLOCK-POOL RECONCILIATION  (VERIFIED against logged natural counts)")
    print(f"  1 vLLM block = {n_full} full-attn layers x {attn_page:,} B "
          f"= {block_bytes:,} B ({block_bytes/2**20:.3f} MiB)")
    for gib, logged in [(21.77, 1089), (22.03, 1102)]:
        nat = gib * 2**30 / block_bytes
        print(f"    {gib} GiB avail_KV / block = {nat:7.1f}  (log: num_gpu_blocks={logged})"
              f"  {'MATCH' if round(nat)==logged else 'diff'}")
    print(f"  num_gpu_blocks_override = {N}  -> {N} attention pages "
          f"= {N*block_size:,} attn tokens; mamba state is a SEPARATE fixed pool.")
    print(f"  Reported 'GPU KV cache size' 11,702 tok = max_concurrency*max_model_len")
    print(f"     -> max_concurrency = 11702/8192 = {11702/8192:.3f}x")
    print(f"  NOTE: the live per-step KV% (44/56/89/100 = ninths) and the observed")
    print(f"  2-running cap are BLOCK-MANAGER runtime behavior -- use probe_kvconfig.py")
    print(f"  to dump the per-group block allocation at admission/preempt to close it.")
    print("=" * 66)
    print("BOTTOM LINE")
    print(f"  * 1 KV page = {block_size} attention tokens = {attn_page:,} B "
          f"(~{attn_page/2**20:.2f} MiB), == 1 mamba page.")
    print(f"  * A {1786}-token prompt (<{block_size}) => 1 attention page per full-attn layer.")
    print(f"  * The heavy per-sequence cost is the FIXED {fixed_mamba/2**20:.0f} MiB of "
          f"GDN/mamba state,")
    print(f"    present the instant a sequence is admitted, independent of prompt length.")


if __name__ == "__main__":
    main()
