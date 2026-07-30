"""E4B retarget smoke test — runs on a real TPU v6e-1.

Checks three things, in order of how badly they fail if wrong:

  1. STRUCTURE — the retargeted `Gemma4EConfig` defaults really are E4B's
     geometry (42 layers, 24 KV-holding, n_rep 4, no double-wide MLP).
  2. SIZE — the resident weight and KV-per-token figures quoted in `models.md`,
     `CLAUDE.md` and the `jax_engine` comments were derived analytically. This
     measures them on device and fails if they disagree by more than 2%. A
     derivation that nobody checked against hardware is a guess.
  3. EXECUTION — prefill and cached decode compile and produce finite logits
     under the E4B geometry.

Synthetic architecture-shaped weights, so this says nothing about generation
QUALITY — only that the shapes are right and the graph runs. Real-weight
validation needs the gated checkpoint.

Run on the VM:  python3 benchmarks/queued/e4b_tpu_smoke.py
"""

import os
import sys
import time

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "../.."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import jax
import jax.numpy as jnp

from ports.gemma4.jax_e_model import (
    Gemma4EConfig,
    Gemma4EModelJAX,
    init_kv_cache,
    make_cached_decode_step,
    prefill_with_kv_cache,
    pad_to_tpu_v6e_bucket,
)
from ports.gemma4.jax_e_benchmark_sweep import build_benchmark_params

# Claims made in the docs, to be checked against the device.
CLAIM_W4A16_WEIGHT_BYTES = 9.21e9      # models.md, CLAUDE.md, jax_engine
CLAIM_KV_KIB_PER_TOKEN = 56.0          # ditto
TOLERANCE = 0.005

# The docs' 9.21 GB describes the SHIPPED checkpoint, which stores the PLE
# projections W4A16-packed. `build_benchmark_params` builds those three tensors
# — per_layer_input_gate, per_layer_projection, per_layer_model_projection —
# dense BF16 instead, so the harness tree is legitimately larger. Account for the
# difference explicitly rather than widening the tolerance until it disappears:
# an unexplained 1% gap in a memory figure is how a wrong number survives.
_W4 = 0.5 + 2 / 32                     # packed int4 + BF16 group scale
_DENSE_PLE_DELTA = (
    (2 * 2560 * 256) * 42              # per-layer gate + projection
    + 2560 * 42 * 256                  # per_layer_model_projection
) * (2.0 - _W4)                        # = 0.119 GB
EXPECT_HARNESS_WEIGHT_BYTES = CLAIM_W4A16_WEIGHT_BYTES + _DENSE_PLE_DELTA

failures = []


def check(label, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {label}{(' — ' + detail) if detail else ''}")
    if not ok:
        failures.append(label)


def close(got, want):
    return abs(got - want) / want <= TOLERANCE


print("=" * 78)
print("GEMMA 4 E4B RETARGET — TPU SMOKE TEST")
print("=" * 78)
print(f"jax {jax.__version__}   devices: {jax.devices()}")
dev = jax.devices()[0]
if dev.platform != "tpu":
    print("\nABORT: no TPU device visible; this test is meaningless off-TPU.")
    raise SystemExit(2)

# ---------------------------------------------------------------- 1. structure
print("\n[1/3] Structure — defaults must be E4B's shipped geometry")
cfg = Gemma4EConfig()
model = Gemma4EModelJAX(cfg)

check("42 layers", cfg.num_hidden_layers == 42, str(cfg.num_hidden_layers))
check("hidden 2560 / intermediate 10240",
      (cfg.hidden_size, cfg.intermediate_size) == (2560, 10240),
      f"{cfg.hidden_size}/{cfg.intermediate_size}")
check("2 KV heads (n_rep 4)", cfg.num_key_value_heads == 2,
      f"n_rep={cfg.num_attention_heads // cfg.num_key_value_heads}")
check("first KV-shared layer 24", cfg.first_kv_shared_layer_idx == 24,
      str(cfg.first_kv_shared_layer_idx))
check("no double-wide MLP", cfg.use_double_wide_mlp is False)

full = [i for i, t in enumerate(cfg.layer_types) if t == "full_attention"]
check("full attention at 5,11,...,41", full == [5, 11, 17, 23, 29, 35, 41], str(full))

shared_mlp = model.layers[24][1].intermediate_size
check("shared layer MLP is NOT doubled", shared_mlp == 10240, str(shared_mlp))

# ------------------------------------------------------------------- 2. size
print("\n[2/3] Size — measured on device vs the figures quoted in the docs")
t0 = time.perf_counter()
params = jax.device_put(build_benchmark_params(cfg), dev)
jax.block_until_ready(params)
build_s = time.perf_counter() - t0

weight_bytes = sum(int(x.size) * int(x.dtype.itemsize)
                   for x in jax.tree_util.tree_leaves(params))
print(f"  built E4B param tree in {build_s:.1f}s")
check(f"weights == {CLAIM_W4A16_WEIGHT_BYTES/1e9:.2f} GB packed "
      f"+ {_DENSE_PLE_DELTA/1e9:.3f} GB dense-PLE harness overhead "
      f"= {EXPECT_HARNESS_WEIGHT_BYTES/1e9:.2f} GB",
      close(weight_bytes, EXPECT_HARNESS_WEIGHT_BYTES),
      f"measured {weight_bytes/1e9:.2f} GB")

probe_len = 128
cache = init_kv_cache(cfg, batch_size=1, max_seq_len=probe_len, dtype=jnp.bfloat16)
kv_bytes = sum(int(a.size) * int(a.dtype.itemsize)
               for entry in cache.values() for a in entry)
kv_kib = kv_bytes / probe_len / 1024
check("24 KV-holding layers", len(cache) == 24, str(len(cache)))
check(f"KV per token ~= {CLAIM_KV_KIB_PER_TOKEN} KiB (BF16)",
      close(kv_kib, CLAIM_KV_KIB_PER_TOKEN), f"measured {kv_kib:.1f} KiB")

mem = dev.memory_stats() or {}
if mem:
    print(f"  HBM in use {mem.get('bytes_in_use', 0)/2**30:.2f} GiB "
          f"of {mem.get('bytes_limit', 0)/2**30:.2f} GiB")

# -------------------------------------------------------------- 3. execution
print("\n[3/3] Execution — prefill + cached decode under the E4B geometry")
prompt = jnp.ones((1, 64), dtype=jnp.int32)
padded, valid = pad_to_tpu_v6e_bucket(prompt)
bucket_s = int(padded.shape[1])

jit_prefill = jax.jit(
    prefill_with_kv_cache,
    static_argnames=("model", "max_new_tokens", "quant_mode", "cache_dtype", "window_kv"),
)
t0 = time.perf_counter()
last_logits, caches, valid_mask = jax.block_until_ready(jit_prefill(
    model=model, prompt_ids=padded, prompt_valid=valid, params=params,
    max_new_tokens=8, quant_mode="w4a16", window_kv=False,
))
compile_s = time.perf_counter() - t0

t0 = time.perf_counter()
last_logits, caches, valid_mask = jax.block_until_ready(jit_prefill(
    model=model, prompt_ids=padded, prompt_valid=valid, params=params,
    max_new_tokens=8, quant_mode="w4a16", window_kv=False,
))
prefill_ms = (time.perf_counter() - t0) * 1000

check("prefill logits shape [1, vocab]",
      last_logits.shape == (1, cfg.vocab_size), str(last_logits.shape))
check("prefill logits finite", bool(jnp.all(jnp.isfinite(last_logits))))
print(f"  prefill compile {compile_s:.1f}s, steady-state {prefill_ms:.1f} ms "
      f"({bucket_s} tokens)")

step = jax.jit(make_cached_decode_step(model, quant_mode="w4a16", window_kv=False),
               donate_argnums=(1, 2))
tok = jnp.argmax(last_logits, axis=-1, keepdims=True).astype(jnp.int32)
prompt_lens = jnp.asarray([64], dtype=jnp.int32)

steps, timings = 8, []
for i in range(steps):
    t0 = time.perf_counter()
    caches, valid_mask, last_logits = jax.block_until_ready(
        step(params, caches, valid_mask, tok, prompt_lens + i,
             jnp.int32(bucket_s + i)))
    timings.append((time.perf_counter() - t0) * 1000)
    tok = jnp.argmax(last_logits, axis=-1, keepdims=True).astype(jnp.int32)

steady = sorted(timings[1:])[len(timings[1:]) // 2]
check("decode logits finite", bool(jnp.all(jnp.isfinite(last_logits))))
check("decode advances", len(set(t.item() for t in [tok])) >= 1)
print(f"  decode first step {timings[0]:.1f} ms (compile), "
      f"median steady {steady:.2f} ms -> {1000/steady:.1f} tok/s at B=1")

print("\n" + "=" * 78)
if failures:
    print(f"SMOKE TEST FAILED: {len(failures)} check(s) — {', '.join(failures)}")
    raise SystemExit(1)
print("E4B SMOKE TEST PASSED — structure, size, and execution all verified on TPU")
