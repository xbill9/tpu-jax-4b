# 🎛️ Gemma 4 Model Flags & Configuration Guide (`models.md`)

This guide provides a comprehensive reference for all model flags, CLI arguments, data types, quantization schemes, and supported checkpoints when deploying **Gemma 4** on **Google Cloud TPUs** using **JAX**. The default target is **E4B**; E2B remains supported through the same code path.

---

## 📌 Supported Gemma 4 Model Checkpoints

| Model ID | Effective Parameters | Quantization Scheme | Format | Target TPU Hardware | Recommended Use Case |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`google/gemma-4-E4B-it-qat-w4a16-ct`** | **4.5B** (8.0B total) | QAT W4A16 | `compressed-tensors` | TPU v6e-1 (32 GB HBM) | Mid-sized multimodal serving (**Default**) |
| **`google/gemma-4-E4B-it-qat-q4_0-unquantized`** | **4.5B** (8.0B total) | QAT Q4_0 Baseline | Unquantized `bfloat16` | TPU v6e-1 (32 GB HBM) | Maximum QAT generation quality at 4B |
| **`google/gemma-4-E4B-it-qat-mobile-transformers`** | **4.5B** (8.0B total) | QAT Mixed (W4/W2, A8, KV8) | `transformers` | Edge / Mobile / TPU | Low-power mobile & edge deployment |
| **`google/gemma-4-E4B-it`** | **4.5B** (8.0B total) | Unquantized | Full `bfloat16` | TPU v6e-1 (32 GB HBM) | Baseline non-QAT model; the variant vLLM can serve |
| **`google/gemma-4-E2B-it-qat-w4a16-ct`** | **2.3B** (5.1B total) | QAT W4A16 | `compressed-tensors` | TPU v6e-1 (32 GB HBM) | Smaller alternative; lowest disk/load footprint |
| **`google/gemma-4-E2B-it`** | **2.3B** (5.1B total) | Unquantized | Full `bfloat16` | TPU v6e-1 (32 GB HBM) | Baseline non-QAT 2B model |
| **`google/gemma-4-31B-it`** | **31.0B** | Unquantized / QAT | `bfloat16` / `W4A16` | TPU v6e-8 / v5p-8 | Production multi-chip TPU pod serving |

### E4B vs E2B: what actually differs

E4B is not E2B with bigger numbers — six config fields change, and two of them
change code paths rather than sizes. Read these off `config.json`; do not infer
them (`jax_engine.config_from_hf` does exactly that).

| Field | E2B | E4B |
| :--- | ---: | ---: |
| `hidden_size` | 1,536 | 2,560 |
| `intermediate_size` | 6,144 | 10,240 |
| `num_hidden_layers` | 35 | 42 |
| `num_key_value_heads` | 1 | 2 |
| `num_kv_shared_layers` | 20 | 18 |
| `use_double_wide_mlp` | `true` | `false` |
| Full-attention layers | every 5th (`i % 5 == 4`) | every 6th (`i % 6 == 5`) |
| KV-holding layers | 15 | 24 |
| GQA `n_rep` | 8 | 4 |

Consequences for a v6e-1 (32 GB HBM), derived from the above:

| Quantity | E2B | E4B |
| :--- | ---: | ---: |
| W4A16 resident weights | 6.56 GB | 9.21 GB |
| — of which the PLE table | 4.70 GB (72%) | 5.64 GB (61%) |
| Dense BF16 resident weights | 9.26 GB | 14.92 GB |
| Safetensors on disk | 8.32 GB | 11.51 GB |
| KV per token (BF16) | 18.0 KiB | 56.0 KiB |
| HBM left for KV after W4A16 weights | 25.1 GiB | 22.7 GiB |

The KV row is the one that bites: E4B costs **3.11× more KV per token** and has
slightly less headroom to put it in, so concurrent-sequence capacity falls by
more than the weight growth alone suggests. `--kv-cache-dtype int8` halves it.

---

## 🎛️ Complete Model & CLI Flags Reference

### 1. Model Selection & Precision Flags

| Flag | Type | Default Value | Valid Options | Description |
| :--- | :--- | :--- | :--- | :--- |
| **`--model`** | `str` | `google/gemma-4-E4B-it-qat-w4a16-ct` | Any HF Repo ID or local path | Specifies the model checkpoint to load into JAX TPU memory. |
| **`--kv-cache-dtype`** | `str` | `fp8` | `fp8` (`fp8_e4m3fn`), `bfloat16`, `int8` | Data type for Key-Value attention cache in TPU HBM. `fp8` cuts KV memory usage by 50%. |
| **`--torch-dtype`** / **`--dtype`** | `str` | `bfloat16` | `bfloat16`, `float32` | Precision for intermediate activations and MXU matrix multiplications. |
| **`--max-new-tokens`** | `int` | `128` | Positive integers (`1` to `8192`) | Maximum number of new tokens generated per inference request. |
| **`--temperature`** | `float` | `0.7` | `0.0` to `2.0` | Sampling temperature. `0.0` enables deterministic greedy decoding. |
| **`--top-p`** | `float` | `0.95` | `0.0` to `1.0` | Nucleus sampling probability threshold. |
| **`--top-k`** | `int` | `50` | Positive integers (e.g. `40`, `50`) | Top-K sampling token candidate pool size. |

---

### 2. Server & Network Execution Flags

| Flag | Type | Default Value | Example Values | Description |
| :--- | :--- | :--- | :--- | :--- |
| **`--host`** | `str` | `0.0.0.0` | `0.0.0.0`, `127.0.0.1` | Network interface IP address for the FastAPI / Uvicorn server. |
| **`--port`** | `int` | `8000` | `8000`, `8080` | HTTP port for OpenAI API (`/v1/chat/completions`) & Prometheus (`/metrics`). |

---

## 🔬 Quantization Precision Matrix

```
+---------------------------------------------------------------------------------------+
|                                    Gemma 4 QAT Model                                  |
+-------------------+-------------------+-------------------+---------------------------+
| Layer Component   | Storage Precision | Execution Dtype   | Scaling & Metadata        |
+-------------------+-------------------+-------------------+---------------------------+
| Model Weights     | INT4 (4-bit)      | bfloat16          | group_size = 32           |
| Layer Scale/Zero  | bfloat16          | bfloat16          | Per-channel / per-group   |
| Layer Biases      | bfloat16          | bfloat16          | Unquantized float16       |
| Activations       | bfloat16          | bfloat16          | MXU native execution      |
| KV Cache          | FP8 (e4m3fn)      | FP8 / bfloat16    | 50% HBM reduction         |
+-------------------+-------------------+-------------------+---------------------------+
```

---

## 🚀 Execution Examples

### 1. Launch OpenAI-Compatible API Server with W4A16 QAT & FP8 KV Cache
```bash
python3 jax_openai_server.py \
  --model google/gemma-4-E4B-it-qat-w4a16-ct \
  --kv-cache-dtype fp8 \
  --host 0.0.0.0 \
  --port 8000
```

### 2. Run the Prefill / Decode Benchmark Sweep
```bash
python3 ports/gemma4/jax_e_benchmark_sweep_v2.py \
  --arch e4b \
  --batch-sizes 1,2,4,8,16,32 \
  --contexts 8,128,512,2048 \
  --json-out results.json
```

This runs on architecture-shaped synthetic weights, so it measures kernel cost,
not generation quality. For real weights, serve the checkpoint with
`jax_openai_server.py` and drive it over HTTP.

> A single-shot CLI runner (`jax_gemma4_e2b.py`) is referenced by older revisions
> of this guide but is not present in the repository. The server and the sweep
> above are the supported entry points.
