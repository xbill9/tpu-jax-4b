"""End-to-end test for JaxGemmaEngine against a synthetic safetensors checkpoint.

Writes a tiny Gemma4E-shaped checkpoint to a temp dir, loads it through the real
loader path (safetensors -> convert_safetensors_to_jax_params -> device_put),
and exercises streaming generation, EOS handling, and length capping.

No network, no PyTorch, no TPU required.

Run: python3 -m unittest tests.test_jax_engine
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from jax_engine import GenerationStats, JaxGemmaEngine, config_from_hf  # noqa: E402
from ports.gemma4.jax_e_model import Gemma4EConfig  # noqa: E402

TINY_HF_CONFIG = {
    "vocab_size": 256,
    "hidden_size": 64,
    "intermediate_size": 96,
    "num_hidden_layers": 10,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 16,
    "num_global_key_value_heads": 2,
    "global_head_dim": 32,
    "num_kv_shared_layers": 4,
    "hidden_size_per_layer_input": 16,
    "vocab_size_per_layer_input": 256,
    "rms_norm_eps": 1e-6,
    "final_logit_softcapping": 30.0,
}


def write_tiny_checkpoint(path: Path, seed: int = 0) -> Gemma4EConfig:
    """Write config.json + model.safetensors in Hugging Face key/layout convention."""
    from safetensors.flax import save_file

    cfg = config_from_hf(TINY_HF_CONFIG)
    rng = np.random.default_rng(seed)

    def w(*shape):
        # HF stores dense Linear weights as [out, in]; the loader transposes.
        return jnp.asarray(rng.normal(0, 0.05, size=shape), dtype=jnp.bfloat16)

    def ones(n):
        return jnp.ones((n,), dtype=jnp.bfloat16)

    L, H, ple = cfg.num_hidden_layers, cfg.hidden_size, cfg.hidden_size_per_layer_input
    tensors = {
        "model.embed_tokens.weight": w(cfg.vocab_size, H),
        "model.norm.weight": ones(H),
        "model.embed_tokens_per_layer.weight": w(cfg.vocab_size_per_layer_input, L * ple),
        "model.per_layer_model_projection.weight": w(L * ple, H),
        "model.per_layer_projection_norm.weight": ones(ple),
    }
    for i in range(L):
        is_sliding = cfg.layer_types[i] == "sliding_attention"
        h_dim = cfg.head_dim if is_sliding else cfg.global_head_dim
        num_kv = cfg.num_key_value_heads if is_sliding else cfg.num_global_key_value_heads
        is_shared = i >= cfg.first_kv_shared_layer_idx
        inter = cfg.intermediate_size * 2 if (is_shared and cfg.use_double_wide_mlp) else cfg.intermediate_size
        p = f"model.layers.{i}"

        tensors[f"{p}.input_layernorm.weight"] = ones(H)
        tensors[f"{p}.post_attention_layernorm.weight"] = ones(H)
        tensors[f"{p}.per_layer_input_gate.weight"] = w(ple, H)
        tensors[f"{p}.per_layer_projection.weight"] = w(H, ple)
        tensors[f"{p}.post_per_layer_input_norm.weight"] = ones(H)

        tensors[f"{p}.self_attn.q_proj.weight"] = w(cfg.num_attention_heads * h_dim, H)
        tensors[f"{p}.self_attn.o_proj.weight"] = w(H, cfg.num_attention_heads * h_dim)
        tensors[f"{p}.self_attn.q_norm.weight"] = ones(h_dim)
        if not is_shared:
            # KV-shared layers legitimately omit k/v projections and k_norm —
            # the omission that upstream's loader mishandles (tpu-inference #3225).
            tensors[f"{p}.self_attn.k_proj.weight"] = w(num_kv * h_dim, H)
            tensors[f"{p}.self_attn.v_proj.weight"] = w(num_kv * h_dim, H)
            tensors[f"{p}.self_attn.k_norm.weight"] = ones(h_dim)

        tensors[f"{p}.mlp.gate_proj.weight"] = w(inter, H)
        tensors[f"{p}.mlp.up_proj.weight"] = w(inter, H)
        tensors[f"{p}.mlp.down_proj.weight"] = w(H, inter)

    path.mkdir(parents=True, exist_ok=True)
    with open(path / "config.json", "w") as fh:
        json.dump(TINY_HF_CONFIG, fh)
    save_file(tensors, str(path / "model.safetensors"))
    return cfg


# Transcribed from google/gemma-4-E4B-it-qat-w4a16-ct's config.json (2026-07-30),
# trimmed to the fields config_from_hf reads. `text_config` nesting is deliberate:
# that is how the shipped multimodal checkpoint stores the decoder config.
E4B_HF_CONFIG = {
    "model_type": "gemma4",
    "text_config": {
        "vocab_size": 262144,
        "hidden_size": 2560,
        "intermediate_size": 10240,
        "num_hidden_layers": 42,
        "num_attention_heads": 8,
        "num_key_value_heads": 2,
        "head_dim": 256,
        "num_global_key_value_heads": None,   # absent -> falls back to num_key_value_heads
        "global_head_dim": 512,
        "num_kv_shared_layers": 18,
        "hidden_size_per_layer_input": 256,
        "vocab_size_per_layer_input": 262144,
        "use_double_wide_mlp": False,
        "sliding_window": 512,
        "rms_norm_eps": 1e-6,
        "final_logit_softcapping": 30.0,
        "attention_k_eq_v": False,
        "rope_parameters": {
            "full_attention": {"rope_theta": 1000000.0, "partial_rotary_factor": 0.25},
            "sliding_attention": {"rope_theta": 10000.0},
        },
        "layer_types": ["full_attention" if i % 6 == 5 else "sliding_attention"
                        for i in range(42)],
    },
}


class ConfigFromHFTest(unittest.TestCase):
    """config_from_hf must read every shape-bearing field off the checkpoint.

    A field it fails to read does not error — it silently inherits the dataclass
    default and builds a differently-shaped model. These assertions are pinned to
    the shipped E4B config so a default that drifts away from it is caught here
    rather than as a shape error 40 layers into a load.
    """

    def setUp(self):
        self.cfg = config_from_hf(E4B_HF_CONFIG)

    def test_reads_e4b_dimensions(self):
        self.assertEqual(self.cfg.hidden_size, 2560)
        self.assertEqual(self.cfg.intermediate_size, 10240)
        self.assertEqual(self.cfg.num_hidden_layers, 42)
        self.assertEqual(self.cfg.num_attention_heads, 8)
        self.assertEqual(self.cfg.head_dim, 256)
        self.assertEqual(self.cfg.global_head_dim, 512)
        self.assertEqual(self.cfg.vocab_size, 262144)

    def test_reads_kv_geometry(self):
        self.assertEqual(self.cfg.num_key_value_heads, 2)
        # config.json reports null; the fallback is num_key_value_heads, not the
        # dataclass default.
        self.assertEqual(self.cfg.num_global_key_value_heads, 2)
        self.assertEqual(self.cfg.num_kv_shared_layers, 18)
        self.assertEqual(self.cfg.first_kv_shared_layer_idx, 24)

    def test_every_shape_field_is_read_from_the_checkpoint(self):
        """Each field must come from config.json, not from the dataclass default.

        Asserting E4B's values against E4B-shaped defaults proves nothing — the
        assertion passes whether or not the field is read. So probe with values
        that DIFFER from every default: if config_from_hf ignores a field, the
        default comes back instead and the subTest fails.

        `use_double_wide_mlp` is why this test exists. It was never read, and
        because it is a bool, the failure mode was not an error but a model built
        with the wrong MLP width on its KV-shared layers.
        """
        defaults = Gemma4EConfig()
        probes = {
            "vocab_size": 1024,
            "hidden_size": 128,
            "intermediate_size": 320,
            "num_hidden_layers": 6,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 32,
            "global_head_dim": 64,
            "num_kv_shared_layers": 2,
            "hidden_size_per_layer_input": 64,
            "vocab_size_per_layer_input": 1024,
            "use_double_wide_mlp": True,
            "sliding_window": 128,
            "rms_norm_eps": 1e-5,
            "attention_k_eq_v": True,
        }
        text = dict(E4B_HF_CONFIG["text_config"])
        text.update(probes)
        text["layer_types"] = ["sliding_attention"] * probes["num_hidden_layers"]
        cfg = config_from_hf({"text_config": text})

        for field, want in probes.items():
            with self.subTest(field=field):
                self.assertNotEqual(
                    want, getattr(defaults, field),
                    f"probe for {field} equals the default, so this cannot detect "
                    f"an unread field — pick a different probe value")
                self.assertEqual(getattr(cfg, field), want)

    def test_reads_rope_and_window(self):
        self.assertEqual(self.cfg.global_rope_theta, 1000000.0)
        self.assertEqual(self.cfg.rope_theta, 10000.0)
        self.assertEqual(self.cfg.partial_rotary_factor, 0.25)
        self.assertEqual(self.cfg.sliding_window, 512)

    def test_layer_pattern_is_period_six(self):
        full = [i for i, t in enumerate(self.cfg.layer_types) if t == "full_attention"]
        self.assertEqual(full, [5, 11, 17, 23, 29, 35, 41])

    def test_kv_share_map_targets_last_layer_of_each_type(self):
        share = self.cfg.kv_share_map()
        self.assertEqual(len(share), 42)
        # Layers below the boundary own their KV.
        self.assertEqual(share[:24], list(range(24)))
        # Above it, each layer reuses the last pre-boundary layer of its own type:
        # full attention -> 23, sliding -> 22.
        self.assertEqual(share[23], 23)
        self.assertEqual(share[29], 23)
        self.assertEqual(share[24], 22)

    def test_defaults_match_the_shipped_e4b_config(self):
        """Gemma4EConfig() with no arguments must BE E4B, not approximately E4B."""
        defaults = Gemma4EConfig()
        for field in ("vocab_size", "hidden_size", "intermediate_size",
                      "num_hidden_layers", "num_attention_heads", "head_dim",
                      "global_head_dim", "num_key_value_heads",
                      "num_global_key_value_heads", "num_kv_shared_layers",
                      "use_double_wide_mlp", "hidden_size_per_layer_input",
                      "vocab_size_per_layer_input", "layer_types"):
            with self.subTest(field=field):
                self.assertEqual(getattr(defaults, field), getattr(self.cfg, field))


class JaxEngineTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.ckpt = Path(cls._tmp.name) / "tiny-gemma4e"
        cls.cfg = write_tiny_checkpoint(cls.ckpt)

        cls.engine = JaxGemmaEngine(
            model_id="synthetic/tiny-gemma4e",
            kv_cache_dtype="bf16",
            quant_mode="fp16",   # dense weights in this fixture
            max_model_len=64,
        )
        cls.engine.load(local_dir=str(cls.ckpt))

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_loads_without_torch(self):
        self.assertTrue(self.engine.is_ready)
        self.assertNotIn("torch", sys.modules, "engine load pulled in PyTorch")

    def test_kv_shared_layers_have_no_k_norm(self):
        """The #3225 shape: layers >= first_kv_shared_layer_idx carry no k/v params."""
        first = self.cfg.first_kv_shared_layer_idx
        for i in range(self.cfg.num_hidden_layers):
            attn = self.engine.params[f"layer_{i}"]["attn"]
            if i >= first:
                self.assertNotIn("k_norm", attn, f"layer {i} should not have k_norm")
                self.assertNotIn("k_proj", attn, f"layer {i} should not have k_proj")
            else:
                self.assertIn("k_norm", attn, f"layer {i} is missing k_norm")
                self.assertIn("k_proj", attn, f"layer {i} is missing k_proj")

    def test_streaming_yields_tokens_then_stats(self):
        out = list(self.engine.generate_stream([5, 9, 12], max_new_tokens=6, temperature=0.0))
        stats = out[-1]
        tokens = out[:-1]
        self.assertIsInstance(stats, GenerationStats)
        self.assertEqual(len(tokens), 6)
        self.assertTrue(all(isinstance(t, int) for t in tokens))
        self.assertEqual(stats.completion_tokens, 6)
        self.assertEqual(stats.prompt_tokens, 3)
        self.assertGreater(stats.prefill_ms, 0.0)
        self.assertEqual(stats.finish_reason, "length")

    def test_greedy_is_deterministic(self):
        a, _ = self.engine.generate([7, 3, 1], max_new_tokens=5, temperature=0.0)
        b, _ = self.engine.generate([7, 3, 1], max_new_tokens=5, temperature=0.0)
        self.assertEqual(a, b)

    def test_eos_stops_generation(self):
        """Feeding the first greedy token back as EOS must halt immediately."""
        tokens, _ = self.engine.generate([4, 4, 4], max_new_tokens=8, temperature=0.0)
        first = tokens[0]
        stopped, stats = self.engine.generate(
            [4, 4, 4], max_new_tokens=8, temperature=0.0, eos_token_ids=[first]
        )
        self.assertEqual(stopped, [])
        self.assertEqual(stats.finish_reason, "stop")
        self.assertEqual(stats.completion_tokens, 0)

    def test_max_model_len_is_enforced(self):
        long_prompt = list(range(1, 40))
        tokens, _ = self.engine.generate(long_prompt, max_new_tokens=1000, temperature=0.0)
        self.assertLessEqual(len(long_prompt) + len(tokens), self.engine.max_model_len)

    def test_rejects_prompt_longer_than_window(self):
        with self.assertRaises(ValueError):
            self.engine.generate(list(range(1, 200)), max_new_tokens=4)

    def test_sampling_respects_temperature(self):
        """Non-zero temperature with distinct seeds should not be locked to one output."""
        outs = {
            tuple(self.engine.generate(
                [2, 8, 6], max_new_tokens=6, temperature=1.5, top_k=50, seed=s
            )[0])
            for s in range(6)
        }
        self.assertGreater(len(outs), 1, "temperature sampling produced identical outputs")


if __name__ == "__main__":
    unittest.main()
