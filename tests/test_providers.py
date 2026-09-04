import unittest
import tempfile
from pathlib import Path

from llmcf.generator import (
    PackPaths,
    configure_bos_embedding_runtime,
    configure_rms0_runtime,
    configure_shortconv0_in_runtime,
    configure_shortconv0_mix_runtime,
    configure_ffn0_runtime,
    configure_quant_matvec_runtime,
    configure_shortconv_layer_runtime,
    configure_attention_layer_runtime,
    configure_lm_head_runtime,
    configure_generation_runtime,
    configure_prefill_runtime,
    configure_tokenizer_runtime,
    configure_token_embedding_runtime,
    generate_kernel_pack,
)
from llmcf.providers import (
    q4_0_term,
    q2k_dot_provider,
    q2k_quant,
    q6k_value,
    quant_dot_provider,
    rms_apply_provider,
    rms_l2_provider,
    rms_scale_provider,
    shortconv_mix_provider,
    shortconv_state_provider,
    swiglu_provider,
)


class ProviderTests(unittest.TestCase):
    def test_progress_bossbar_tracks_token_and_layer(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = generate_kernel_pack(directory, clean=True, kernel_widths=(256,))
            load = (paths.functions / "load.mcfunction").read_text()
            tick = (paths.functions / "tick.mcfunction").read_text().splitlines()
            ui = (paths.functions / "ui" / "tick.mcfunction").read_text()
            lm_head_ui = (paths.functions / "ui" / "lm_head.mcfunction").read_text()
        self.assertIn("scoreboard objectives add lfm.ui dummy", load)
        self.assertEqual(tick[0], "function lfm:ui/tick")
        self.assertIn("bossbar lfm:inference value", ui)
        self.assertIn('Token ","color":"light_purple"', ui)
        self.assertIn("#layer", ui)
        self.assertIn("matches 17..", ui)
        self.assertIn("Output token", lm_head_ui)
        self.assertIn("Vocab", lm_head_ui)

    def test_top_level_tick_batches_runtime_steps(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = generate_kernel_pack(directory, clean=True, kernel_widths=(256,))
            tick = (paths.functions / "tick.mcfunction").read_text().splitlines()
            fast = (paths.functions / "runtime" / "tick_fast.mcfunction").read_text()
        self.assertEqual(tick.count('execute if data storage lfm:job {state:"running"} run function lfm:runtime/tick'), 1)
        self.assertEqual(sum("runtime/tick_fast" in line for line in tick), 31)
        self.assertEqual(fast, "return 0\n")

    def test_only_matrix_rows_are_batched(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = generate_kernel_pack(directory, clean=True, kernel_widths=(256,))
            configure_bos_embedding_runtime(paths, total_work=4096, next_phase="rms0_load")
            configure_rms0_runtime(paths, next_phase="shortconv0_in_load")
            configure_shortconv0_in_runtime(paths)
            fast = (paths.functions / "runtime" / "tick_fast.mcfunction").read_text()
            classifier = (paths.functions / "runtime" / "classify_batch.mcfunction").read_text()
        self.assertIn("lfm:runtime/shortconv0_in/row", fast)
        self.assertNotIn("lfm:runtime/shortconv0_in/load", fast)
        self.assertNotIn("lfm:runtime/rms0/step", fast)
        self.assertIn('{phase:"shortconv0_in"}', classifier)

    def test_tick_batch_size_must_be_positive(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                generate_kernel_pack(directory, steps_per_tick=0)

    def test_supported_widths(self):
        for width in (32, 64, 128, 256):
            provider = q2k_dot_provider(width)
            self.assertEqual(provider["type"], "minecraft:add")

    def test_q2_dot_supports_activation_offset(self):
        provider = repr(q2k_dot_provider(32, activation_path="normalized", activation_offset=256))
        self.assertIn("normalized[256]", provider)
        self.assertIn("normalized[287]", provider)

    def test_quant_paths_stay_in_qs_words(self):
        provider = repr(q2k_quant(255))
        self.assertIn("weight.qs[15]", provider)
        self.assertIn("minecraft:floor_mod", provider)

    def test_all_observed_quant_types(self):
        for kind in ("Q4_0", "Q2_K", "Q3_K", "Q6_K"):
            provider = repr(quant_dot_provider(kind, 32))
            self.assertIn("weight.d", provider)
            self.assertIn("activation[31]", provider)

    def test_q4_0_term_selects_scale_and_nibbles(self):
        low = repr(q4_0_term(32))
        high = repr(q4_0_term(63))
        self.assertIn("weight.d[1]", low)
        self.assertIn("weight.qs[4]", low)
        self.assertIn("weight.qs[7]", high)
        self.assertIn("minecraft:floor_div", high)

    def test_q6_value_has_no_activation_dependency(self):
        provider = repr(q6k_value(255))
        self.assertIn("weight.qh[15]", provider)
        self.assertNotIn("activation", provider)

    def test_q6_value_rejects_out_of_range_lane(self):
        with self.assertRaises(ValueError):
            q6k_value(256)

    def test_bos_runtime_uses_parser_supported_data_modify_forms(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = PackPaths(Path(directory))
            configure_bos_embedding_runtime(paths)
            append = (paths.functions / "runtime" / "embedding" / "append_block.mcfunction").read_text()
            final_block = (paths.functions / "runtime" / "embedding" / "block_7.mcfunction").read_text()
        self.assertNotIn("append set compute", append)
        self.assertIn("activation append from storage lfm:scratch scalar", append)
        self.assertNotIn("data merge storage lfm:job result", final_block)

    def test_rms_providers(self):
        l2 = rms_l2_provider(4)
        self.assertEqual(l2["type"], "minecraft:length")
        self.assertEqual(len(l2["inputs"]), 4)
        self.assertIn("minecraft:sqrt", repr(rms_scale_provider(4)))
        self.assertIn("rms0.weight.values[3]", repr(rms_apply_provider(3)))

    def test_rms_runtime_is_chunked(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = PackPaths(Path(directory))
            configure_bos_embedding_runtime(paths, total_work=4096, next_phase="rms0_load")
            configure_rms0_runtime(paths)
            block = (paths.functions / "runtime" / "rms0" / "block_0.mcfunction").read_text()
            tick = (paths.functions / "runtime" / "tick.mcfunction").read_text()
        self.assertEqual(block.count(" set compute "), 256)
        self.assertIn('phase:"rms0_load"', tick)

    def test_shortconv_matvec_runtime_shape(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = PackPaths(Path(directory))
            configure_bos_embedding_runtime(paths, total_work=4096, next_phase="rms0_load")
            configure_rms0_runtime(paths, next_phase="shortconv0_in_load")
            configure_shortconv0_in_runtime(paths)
            row = (paths.functions / "runtime" / "shortconv0_in" / "row.mcfunction").read_text()
            loader = (paths.functions / "runtime" / "shortconv0_in" / "load.mcfunction").read_text()
        self.assertEqual(row.count("row_acc set compute"), 8)
        self.assertEqual(loader.count("load_"), 24)
        self.assertIn("shortconv0.in_proj set value []", loader)

    def test_shortconv_mix_layout(self):
        state = repr(shortconv_state_provider(7))
        mixed = repr(shortconv_mix_provider(7))
        self.assertIn("shortconv0.in_proj[4103]", state)
        self.assertIn("shortconv0.in_proj[2055]", mixed)
        self.assertIn("shortconv0.prev2[7]", mixed)
        self.assertIn("shortconv0.prev1[7]", mixed)
        self.assertIn("shortconv0.conv_weight.values[21]", mixed)
        self.assertIn("shortconv0.conv_weight.values[22]", mixed)
        self.assertIn("shortconv0.conv_weight.values[23]", mixed)

    def test_shortconv_mix_runtime_is_chunked(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = PackPaths(Path(directory))
            configure_bos_embedding_runtime(paths)
            configure_shortconv0_mix_runtime(paths)
            block = (paths.functions / "runtime" / "shortconv0_mix" / "block_0.mcfunction").read_text()
        self.assertEqual(block.count(" set compute "), 512)

    def test_ffn_gate_runtime_shape(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = PackPaths(Path(directory))
            configure_bos_embedding_runtime(paths)
            configure_quant_matvec_runtime(
                paths, "ffn_gate0", "Q2_K", "ffn_input0", "ffn0.gate", 2048, 8192, "done"
            )
            row = (paths.functions / "runtime" / "ffn_gate0" / "row.mcfunction").read_text()
            loader = (paths.functions / "runtime" / "ffn_gate0" / "load.mcfunction").read_text()
        self.assertEqual(row.count("row_acc set compute"), 8)
        self.assertEqual(loader.count("load_"), 32)

    def test_ffn_down_runtime_shape(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = PackPaths(Path(directory))
            configure_bos_embedding_runtime(paths)
            configure_quant_matvec_runtime(
                paths, "ffn_down0", "Q3_K", "ffn0.activated", "ffn0.down", 8192, 2048, "done"
            )
            row = (paths.functions / "runtime" / "ffn_down0" / "row.mcfunction").read_text()
            loader = (paths.functions / "runtime" / "ffn_down0" / "load.mcfunction").read_text()
        self.assertEqual(row.count("row_acc set compute"), 32)
        self.assertEqual(loader.count("load_"), 32)
        self.assertIn("chunk_row:64", row)

    def test_swiglu_provider_uses_pow_and_both_inputs(self):
        provider = repr(swiglu_provider(7))
        self.assertIn("minecraft:pow", provider)
        self.assertIn("ffn0.gate[7]", provider)
        self.assertIn("ffn0.up[7]", provider)

    def test_layer1_runtime_reuses_layer0_float_providers(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = generate_kernel_pack(directory, clean=True, kernel_widths=(256,))
            float_before = len(list(paths.float_providers.rglob("*.json")))
            configure_shortconv_layer_runtime(paths, 1)
            float_after = len(list(paths.float_providers.rglob("*.json")))
            row = (paths.functions / "runtime" / "layer1_shortconv_in" / "row.mcfunction").read_text()
            loader = (paths.functions / "runtime" / "layer1_shortconv_in" / "load_00.mcfunction").read_text()
            final = (paths.functions / "runtime" / "layer1_ffn_residual" / "block_7.mcfunction").read_text()
            mix_load = (paths.functions / "runtime" / "layer1_shortconv_mix" / "load.mcfunction").read_text()
            mix_final = (paths.functions / "runtime" / "layer1_shortconv_mix" / "block_7.mcfunction").read_text()
        self.assertEqual(float_before, float_after)
        self.assertIn("lfm:shortconv0_in/block_7", row)
        self.assertIn("lfm:runtime/layers/001/shortconv_in/000", loader)
        self.assertIn('phase set value "layer1_done"', final)
        self.assertIn("lfm:cache shortconv.layer1.prev1", mix_load)
        self.assertIn("lfm:cache shortconv.layer1.prev2", mix_final)

    def test_layer2_cached_attention_projects_q_and_uses_gqa_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = generate_kernel_pack(directory, clean=True, kernel_widths=(256,))
            float_before = len(list(paths.float_providers.rglob("*.json")))
            configure_attention_layer_runtime(paths, 2)
            float_after = len(list(paths.float_providers.rglob("*.json")))
            v_row = (paths.functions / "runtime" / "layer2_attn_v" / "row.mcfunction").read_text()
            k_row = (paths.functions / "runtime" / "layer2_attn_k" / "row.mcfunction").read_text()
            q_row = (paths.functions / "runtime" / "layer2_attn_q" / "row.mcfunction").read_text()
            out_row = (paths.functions / "runtime" / "layer2_attn_out" / "row.mcfunction").read_text()
            cache_load = (paths.functions / "runtime" / "layer2_attn_context" / "load.mcfunction").read_text()
            key_select = (paths.functions / "runtime" / "layer2_attn_context" / "select_key_04.mcfunction").read_text()
            score_done = (paths.functions / "runtime" / "layer2_attn_context" / "score_done_check.mcfunction").read_text()
            head0 = (paths.functions / "runtime" / "layer2_attn_k_norm" / "head_00.mcfunction").read_text()
            q_rope = (paths.functions / "runtime" / "layer2_attn_q_rope" / "head_31.mcfunction").read_text()
        self.assertEqual(float_after - float_before, 222)
        self.assertIn("lfm:attention_v/block_7", v_row)
        self.assertIn("lfm:attention_k/block_7", k_row)
        self.assertIn("lfm:attention_k/block_7", q_row)
        self.assertIn("lfm:attention_out/block_7", out_row)
        self.assertIn("lfm:cache attention.layer2.keys append", cache_load)
        self.assertIn("keys[$(pos)][64]", key_select)
        self.assertIn("$(context_length)", score_done)
        self.assertEqual(head0.count("attention_head_norm/apply/"), 64)
        self.assertEqual(q_rope.count("attention_rope/"), 64)

    def test_lm_head_updates_argmax_without_storing_all_logits(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = generate_kernel_pack(directory, clean=True, kernel_widths=(256,))
            float_before = len(list(paths.float_providers.rglob("*.json")))
            configure_lm_head_runtime(paths, rows=512)
            float_after = len(list(paths.float_providers.rglob("*.json")))
            row = (paths.functions / "runtime" / "lm_head" / "row.mcfunction").read_text()
            loader = (paths.functions / "runtime" / "lm_head" / "load.mcfunction").read_text()
            done = (paths.functions / "runtime" / "lm_head" / "done.mcfunction").read_text()
            load = (paths.functions / "load.mcfunction").read_text()
        self.assertEqual(float_after - float_before, 8)
        self.assertNotIn("logits append", row)
        self.assertIn("row_acc 10000000", row)
        self.assertLess(row.index("result.token_id"), row.index("increment_row"))
        self.assertIn("#best lfm.argmax -2147483648", loader)
        self.assertIn('state set value "done"', done)
        self.assertIn("scoreboard objectives add lfm.argmax dummy", load)
        self.assertIn("runtime/generation/on_token", done)

    def test_generation_runtime_schedules_continuation_and_collects_output(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = generate_kernel_pack(directory, clean=True, kernel_widths=(256,))
            configure_generation_runtime(paths)
            on_token = (paths.functions / "runtime" / "generation" / "on_token.mcfunction").read_text()
            start = (paths.functions / "api" / "start_generate_bos.mcfunction").read_text()
            more = (paths.functions / "api" / "generate_more.mcfunction").read_text()
        self.assertIn("token_ids append", on_token)
        self.assertIn("pieces append", on_token)
        self.assertIn("schedule function lfm:runtime/generation/continue 1t replace", on_token)
        self.assertIn("result:{token_id:7}", on_token)
        self.assertIn("$(max_tokens)", start)
        self.assertIn("function lfm:api/continue", more)

    def test_prefill_runtime_skips_lm_head_until_the_prompt_queue_is_empty(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = generate_kernel_pack(directory, clean=True, kernel_widths=(256,))
            configure_lm_head_runtime(paths, rows=512)
            configure_token_embedding_runtime(paths, vocab_size=512)
            configure_prefill_runtime(paths)
            complete = (paths.functions / "runtime" / "prefill" / "token_complete.mcfunction").read_text()
            next_token = (paths.functions / "runtime" / "prefill" / "next.mcfunction").read_text()
            api = (paths.functions / "api" / "start_prompt.mcfunction").read_text()
            append_api = (paths.functions / "api" / "append_prompt.mcfunction").read_text()
            tick = (paths.functions / "runtime" / "tick.mcfunction").read_text()
        self.assertIn("queue[0] run return run function lfm:runtime/prefill/next", complete)
        self.assertIn('phase set value "final_norm_load"', complete)
        self.assertIn("cache position set compute", next_token)
        self.assertIn("cache tokens append", next_token)
        self.assertIn("token_embedding/start", next_token)
        self.assertIn("$(tokens)", api)
        self.assertIn("#prefill_length", api)
        self.assertIn("cache position", append_api)
        self.assertIn("function lfm:api/continue", append_api)
        self.assertIn('phase:"prefill_token_complete"', tick)

    def test_tokenizer_lookup_uses_dynamic_macro_index(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = generate_kernel_pack(directory, clean=True, kernel_widths=(256,))
            configure_tokenizer_runtime(paths)
            lookup = (paths.functions / "runtime" / "tokenizer" / "lookup.mcfunction").read_text()
            loader = (paths.functions / "runtime" / "tokenizer" / "load.mcfunction").read_text()
            load = (paths.functions / "load.mcfunction").read_text()
        self.assertEqual(
            lookup,
            "$data modify storage lfm:job result.piece set from storage lfm:tokenizer tokens[$(token_id)]\n",
        )
        self.assertIn("data.values", loader)
        self.assertIn("function lfm:runtime/tokenizer/load", load)

    def test_token_embedding_selects_shared_lm_head_row(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = generate_kernel_pack(directory, clean=True, kernel_widths=(256,))
            configure_bos_embedding_runtime(paths)
            configure_token_embedding_runtime(paths)
            prepare = (paths.functions / "runtime" / "token_embedding" / "prepare.mcfunction").read_text()
            loader = (paths.functions / "runtime" / "token_embedding" / "load.mcfunction").read_text()
            selected = (paths.functions / "runtime" / "token_embedding" / "append_selected.mcfunction").read_text()
            step = (paths.functions / "runtime" / "token_embedding" / "step.mcfunction").read_text()
            api = (paths.functions / "api" / "start_token.mcfunction").read_text()
            continuation = (paths.functions / "api" / "continue.mcfunction").read_text()
            block_index = (paths.int_providers / "token_embedding" / "block_index.json").read_text()
            load255 = (paths.functions / "runtime" / "token_embedding" / "load_255.mcfunction").read_text()
        self.assertIn("token_embedding/chunk", prepare)
        self.assertEqual(loader.count("load_"), 256)
        self.assertIn("lm_head/255", load255)
        self.assertIn("embedding.blocks[$(index)]", selected)
        self.assertIn("block_index", step)
        self.assertIn("2047", block_index)
        self.assertIn("$(token)", api)
        self.assertIn("result.token_id", continuation)
        self.assertIn("token_embedding/start", continuation)


if __name__ == "__main__":
    unittest.main()
