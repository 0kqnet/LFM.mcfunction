from __future__ import annotations

import json
import math
from pathlib import Path
import struct

from .generator import (
    configure_bos_embedding_runtime,
    configure_rms0_runtime,
    configure_shortconv0_in_runtime,
    configure_shortconv0_mix_runtime,
    configure_shortconv0_out_runtime,
    configure_residual_ffn_norm0_runtime,
    configure_ffn0_runtime,
    configure_shortconv_layer_runtime,
    configure_attention_layer_runtime,
    configure_final_norm_runtime,
    configure_generation_runtime,
    configure_prefill_runtime,
    configure_lm_head_runtime,
    configure_tokenizer_runtime,
    configure_minecraft_input_runtime,
    configure_token_embedding_runtime,
    configure_rope_table_runtime,
    generate_kernel_pack,
    PackPaths,
)
from .gguf import GGUFFile, file_sha256, parse_gguf
from .q2k import (
    Q40BlockGroup,
    Q4_0_GROUP_BYTES,
    Q2KBlock,
    Q2_K_BLOCK_BYTES,
    Q3KBlock,
    Q3_K_BLOCK_BYTES,
    Q6KBlock,
    Q6_K_BLOCK_BYTES,
)
from .structure import (
    write_character_token_table,
    write_float_tensor,
    write_quant_chunk,
    write_rope_table,
    write_string_table,
)
from .tokenizer import (
    byte_token_ids,
    character_byte_token_table,
    display_vocabulary,
)


EXPECTED = {
    "general.architecture": "lfm2",
    "lfm2.block_count": 16,
    "lfm2.embedding_length": 2048,
    "lfm2.feed_forward_length": 8192,
    "lfm2.attention.head_count": 32,
    "lfm2.vocab_size": 65536,
    "lfm2.shortconv.l_cache": 3,
    "tokenizer.ggml.bos_token_id": 1,
    "tokenizer.ggml.eos_token_id": 7,
}

QUANT_LAYOUTS = {
    "Q4_0": (Q4_0_GROUP_BYTES, Q40BlockGroup),
    "Q2_K": (Q2_K_BLOCK_BYTES, Q2KBlock),
    "Q3_K": (Q3_K_BLOCK_BYTES, Q3KBlock),
    "Q6_K": (Q6_K_BLOCK_BYTES, Q6KBlock),
}


def _chat_wrapper_tokens(vocabulary: list[str]) -> tuple[list[int], list[int]]:
    token_by_piece = {piece: token_id for token_id, piece in enumerate(vocabulary)}
    newline = byte_token_ids(vocabulary)[ord("\n")]
    return (
        [6, token_by_piece["user"], newline],
        [7, newline, 6, token_by_piece["assistant"], newline],
    )


def compile_tokenizer_assets(
    model_path: str | Path,
    destination: str | Path,
    lock_path: str | Path,
) -> dict:
    source = Path(model_path)
    destination = Path(destination)
    lock = json.loads(Path(lock_path).read_text(encoding="utf-8"))
    _validate_locked_file(source, lock)
    model = parse_gguf(source)
    _validate_model(model)
    if not (destination / "pack.mcmeta").is_file():
        raise FileNotFoundError(f"not an existing datapack: {destination}")
    paths = PackPaths(destination)
    vocabulary = model.metadata["tokenizer.ggml.tokens"]
    token_types = model.metadata["tokenizer.ggml.token_type"]
    character_table = character_byte_token_table(vocabulary)
    write_string_table(
        paths.structures / "runtime" / "tokenizer.nbt",
        "tokenizer.ggml.tokens.display",
        display_vocabulary(vocabulary, token_types),
        0x12001,
    )
    write_character_token_table(
        paths.structures / "runtime" / "input_chars.nbt",
        character_table,
        0x12003,
    )
    configure_tokenizer_runtime(paths)
    prefix_tokens, suffix_tokens = _chat_wrapper_tokens(vocabulary)
    configure_minecraft_input_runtime(paths, prefix_tokens, suffix_tokens)
    manifest = {
        "model_sha256": lock["sha256"],
        "method": "reversible_character_tokens_with_utf8_byte_fallback",
        "supported_characters": len(character_table),
        "max_input_characters": 64,
        "max_context_tokens": 256,
    }
    _json_path = destination / "minecraft-input-tokenizer.json"
    _json_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def compile_model(
    model_path: str | Path,
    destination: str | Path,
    lock_path: str | Path,
    blocks_per_chunk: int = 2048,
) -> dict:
    if blocks_per_chunk <= 0:
        raise ValueError("blocks_per_chunk must be positive")
    source = Path(model_path)
    lock = json.loads(Path(lock_path).read_text(encoding="utf-8"))
    _validate_locked_file(source, lock)
    model = parse_gguf(source)
    _validate_model(model)
    paths = generate_kernel_pack(destination, clean=True, kernel_widths=(256,))

    chunks: list[dict] = []
    tensor_entries: list[dict] = []
    chunk_index = 0
    bos_blocks: list[Q6KBlock] | None = None
    rms0_values: tuple[float, ...] | None = None
    shortconv0_chunk_count = 0
    shortconv0_conv_values: tuple[float, ...] | None = None
    shortconv0_out_chunk_count = 0
    ffn_norm0_values: tuple[float, ...] | None = None
    ffn0_chunk_counts = {"gate": 0, "up": 0, "down": 0}
    ffn0_tensor_parts = {
        "blk.0.ffn_gate.weight": "gate",
        "blk.0.ffn_up.weight": "up",
        "blk.0.ffn_down.weight": "down",
    }
    layer1_float_values: dict[str, tuple[float, ...]] = {}
    layer1_chunk_counts = {"shortconv_in": 0, "shortconv_out": 0, "ffn_gate": 0, "ffn_up": 0, "ffn_down": 0}
    layer1_tensor_parts = {
        "blk.1.shortconv.in_proj.weight": "shortconv_in",
        "blk.1.shortconv.out_proj.weight": "shortconv_out",
        "blk.1.ffn_gate.weight": "ffn_gate",
        "blk.1.ffn_up.weight": "ffn_up",
        "blk.1.ffn_down.weight": "ffn_down",
    }
    layer2_float_values: dict[str, tuple[float, ...]] = {}
    layer2_chunk_counts = {"attn_v": 0, "attn_out": 0, "ffn_gate": 0, "ffn_up": 0, "ffn_down": 0}
    layer2_tensor_parts = {
        "blk.2.attn_v.weight": "attn_v",
        "blk.2.attn_output.weight": "attn_out",
        "blk.2.ffn_gate.weight": "ffn_gate",
        "blk.2.ffn_up.weight": "ffn_up",
        "blk.2.ffn_down.weight": "ffn_down",
    }
    with source.open("rb") as handle:
        for tensor_index, tensor in enumerate(model.tensors):
            print(
                f"[{tensor_index + 1:03d}/{len(model.tensors):03d}] "
                f"{tensor.type_name:4} {tensor.name} {tensor.dimensions}",
                flush=True,
            )
            entry = {
                "name": tensor.name,
                "type": tensor.type_name,
                "dimensions": list(tensor.dimensions),
                "elements": tensor.elements,
                "input_width": tensor.dimensions[0],
                "rows": tensor.elements // tensor.dimensions[0],
                "chunks": [],
            }
            if tensor.type_name == "F32":
                handle.seek(model.data_offset + tensor.offset)
                raw = handle.read(tensor.elements * 4)
                if len(raw) != tensor.elements * 4:
                    raise EOFError(tensor.name)
                values = struct.unpack(f"<{tensor.elements}f", raw)
                if tensor.name == "blk.0.attn_norm.weight":
                    rms0_values = values
                if tensor.name == "blk.0.shortconv.conv.weight":
                    shortconv0_conv_values = values
                if tensor.name == "blk.0.ffn_norm.weight":
                    ffn_norm0_values = values
                if tensor.name in {
                    "blk.1.attn_norm.weight",
                    "blk.1.shortconv.conv.weight",
                    "blk.1.ffn_norm.weight",
                }:
                    layer1_float_values[tensor.name] = values
                if tensor.name in {"blk.2.attn_norm.weight", "blk.2.ffn_norm.weight"}:
                    layer2_float_values[tensor.name] = values
                resource = f"lfm:weights/{chunk_index:05d}"
                write_float_tensor(
                    paths.structures / "weights" / f"{chunk_index:05d}.nbt",
                    tensor.name,
                    values,
                    chunk_index,
                )
                chunk = {
                    "index": chunk_index,
                    "resource": resource,
                    "tensor": tensor.name,
                    "type": "F32",
                    "first_value": 0,
                    "value_count": tensor.elements,
                }
                chunks.append(chunk)
                entry["chunks"].append(chunk_index)
                chunk_index += 1
            elif tensor.type_name in QUANT_LAYOUTS:
                if tensor.elements % 256:
                    raise ValueError(f"quant tensor is not QK_K-aligned: {tensor.name}")
                block_bytes, block_class = QUANT_LAYOUTS[tensor.type_name]
                total_blocks = tensor.elements // 256
                handle.seek(model.data_offset + tensor.offset)
                for first_block in range(0, total_blocks, blocks_per_chunk):
                    count = min(blocks_per_chunk, total_blocks - first_block)
                    raw = handle.read(count * block_bytes)
                    if len(raw) != count * block_bytes:
                        raise EOFError(f"{tensor.name} block {first_block}")
                    blocks = [
                        block_class.from_bytes(raw[offset : offset + block_bytes])
                        for offset in range(0, len(raw), block_bytes)
                    ]
                    if tensor.name == "token_embd.weight" and first_block == 0:
                        bos_blocks = list(blocks[8:16])
                    resource = f"lfm:weights/{chunk_index:05d}"
                    write_quant_chunk(
                        paths.structures / "weights" / f"{chunk_index:05d}.nbt",
                        tensor.name,
                        first_block,
                        blocks,
                    )
                    if tensor.name == "blk.0.shortconv.in_proj.weight":
                        write_quant_chunk(
                            paths.structures / "runtime" / "shortconv0_in" / f"{shortconv0_chunk_count:03d}.nbt",
                            tensor.name,
                            first_block,
                            blocks,
                        )
                        shortconv0_chunk_count += 1
                    if tensor.name == "blk.0.shortconv.out_proj.weight":
                        write_quant_chunk(paths.structures / "runtime" / "shortconv0_out" / f"{shortconv0_out_chunk_count:03d}.nbt", tensor.name, first_block, blocks)
                        shortconv0_out_chunk_count += 1
                    if tensor.name in ffn0_tensor_parts:
                        part = ffn0_tensor_parts[tensor.name]
                        write_quant_chunk(
                            paths.structures / "runtime" / f"ffn_{part}0" / f"{ffn0_chunk_counts[part]:03d}.nbt",
                            tensor.name,
                            first_block,
                            blocks,
                        )
                        ffn0_chunk_counts[part] += 1
                    if tensor.name in layer1_tensor_parts:
                        part = layer1_tensor_parts[tensor.name]
                        write_quant_chunk(
                            paths.structures / "runtime" / "layers" / "001" / part / f"{layer1_chunk_counts[part]:03d}.nbt",
                            tensor.name,
                            first_block,
                            blocks,
                        )
                        layer1_chunk_counts[part] += 1
                    if tensor.name in layer2_tensor_parts:
                        part = layer2_tensor_parts[tensor.name]
                        write_quant_chunk(
                            paths.structures / "runtime" / "layers" / "002" / part / f"{layer2_chunk_counts[part]:03d}.nbt",
                            tensor.name,
                            first_block,
                            blocks,
                        )
                        layer2_chunk_counts[part] += 1
                    chunk = {
                        "index": chunk_index,
                        "resource": resource,
                        "tensor": tensor.name,
                        "type": tensor.type_name,
                        "first_block": first_block,
                        "block_count": count,
                    }
                    chunks.append(chunk)
                    entry["chunks"].append(chunk_index)
                    chunk_index += 1
            else:
                raise ValueError(f"unsupported tensor type {tensor.type_name}: {tensor.name}")
            tensor_entries.append(entry)

    if bos_blocks is None or len(bos_blocks) != 8:
        raise ValueError("could not extract BOS embedding row")
    if rms0_values is None or len(rms0_values) != 2048:
        raise ValueError("could not extract Layer 0 RMSNorm weights")
    if shortconv0_chunk_count != 24:
        raise ValueError(f"unexpected Layer 0 ShortConv chunk count: {shortconv0_chunk_count}")
    if shortconv0_conv_values is None or len(shortconv0_conv_values) != 6144:
        raise ValueError("could not extract Layer 0 ShortConv kernel")
    if shortconv0_out_chunk_count != 8:
        raise ValueError(f"unexpected Layer 0 ShortConv output chunk count: {shortconv0_out_chunk_count}")
    if ffn_norm0_values is None or len(ffn_norm0_values) != 2048:
        raise ValueError("could not extract Layer 0 FFN norm")
    for part, count in ffn0_chunk_counts.items():
        if count != 32:
            raise ValueError(f"unexpected Layer 0 FFN {part} chunk count: {count}")
    expected_layer1_chunks = {"shortconv_in": 24, "shortconv_out": 8, "ffn_gate": 32, "ffn_up": 32, "ffn_down": 32}
    for part, expected_count in expected_layer1_chunks.items():
        if layer1_chunk_counts[part] != expected_count:
            raise ValueError(f"unexpected Layer 1 {part} chunk count: {layer1_chunk_counts[part]}")
    expected_layer1_float_lengths = {
        "blk.1.attn_norm.weight": 2048,
        "blk.1.shortconv.conv.weight": 6144,
        "blk.1.ffn_norm.weight": 2048,
    }
    for tensor_name, expected_length in expected_layer1_float_lengths.items():
        if len(layer1_float_values.get(tensor_name, ())) != expected_length:
            raise ValueError(f"could not extract {tensor_name}")
    expected_layer2_chunks = {"attn_v": 2, "attn_out": 8, "ffn_gate": 32, "ffn_up": 32, "ffn_down": 32}
    for part, expected_count in expected_layer2_chunks.items():
        if layer2_chunk_counts[part] != expected_count:
            raise ValueError(f"unexpected Layer 2 {part} chunk count: {layer2_chunk_counts[part]}")
    for tensor_name in ("blk.2.attn_norm.weight", "blk.2.ffn_norm.weight"):
        if len(layer2_float_values.get(tensor_name, ())) != 2048:
            raise ValueError(f"could not extract {tensor_name}")
    write_quant_chunk(
        paths.structures / "runtime" / "bos_embedding.nbt",
        "token_embd.weight",
        8,
        bos_blocks,
    )
    write_float_tensor(
        paths.structures / "runtime" / "rms0_weight.nbt",
        "blk.0.attn_norm.weight",
        rms0_values,
        0x10000,
    )
    write_float_tensor(paths.structures / "runtime" / "shortconv0_conv.nbt", "blk.0.shortconv.conv.weight", shortconv0_conv_values, 0x10001)
    write_float_tensor(paths.structures / "runtime" / "ffn_norm0_weight.nbt", "blk.0.ffn_norm.weight", ffn_norm0_values, 0x10002)
    write_float_tensor(paths.structures / "runtime" / "layers" / "001" / "attn_norm.nbt", "blk.1.attn_norm.weight", layer1_float_values["blk.1.attn_norm.weight"], 0x10100)
    write_float_tensor(paths.structures / "runtime" / "layers" / "001" / "shortconv_conv.nbt", "blk.1.shortconv.conv.weight", layer1_float_values["blk.1.shortconv.conv.weight"], 0x10101)
    write_float_tensor(paths.structures / "runtime" / "layers" / "001" / "ffn_norm.nbt", "blk.1.ffn_norm.weight", layer1_float_values["blk.1.ffn_norm.weight"], 0x10102)
    write_float_tensor(paths.structures / "runtime" / "layers" / "002" / "attn_norm.nbt", "blk.2.attn_norm.weight", layer2_float_values["blk.2.attn_norm.weight"], 0x10200)
    write_float_tensor(paths.structures / "runtime" / "layers" / "002" / "ffn_norm.nbt", "blk.2.ffn_norm.weight", layer2_float_values["blk.2.ffn_norm.weight"], 0x10201)
    total_work = (
        4096
        + 2048 * 6144
        + 2048
        + 2048 * 2048
        + 4096
        + 2048 * 8192 * 2
        + 8192
        + 8192 * 2048
        + 2048
    )
    configure_bos_embedding_runtime(paths, total_work=total_work, next_phase="rms0_load", model_sha=lock["sha256"])
    configure_rms0_runtime(paths, total_work=total_work, next_phase="shortconv0_in_load")
    tensor_types = {tensor.name: tensor.type_name for tensor in model.tensors}
    configure_shortconv0_in_runtime(paths, total_work=total_work, next_phase="shortconv0_mix_load", kind=tensor_types["blk.0.shortconv.in_proj.weight"])
    configure_shortconv0_mix_runtime(paths, total_work=total_work, next_phase="shortconv0_out_load")
    configure_shortconv0_out_runtime(paths, next_phase="residual0_load", kind=tensor_types["blk.0.shortconv.out_proj.weight"])
    configure_residual_ffn_norm0_runtime(paths, next_phase="ffn_gate0_load")
    configure_ffn0_runtime(paths, next_phase="layer1_attn_norm_load", gate_kind=tensor_types["blk.0.ffn_gate.weight"], up_kind=tensor_types["blk.0.ffn_up.weight"], down_kind=tensor_types["blk.0.ffn_down.weight"])
    configure_shortconv_layer_runtime(paths, 1, next_phase="layer2_attn_norm_load", short_in_kind=tensor_types["blk.1.shortconv.in_proj.weight"], short_out_kind=tensor_types["blk.1.shortconv.out_proj.weight"], gate_kind=tensor_types["blk.1.ffn_gate.weight"], up_kind=tensor_types["blk.1.ffn_up.weight"], down_kind=tensor_types["blk.1.ffn_down.weight"])
    configure_attention_layer_runtime(paths, 2, key_kind=tensor_types["blk.2.attn_k.weight"], value_kind=tensor_types["blk.2.attn_v.weight"], query_kind=tensor_types["blk.2.attn_q.weight"], output_kind=tensor_types["blk.2.attn_output.weight"], gate_kind=tensor_types["blk.2.ffn_gate.weight"], up_kind=tensor_types["blk.2.ffn_up.weight"], down_kind=tensor_types["blk.2.ffn_down.weight"])

    manifest = {
        "schema": 1,
        "model": {
            "repository": lock["repository"],
            "revision": lock["revision"],
            "filename": lock["filename"],
            "sha256": lock["sha256"],
            "size": lock["size"],
            "name": model.metadata.get("general.name"),
        },
        "architecture": {key: model.metadata[key] for key in EXPECTED},
        "blocks_per_chunk": blocks_per_chunk,
        "chunk_count": len(chunks),
        "chunks": chunks,
        "tensors": tensor_entries,
        "execution": _execution_plan(model),
    }
    manifest_path = paths.root / "lfm-model-manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(chunks)} chunks to {paths.root}", flush=True)
    return manifest


def compile_bos_pack(
    model_path: str | Path,
    destination: str | Path,
    lock_path: str | Path,
) -> dict:
    source = Path(model_path)
    lock = json.loads(Path(lock_path).read_text(encoding="utf-8"))
    _validate_locked_file(source, lock)
    model = parse_gguf(source)
    _validate_model(model)
    tensor = next(item for item in model.tensors if item.name == "token_embd.weight")
    if tensor.type_name != "Q6_K" or tensor.dimensions != (2048, 65536):
        raise ValueError(f"unexpected embedding tensor: {tensor.type_name} {tensor.dimensions}")

    first_block = 8
    with source.open("rb") as handle:
        handle.seek(model.data_offset + tensor.offset + first_block * Q6_K_BLOCK_BYTES)
        raw = handle.read(8 * Q6_K_BLOCK_BYTES)
    if len(raw) != 8 * Q6_K_BLOCK_BYTES:
        raise EOFError("token_embd.weight BOS row")
    blocks = [
        Q6KBlock.from_bytes(raw[offset : offset + Q6_K_BLOCK_BYTES])
        for offset in range(0, len(raw), Q6_K_BLOCK_BYTES)
    ]
    values = [value for block in blocks for value in block.dequantize()]
    rms_tensor = next(item for item in model.tensors if item.name == "blk.0.attn_norm.weight")
    with source.open("rb") as handle:
        handle.seek(model.data_offset + rms_tensor.offset)
        rms_raw = handle.read(rms_tensor.elements * 4)
    if len(rms_raw) != rms_tensor.elements * 4:
        raise EOFError(rms_tensor.name)
    rms_weights = struct.unpack(f"<{rms_tensor.elements}f", rms_raw)
    epsilon = float(model.metadata["lfm2.attention.layer_norm_rms_epsilon"])
    l2 = sum(value * value for value in values) ** 0.5
    scale = 1.0 / ((l2 * l2 / len(values) + epsilon) ** 0.5)
    normalized = [value * scale * weight for value, weight in zip(values, rms_weights)]
    short_tensor = next(item for item in model.tensors if item.name == "blk.0.shortconv.in_proj.weight")
    conv_tensor = next(item for item in model.tensors if item.name == "blk.0.shortconv.conv.weight")
    out_tensor = next(item for item in model.tensors if item.name == "blk.0.shortconv.out_proj.weight")
    ffn_norm_tensor = next(item for item in model.tensors if item.name == "blk.0.ffn_norm.weight")
    ffn_gate_tensor = next(item for item in model.tensors if item.name == "blk.0.ffn_gate.weight")
    ffn_up_tensor = next(item for item in model.tensors if item.name == "blk.0.ffn_up.weight")
    ffn_down_tensor = next(item for item in model.tensors if item.name == "blk.0.ffn_down.weight")
    expected_ffn = (
        (ffn_gate_tensor, (2048, 8192)),
        (ffn_up_tensor, (2048, 8192)),
        (ffn_down_tensor, (8192, 2048)),
    )
    for ffn_tensor, expected_dimensions in expected_ffn:
        if ffn_tensor.type_name not in QUANT_LAYOUTS or ffn_tensor.dimensions != expected_dimensions:
            raise ValueError(
                f"unexpected FFN tensor {ffn_tensor.name}: "
                f"{ffn_tensor.type_name} {ffn_tensor.dimensions}"
            )
    with source.open("rb") as handle:
        handle.seek(model.data_offset + conv_tensor.offset)
        conv_raw = handle.read(conv_tensor.elements * 4)
    if len(conv_raw) != conv_tensor.elements * 4:
        raise EOFError(conv_tensor.name)
    conv_weights = struct.unpack(f"<{conv_tensor.elements}f", conv_raw)
    with source.open("rb") as handle:
        handle.seek(model.data_offset + ffn_norm_tensor.offset)
        ffn_norm_raw = handle.read(ffn_norm_tensor.elements * 4)
    ffn_norm_weights = struct.unpack(f"<{ffn_norm_tensor.elements}f", ffn_norm_raw)

    paths = generate_kernel_pack(destination, clean=True, kernel_widths=(256,))

    def compile_quant_matrix(matrix_tensor, structure_path: str, activation: list[float]) -> list[float]:
        try:
            block_bytes, block_class = QUANT_LAYOUTS[matrix_tensor.type_name]
        except KeyError as error:
            raise ValueError(
                f"unsupported quant tensor {matrix_tensor.name}: {matrix_tensor.type_name}"
            ) from error
        if matrix_tensor.dimensions[0] % 256:
            raise ValueError(f"quant matrix width is not 256-aligned: {matrix_tensor.name}")
        blocks_per_row = matrix_tensor.dimensions[0] // 256
        rows_per_chunk = 2048 // blocks_per_row
        chunk_count = matrix_tensor.elements // 256 // 2048
        outputs: list[float] = []
        with source.open("rb") as handle:
            handle.seek(model.data_offset + matrix_tensor.offset)
            for chunk in range(chunk_count):
                raw = handle.read(2048 * block_bytes)
                if len(raw) != 2048 * block_bytes:
                    raise EOFError(f"{matrix_tensor.name} chunk {chunk}")
                matrix_blocks = [
                    block_class.from_bytes(raw[offset : offset + block_bytes])
                    for offset in range(0, len(raw), block_bytes)
                ]
                write_quant_chunk(
                    paths.structures / "runtime" / structure_path / f"{chunk:03d}.nbt",
                    matrix_tensor.name,
                    chunk * 2048,
                    matrix_blocks,
                )
                for row_offset in range(0, len(matrix_blocks), blocks_per_row):
                    outputs.append(
                        sum(
                            matrix_blocks[row_offset + block].dot(
                                activation[block * 256 : (block + 1) * 256]
                            )
                            for block in range(blocks_per_row)
                        )
                    )
        expected_rows = rows_per_chunk * chunk_count
        if len(outputs) != expected_rows:
            raise ValueError(f"unexpected {matrix_tensor.name} output count: {len(outputs)}")
        return outputs
    write_quant_chunk(
        paths.structures / "runtime" / "bos_embedding.nbt",
        tensor.name,
        first_block,
        blocks,
    )
    write_float_tensor(
        paths.structures / "runtime" / "rms0_weight.nbt",
        rms_tensor.name,
        rms_weights,
        0x10000,
    )
    total_work = (
        4096
        + 2048 * 6144
        + 2048
        + 2048 * 2048
        + 4096
        + 2048 * 8192 * 2
        + 8192
        + 8192 * 2048
        + 2048
    )
    configure_bos_embedding_runtime(paths, total_work=total_work, next_phase="rms0_load", model_sha=lock["sha256"])
    configure_rms0_runtime(paths, epsilon=epsilon, total_work=total_work, next_phase="shortconv0_in_load")
    short_outputs = compile_quant_matrix(short_tensor, "shortconv0_in", normalized)
    write_float_tensor(paths.structures / "runtime" / "shortconv0_conv.nbt", conv_tensor.name, conv_weights, 0x10001)
    configure_shortconv0_in_runtime(paths, total_work=total_work, next_phase="shortconv0_mix_load", kind=short_tensor.type_name)
    configure_shortconv0_mix_runtime(paths, total_work=total_work, next_phase="shortconv0_out_load")
    conv_state = [short_outputs[index] * short_outputs[4096 + index] for index in range(2048)]
    mixed = [short_outputs[2048 + index] * conv_state[index] * conv_weights[index * 3 + 2] for index in range(2048)]
    out_outputs = compile_quant_matrix(out_tensor, "shortconv0_out", mixed)
    configure_shortconv0_out_runtime(paths, next_phase="residual0_load", kind=out_tensor.type_name)
    write_float_tensor(paths.structures / "runtime" / "ffn_norm0_weight.nbt", ffn_norm_tensor.name, ffn_norm_weights, 0x10002)
    configure_residual_ffn_norm0_runtime(paths, epsilon=epsilon, next_phase="ffn_gate0_load")
    residual = [values[index] + out_outputs[index] for index in range(2048)]
    ffn_l2 = sum(value * value for value in residual) ** 0.5
    ffn_scale = 1.0 / ((ffn_l2 * ffn_l2 / 2048 + epsilon) ** 0.5)
    ffn_input = [residual[index] * ffn_scale * ffn_norm_weights[index] for index in range(2048)]

    ffn_gate = compile_quant_matrix(ffn_gate_tensor, "ffn_gate0", ffn_input)
    ffn_up = compile_quant_matrix(ffn_up_tensor, "ffn_up0", ffn_input)
    ffn_activated = []
    for gate, up in zip(ffn_gate, ffn_up):
        if gate >= 0.0:
            sigmoid = 1.0 / (1.0 + math.exp(-gate))
        else:
            exp_gate = math.exp(gate)
            sigmoid = exp_gate / (1.0 + exp_gate)
        ffn_activated.append(gate * sigmoid * up)
    ffn_down = compile_quant_matrix(ffn_down_tensor, "ffn_down0", ffn_activated)
    layer0_output = [residual[index] + ffn_down[index] for index in range(2048)]

    def tensor_named(name: str):
        return next(item for item in model.tensors if item.name == name)

    def read_f32_tensor(matrix_tensor) -> tuple[float, ...]:
        if matrix_tensor.type_name != "F32":
            raise ValueError(f"expected F32 tensor: {matrix_tensor.name}")
        with source.open("rb") as handle:
            handle.seek(model.data_offset + matrix_tensor.offset)
            raw = handle.read(matrix_tensor.elements * 4)
        if len(raw) != matrix_tensor.elements * 4:
            raise EOFError(matrix_tensor.name)
        return struct.unpack(f"<{matrix_tensor.elements}f", raw)

    layer1_attn_norm_tensor = tensor_named("blk.1.attn_norm.weight")
    layer1_short_in_tensor = tensor_named("blk.1.shortconv.in_proj.weight")
    layer1_conv_tensor = tensor_named("blk.1.shortconv.conv.weight")
    layer1_short_out_tensor = tensor_named("blk.1.shortconv.out_proj.weight")
    layer1_ffn_norm_tensor = tensor_named("blk.1.ffn_norm.weight")
    layer1_gate_tensor = tensor_named("blk.1.ffn_gate.weight")
    layer1_up_tensor = tensor_named("blk.1.ffn_up.weight")
    layer1_down_tensor = tensor_named("blk.1.ffn_down.weight")
    layer1_attn_norm_weights = read_f32_tensor(layer1_attn_norm_tensor)
    layer1_conv_weights = read_f32_tensor(layer1_conv_tensor)
    layer1_ffn_norm_weights = read_f32_tensor(layer1_ffn_norm_tensor)
    write_float_tensor(paths.structures / "runtime" / "layers" / "001" / "attn_norm.nbt", layer1_attn_norm_tensor.name, layer1_attn_norm_weights, 0x10100)
    write_float_tensor(paths.structures / "runtime" / "layers" / "001" / "shortconv_conv.nbt", layer1_conv_tensor.name, layer1_conv_weights, 0x10101)
    write_float_tensor(paths.structures / "runtime" / "layers" / "001" / "ffn_norm.nbt", layer1_ffn_norm_tensor.name, layer1_ffn_norm_weights, 0x10102)

    layer1_l2 = sum(value * value for value in layer0_output) ** 0.5
    layer1_scale = 1.0 / ((layer1_l2 * layer1_l2 / 2048 + epsilon) ** 0.5)
    layer1_normalized = [value * layer1_scale * weight for value, weight in zip(layer0_output, layer1_attn_norm_weights)]
    layer1_short_in = compile_quant_matrix(layer1_short_in_tensor, "layers/001/shortconv_in", layer1_normalized)
    layer1_conv_state = [layer1_short_in[index] * layer1_short_in[4096 + index] for index in range(2048)]
    layer1_mixed = [layer1_short_in[2048 + index] * layer1_conv_state[index] * layer1_conv_weights[index * 3 + 2] for index in range(2048)]
    layer1_short_out = compile_quant_matrix(layer1_short_out_tensor, "layers/001/shortconv_out", layer1_mixed)
    layer1_residual = [layer0_output[index] + layer1_short_out[index] for index in range(2048)]
    layer1_ffn_l2 = sum(value * value for value in layer1_residual) ** 0.5
    layer1_ffn_scale = 1.0 / ((layer1_ffn_l2 * layer1_ffn_l2 / 2048 + epsilon) ** 0.5)
    layer1_ffn_input = [value * layer1_ffn_scale * weight for value, weight in zip(layer1_residual, layer1_ffn_norm_weights)]
    layer1_gate = compile_quant_matrix(layer1_gate_tensor, "layers/001/ffn_gate", layer1_ffn_input)
    layer1_up = compile_quant_matrix(layer1_up_tensor, "layers/001/ffn_up", layer1_ffn_input)
    layer1_activated: list[float] = []
    for gate, up in zip(layer1_gate, layer1_up):
        if gate >= 0.0:
            sigmoid = 1.0 / (1.0 + math.exp(-gate))
        else:
            exp_gate = math.exp(gate)
            sigmoid = exp_gate / (1.0 + exp_gate)
        layer1_activated.append(gate * sigmoid * up)
    layer1_down = compile_quant_matrix(layer1_down_tensor, "layers/001/ffn_down", layer1_activated)
    layer1_output = [layer1_residual[index] + layer1_down[index] for index in range(2048)]

    layer2_attn_norm_tensor = tensor_named("blk.2.attn_norm.weight")
    layer2_k_tensor = tensor_named("blk.2.attn_k.weight")
    layer2_k_norm_tensor = tensor_named("blk.2.attn_k_norm.weight")
    layer2_q_tensor = tensor_named("blk.2.attn_q.weight")
    layer2_q_norm_tensor = tensor_named("blk.2.attn_q_norm.weight")
    layer2_v_tensor = tensor_named("blk.2.attn_v.weight")
    layer2_out_tensor = tensor_named("blk.2.attn_output.weight")
    layer2_ffn_norm_tensor = tensor_named("blk.2.ffn_norm.weight")
    layer2_gate_tensor = tensor_named("blk.2.ffn_gate.weight")
    layer2_up_tensor = tensor_named("blk.2.ffn_up.weight")
    layer2_down_tensor = tensor_named("blk.2.ffn_down.weight")
    layer2_attn_norm_weights = read_f32_tensor(layer2_attn_norm_tensor)
    layer2_k_norm_weights = read_f32_tensor(layer2_k_norm_tensor)
    layer2_q_norm_weights = read_f32_tensor(layer2_q_norm_tensor)
    layer2_ffn_norm_weights = read_f32_tensor(layer2_ffn_norm_tensor)
    write_float_tensor(paths.structures / "runtime" / "layers" / "002" / "attn_norm.nbt", layer2_attn_norm_tensor.name, layer2_attn_norm_weights, 0x10200)
    write_float_tensor(paths.structures / "runtime" / "layers" / "002" / "ffn_norm.nbt", layer2_ffn_norm_tensor.name, layer2_ffn_norm_weights, 0x10201)
    write_float_tensor(paths.structures / "runtime" / "layers" / "002" / "attn_k_norm.nbt", layer2_k_norm_tensor.name, layer2_k_norm_weights, 0x10202)
    write_float_tensor(paths.structures / "runtime" / "layers" / "002" / "attn_q_norm.nbt", layer2_q_norm_tensor.name, layer2_q_norm_weights, 0x10203)

    layer2_l2 = sum(value * value for value in layer1_output) ** 0.5
    layer2_scale = 1.0 / ((layer2_l2 * layer2_l2 / 2048 + epsilon) ** 0.5)
    layer2_normalized = [value * layer2_scale * weight for value, weight in zip(layer1_output, layer2_attn_norm_weights)]
    layer2_k = compile_quant_matrix(layer2_k_tensor, "layers/002/attn_k", layer2_normalized)
    layer2_k_norm: list[float] = []
    for head in range(8):
        head_values = layer2_k[head * 64 : (head + 1) * 64]
        head_l2 = sum(value * value for value in head_values) ** 0.5
        head_scale = 1.0 / ((head_l2 * head_l2 / 64 + epsilon) ** 0.5)
        layer2_k_norm.extend(value * head_scale * weight for value, weight in zip(head_values, layer2_k_norm_weights))
    layer2_v = compile_quant_matrix(layer2_v_tensor, "layers/002/attn_v", layer2_normalized)
    layer2_q = compile_quant_matrix(layer2_q_tensor, "layers/002/attn_q", layer2_normalized)
    layer2_q_norm: list[float] = []
    for head in range(32):
        head_values = layer2_q[head * 64 : (head + 1) * 64]
        head_l2 = sum(value * value for value in head_values) ** 0.5
        head_scale = 1.0 / ((head_l2 * head_l2 / 64 + epsilon) ** 0.5)
        layer2_q_norm.extend(value * head_scale * weight for value, weight in zip(head_values, layer2_q_norm_weights))
    layer2_expanded: list[float] = []
    for kv_head in range(8):
        head_values = layer2_v[kv_head * 64 : (kv_head + 1) * 64]
        for _ in range(4):
            layer2_expanded.extend(head_values)
    if len(layer2_expanded) != 2048:
        raise AssertionError(len(layer2_expanded))
    layer2_attn_out = compile_quant_matrix(layer2_out_tensor, "layers/002/attn_out", layer2_expanded)
    layer2_residual = [layer1_output[index] + layer2_attn_out[index] for index in range(2048)]
    layer2_ffn_l2 = sum(value * value for value in layer2_residual) ** 0.5
    layer2_ffn_scale = 1.0 / ((layer2_ffn_l2 * layer2_ffn_l2 / 2048 + epsilon) ** 0.5)
    layer2_ffn_input = [value * layer2_ffn_scale * weight for value, weight in zip(layer2_residual, layer2_ffn_norm_weights)]
    layer2_gate = compile_quant_matrix(layer2_gate_tensor, "layers/002/ffn_gate", layer2_ffn_input)
    layer2_up = compile_quant_matrix(layer2_up_tensor, "layers/002/ffn_up", layer2_ffn_input)
    layer2_activated: list[float] = []
    for gate, up in zip(layer2_gate, layer2_up):
        if gate >= 0.0:
            sigmoid = 1.0 / (1.0 + math.exp(-gate))
        else:
            exp_gate = math.exp(gate)
            sigmoid = exp_gate / (1.0 + exp_gate)
        layer2_activated.append(gate * sigmoid * up)
    layer2_down = compile_quant_matrix(layer2_down_tensor, "layers/002/ffn_down", layer2_activated)
    layer2_output = [layer2_residual[index] + layer2_down[index] for index in range(2048)]

    configure_ffn0_runtime(paths, next_phase="layer1_attn_norm_load", gate_kind=ffn_gate_tensor.type_name, up_kind=ffn_up_tensor.type_name, down_kind=ffn_down_tensor.type_name)
    configure_shortconv_layer_runtime(paths, 1, next_phase="layer2_attn_norm_load", short_in_kind=layer1_short_in_tensor.type_name, short_out_kind=layer1_short_out_tensor.type_name, gate_kind=layer1_gate_tensor.type_name, up_kind=layer1_up_tensor.type_name, down_kind=layer1_down_tensor.type_name)
    configure_attention_layer_runtime(paths, 2, next_phase="layer3_attn_norm_load", key_kind=layer2_k_tensor.type_name, value_kind=layer2_v_tensor.type_name, query_kind=layer2_q_tensor.type_name, output_kind=layer2_out_tensor.type_name, gate_kind=layer2_gate_tensor.type_name, up_kind=layer2_up_tensor.type_name, down_kind=layer2_down_tensor.type_name)

    later_layer_proofs: dict[str, dict] = {}
    current_output = layer2_output
    kv_heads = model.metadata["lfm2.attention.head_count_kv"]
    for layer in range(3, 16):
        structure_root = f"layers/{layer:03d}"
        attn_norm_tensor = tensor_named(f"blk.{layer}.attn_norm.weight")
        ffn_norm_tensor_n = tensor_named(f"blk.{layer}.ffn_norm.weight")
        attn_norm_weights = read_f32_tensor(attn_norm_tensor)
        ffn_norm_weights_n = read_f32_tensor(ffn_norm_tensor_n)
        write_float_tensor(paths.structures / "runtime" / structure_root / "attn_norm.nbt", attn_norm_tensor.name, attn_norm_weights, 0x11000 + layer * 16)
        write_float_tensor(paths.structures / "runtime" / structure_root / "ffn_norm.nbt", ffn_norm_tensor_n.name, ffn_norm_weights_n, 0x11001 + layer * 16)

        operator_l2 = sum(value * value for value in current_output) ** 0.5
        operator_scale = 1.0 / ((operator_l2 * operator_l2 / 2048 + epsilon) ** 0.5)
        operator_input = [value * operator_scale * weight for value, weight in zip(current_output, attn_norm_weights)]
        if kv_heads[layer] == 0:
            short_in_tensor_n = tensor_named(f"blk.{layer}.shortconv.in_proj.weight")
            conv_tensor_n = tensor_named(f"blk.{layer}.shortconv.conv.weight")
            short_out_tensor_n = tensor_named(f"blk.{layer}.shortconv.out_proj.weight")
            conv_weights_n = read_f32_tensor(conv_tensor_n)
            write_float_tensor(paths.structures / "runtime" / structure_root / "shortconv_conv.nbt", conv_tensor_n.name, conv_weights_n, 0x11002 + layer * 16)
            short_in_n = compile_quant_matrix(short_in_tensor_n, f"{structure_root}/shortconv_in", operator_input)
            conv_state_n = [short_in_n[index] * short_in_n[4096 + index] for index in range(2048)]
            core_mixed = [short_in_n[2048 + index] * conv_state_n[index] * conv_weights_n[index * 3 + 2] for index in range(2048)]
            core_output = compile_quant_matrix(short_out_tensor_n, f"{structure_root}/shortconv_out", core_mixed)
            core_proof = {
                "type": "shortconv",
                "projection_first_16": short_in_n[:16],
                "mixed_first_16": core_mixed[:16],
                "output_first_16": core_output[:16],
            }
        else:
            key_tensor_n = tensor_named(f"blk.{layer}.attn_k.weight")
            key_norm_tensor_n = tensor_named(f"blk.{layer}.attn_k_norm.weight")
            query_tensor_n = tensor_named(f"blk.{layer}.attn_q.weight")
            query_norm_tensor_n = tensor_named(f"blk.{layer}.attn_q_norm.weight")
            value_tensor_n = tensor_named(f"blk.{layer}.attn_v.weight")
            output_tensor_n = tensor_named(f"blk.{layer}.attn_output.weight")
            key_norm_weights_n = read_f32_tensor(key_norm_tensor_n)
            query_norm_weights_n = read_f32_tensor(query_norm_tensor_n)
            write_float_tensor(paths.structures / "runtime" / structure_root / "attn_k_norm.nbt", key_norm_tensor_n.name, key_norm_weights_n, 0x11002 + layer * 16)
            write_float_tensor(paths.structures / "runtime" / structure_root / "attn_q_norm.nbt", query_norm_tensor_n.name, query_norm_weights_n, 0x11003 + layer * 16)
            key_n = compile_quant_matrix(key_tensor_n, f"{structure_root}/attn_k", operator_input)
            key_norm_n: list[float] = []
            for head in range(8):
                head_values = key_n[head * 64 : (head + 1) * 64]
                head_l2 = sum(value * value for value in head_values) ** 0.5
                head_scale = 1.0 / ((head_l2 * head_l2 / 64 + epsilon) ** 0.5)
                key_norm_n.extend(value * head_scale * weight for value, weight in zip(head_values, key_norm_weights_n))
            value_n = compile_quant_matrix(value_tensor_n, f"{structure_root}/attn_v", operator_input)
            query_n = compile_quant_matrix(query_tensor_n, f"{structure_root}/attn_q", operator_input)
            query_norm_n: list[float] = []
            for head in range(32):
                head_values = query_n[head * 64 : (head + 1) * 64]
                head_l2 = sum(value * value for value in head_values) ** 0.5
                head_scale = 1.0 / ((head_l2 * head_l2 / 64 + epsilon) ** 0.5)
                query_norm_n.extend(value * head_scale * weight for value, weight in zip(head_values, query_norm_weights_n))
            expanded_n: list[float] = []
            for kv_head in range(8):
                head_values = value_n[kv_head * 64 : (kv_head + 1) * 64]
                for _ in range(4):
                    expanded_n.extend(head_values)
            core_output = compile_quant_matrix(output_tensor_n, f"{structure_root}/attn_out", expanded_n)
            core_proof = {
                "type": "attention_single_token",
                "key_norm_first_16": key_norm_n[:16],
                "query_norm_first_16": query_norm_n[:16],
                "value_first_16": value_n[:16],
                "expanded_first_16": expanded_n[:16],
                "output_first_16": core_output[:16],
            }

        residual_n = [current_output[index] + core_output[index] for index in range(2048)]
        ffn_l2_n = sum(value * value for value in residual_n) ** 0.5
        ffn_scale_n = 1.0 / ((ffn_l2_n * ffn_l2_n / 2048 + epsilon) ** 0.5)
        ffn_input_n = [value * ffn_scale_n * weight for value, weight in zip(residual_n, ffn_norm_weights_n)]
        gate_tensor_n = tensor_named(f"blk.{layer}.ffn_gate.weight")
        up_tensor_n = tensor_named(f"blk.{layer}.ffn_up.weight")
        down_tensor_n = tensor_named(f"blk.{layer}.ffn_down.weight")
        gate_n = compile_quant_matrix(gate_tensor_n, f"{structure_root}/ffn_gate", ffn_input_n)
        up_n = compile_quant_matrix(up_tensor_n, f"{structure_root}/ffn_up", ffn_input_n)
        activated_n: list[float] = []
        for gate, up in zip(gate_n, up_n):
            if gate >= 0.0:
                sigmoid = 1.0 / (1.0 + math.exp(-gate))
            else:
                exp_gate = math.exp(gate)
                sigmoid = exp_gate / (1.0 + exp_gate)
            activated_n.append(gate * sigmoid * up)
        down_n = compile_quant_matrix(down_tensor_n, f"{structure_root}/ffn_down", activated_n)
        output_n = [residual_n[index] + down_n[index] for index in range(2048)]
        later_layer_proofs[str(layer)] = {
            "operator_norm": {"l2": operator_l2, "scale": operator_scale, "first_16": operator_input[:16]},
            "core": core_proof,
            "residual_first_16": residual_n[:16],
            "ffn_norm": {"l2": ffn_l2_n, "scale": ffn_scale_n, "first_16": ffn_input_n[:16]},
            "ffn_gate_first_16": gate_n[:16],
            "ffn_up_first_16": up_n[:16],
            "swiglu_first_16": activated_n[:16],
            "ffn_down_first_16": down_n[:16],
            "output": {"first_16": output_n[:16], "last_16": output_n[-16:], "sum": sum(output_n), "sum_squares": sum(value * value for value in output_n)},
        }
        next_phase = f"layer{layer + 1}_attn_norm_load" if layer < 15 else "prefill_token_complete"
        if kv_heads[layer] == 0:
            configure_shortconv_layer_runtime(paths, layer, next_phase=next_phase, short_in_kind=short_in_tensor_n.type_name, short_out_kind=short_out_tensor_n.type_name, gate_kind=gate_tensor_n.type_name, up_kind=up_tensor_n.type_name, down_kind=down_tensor_n.type_name)
        else:
            configure_attention_layer_runtime(paths, layer, next_phase=next_phase, write_attention_providers=False, key_kind=key_tensor_n.type_name, value_kind=value_tensor_n.type_name, query_kind=query_tensor_n.type_name, output_kind=output_tensor_n.type_name, gate_kind=gate_tensor_n.type_name, up_kind=up_tensor_n.type_name, down_kind=down_tensor_n.type_name)
        current_output = output_n

    final_norm_tensor = tensor_named("token_embd_norm.weight")
    final_norm_weights = read_f32_tensor(final_norm_tensor)
    write_float_tensor(paths.structures / "runtime" / "final_norm.nbt", final_norm_tensor.name, final_norm_weights, 0x12000)
    final_l2 = sum(value * value for value in current_output) ** 0.5
    final_scale = 1.0 / ((final_l2 * final_l2 / 2048 + epsilon) ** 0.5)
    final_normalized = [value * final_scale * weight for value, weight in zip(current_output, final_norm_weights)]
    logits = compile_quant_matrix(tensor, "lm_head", final_normalized)
    top_token_ids = sorted(range(len(logits)), key=lambda token_id: (-logits[token_id], token_id))[:5]
    tokenizer_tokens = model.metadata["tokenizer.ggml.tokens"]
    tokenizer_types = model.metadata["tokenizer.ggml.token_type"]
    write_string_table(
        paths.structures / "runtime" / "tokenizer.nbt",
        "tokenizer.ggml.tokens.display",
        display_vocabulary(tokenizer_tokens, tokenizer_types),
        0x12001,
    )
    write_character_token_table(
        paths.structures / "runtime" / "input_chars.nbt",
        character_byte_token_table(tokenizer_tokens),
        0x12003,
    )
    rope_base = float(model.metadata["lfm2.rope.freq_base"])
    rope_positions = 256
    rope_cos: list[list[float]] = []
    rope_sin: list[list[float]] = []
    for position in range(rope_positions):
        angles = [position / (rope_base ** ((2 * lane) / 64)) for lane in range(32)]
        rope_cos.append([math.cos(angle) for angle in angles])
        rope_sin.append([math.sin(angle) for angle in angles])
    write_rope_table(paths.structures / "runtime" / "rope.nbt", rope_cos, rope_sin, 0x12002)
    configure_final_norm_runtime(paths)
    configure_tokenizer_runtime(paths)
    prefix_tokens, suffix_tokens = _chat_wrapper_tokens(tokenizer_tokens)
    configure_minecraft_input_runtime(paths, prefix_tokens, suffix_tokens)
    configure_rope_table_runtime(paths)
    configure_lm_head_runtime(paths, rows=len(logits))
    configure_token_embedding_runtime(paths, vocab_size=len(logits), next_phase="rms0_load")
    configure_generation_runtime(paths)
    configure_prefill_runtime(paths)
    proof = {
        "model_sha256": lock["sha256"],
        "tensor": tensor.name,
        "token": 1,
        "first_block": first_block,
        "block_count": 8,
        "width": len(values),
        "first_16": values[:16],
        "last_16": values[-16:],
        "sum": sum(values),
        "sum_squares": sum(value * value for value in values),
        "rms0": {
            "epsilon": epsilon,
            "l2": l2,
            "scale": scale,
            "first_16": normalized[:16],
            "last_16": normalized[-16:],
            "sum": sum(normalized),
            "sum_squares": sum(value * value for value in normalized),
        },
        "shortconv0_in": {
            "shape": [6144, 2048],
            "chunk_count": 24,
            "first_16": short_outputs[:16],
            "last_16": short_outputs[-16:],
            "sum": sum(short_outputs),
            "sum_squares": sum(value * value for value in short_outputs),
        },
        "shortconv0_mix": {
            "conv_kernel_index": 2,
            "first_16": mixed[:16],
            "last_16": mixed[-16:],
            "sum": sum(mixed),
            "sum_squares": sum(value * value for value in mixed),
        },
        "shortconv0_out": {"shape": [2048, 2048], "first_16": out_outputs[:16], "last_16": out_outputs[-16:], "sum": sum(out_outputs)},
        "residual0": {"first_16": residual[:16], "last_16": residual[-16:]},
        "ffn_norm0": {"l2": ffn_l2, "scale": ffn_scale, "first_16": ffn_input[:16], "last_16": ffn_input[-16:]},
        "ffn_gate0": {"shape": [8192, 2048], "first_16": ffn_gate[:16], "last_16": ffn_gate[-16:]},
        "ffn_up0": {"shape": [8192, 2048], "first_16": ffn_up[:16], "last_16": ffn_up[-16:]},
        "swiglu0": {"first_16": ffn_activated[:16], "last_16": ffn_activated[-16:]},
        "ffn_down0": {"shape": [2048, 8192], "first_16": ffn_down[:16], "last_16": ffn_down[-16:]},
        "layer0_output": {
            "first_16": layer0_output[:16],
            "last_16": layer0_output[-16:],
            "sum": sum(layer0_output),
            "sum_squares": sum(value * value for value in layer0_output),
        },
        "layer1": {
            "attn_norm": {"l2": layer1_l2, "scale": layer1_scale, "first_16": layer1_normalized[:16], "last_16": layer1_normalized[-16:]},
            "shortconv_in": {"first_16": layer1_short_in[:16], "last_16": layer1_short_in[-16:]},
            "shortconv_mix": {"first_16": layer1_mixed[:16], "last_16": layer1_mixed[-16:]},
            "shortconv_out": {"first_16": layer1_short_out[:16], "last_16": layer1_short_out[-16:]},
            "residual": {"first_16": layer1_residual[:16], "last_16": layer1_residual[-16:]},
            "ffn_norm": {"l2": layer1_ffn_l2, "scale": layer1_ffn_scale, "first_16": layer1_ffn_input[:16], "last_16": layer1_ffn_input[-16:]},
            "ffn_gate": {"first_16": layer1_gate[:16], "last_16": layer1_gate[-16:]},
            "ffn_up": {"first_16": layer1_up[:16], "last_16": layer1_up[-16:]},
            "swiglu": {"first_16": layer1_activated[:16], "last_16": layer1_activated[-16:]},
            "ffn_down": {"first_16": layer1_down[:16], "last_16": layer1_down[-16:]},
            "output": {"first_16": layer1_output[:16], "last_16": layer1_output[-16:], "sum": sum(layer1_output), "sum_squares": sum(value * value for value in layer1_output)},
        },
        "layer2": {
            "attn_norm": {"l2": layer2_l2, "scale": layer2_scale, "first_16": layer2_normalized[:16], "last_16": layer2_normalized[-16:]},
            "key_norm": {"first_16": layer2_k_norm[:16], "last_16": layer2_k_norm[-16:]},
            "value": {"first_16": layer2_v[:16], "last_16": layer2_v[-16:]},
            "expanded_value": {"first_16": layer2_expanded[:16], "last_16": layer2_expanded[-16:]},
            "attention_output": {"first_16": layer2_attn_out[:16], "last_16": layer2_attn_out[-16:]},
            "residual": {"first_16": layer2_residual[:16], "last_16": layer2_residual[-16:]},
            "ffn_norm": {"l2": layer2_ffn_l2, "scale": layer2_ffn_scale, "first_16": layer2_ffn_input[:16], "last_16": layer2_ffn_input[-16:]},
            "ffn_gate": {"first_16": layer2_gate[:16], "last_16": layer2_gate[-16:]},
            "ffn_up": {"first_16": layer2_up[:16], "last_16": layer2_up[-16:]},
            "swiglu": {"first_16": layer2_activated[:16], "last_16": layer2_activated[-16:]},
            "ffn_down": {"first_16": layer2_down[:16], "last_16": layer2_down[-16:]},
            "output": {"first_16": layer2_output[:16], "last_16": layer2_output[-16:], "sum": sum(layer2_output), "sum_squares": sum(value * value for value in layer2_output)},
        },
        "layers_3_15": later_layer_proofs,
        "final_norm": {
            "l2": final_l2,
            "scale": final_scale,
            "first_16": final_normalized[:16],
            "last_16": final_normalized[-16:],
            "sum": sum(final_normalized),
            "sum_squares": sum(value * value for value in final_normalized),
        },
        "lm_head": {
            "shape": [65536, 2048],
            "argmax_token_id": top_token_ids[0],
            "argmax_piece": tokenizer_tokens[top_token_ids[0]],
            "argmax_logit": logits[top_token_ids[0]],
            "top_5": [
                {"token_id": token_id, "piece": tokenizer_tokens[token_id], "logit": logits[token_id]}
                for token_id in top_token_ids
            ],
        },
    }
    (paths.root / "bos-embedding-proof.json").write_text(json.dumps(proof, indent=2) + "\n", encoding="utf-8")
    return proof


def _validate_locked_file(source: Path, lock: dict) -> None:
    if source.name != lock["filename"]:
        raise ValueError(f"model filename mismatch: {source.name}")
    size = source.stat().st_size
    if size != lock["size"]:
        raise ValueError(f"model size mismatch: {size} != {lock['size']}")
    digest = file_sha256(source)
    if digest != lock["sha256"]:
        raise ValueError(f"model SHA-256 mismatch: {digest}")


def _validate_model(model: GGUFFile) -> None:
    failures = []
    for key, expected in EXPECTED.items():
        actual = model.metadata.get(key)
        if actual != expected:
            failures.append(f"{key}: {actual!r} != {expected!r}")
    if failures:
        raise ValueError("unexpected LFM2 model:\n" + "\n".join(failures))
    observed = {tensor.type_name for tensor in model.tensors}
    unsupported = observed - {"F32", *QUANT_LAYOUTS}
    if unsupported or "F32" not in observed or "Q6_K" not in observed:
        raise ValueError(f"unexpected tensor type set: {sorted(observed)}")


def _execution_plan(model: GGUFFile) -> list[dict]:
    plan: list[dict] = [{"op": "embedding", "tensor": "token_embd.weight", "token": 1}]
    kv_heads = model.metadata["lfm2.attention.head_count_kv"]
    tensor_names = {tensor.name for tensor in model.tensors}
    for layer in range(16):
        plan.append({"op": "rms_norm", "tensor": f"blk.{layer}.attn_norm.weight"})
        if kv_heads[layer]:
            plan.append(
                {
                    "op": "attention",
                    "layer": layer,
                    "q": f"blk.{layer}.attn_q.weight",
                    "q_norm": f"blk.{layer}.attn_q_norm.weight",
                    "k": f"blk.{layer}.attn_k.weight",
                    "k_norm": f"blk.{layer}.attn_k_norm.weight",
                    "v": f"blk.{layer}.attn_v.weight",
                    "output": f"blk.{layer}.attn_output.weight",
                }
            )
        else:
            plan.append(
                {
                    "op": "shortconv",
                    "layer": layer,
                    "in_proj": f"blk.{layer}.shortconv.in_proj.weight",
                    "conv": f"blk.{layer}.shortconv.conv.weight",
                    "out_proj": f"blk.{layer}.shortconv.out_proj.weight",
                }
            )
        plan.extend(
            [
                {"op": "residual"},
                {"op": "rms_norm", "tensor": f"blk.{layer}.ffn_norm.weight"},
                {
                    "op": "swiglu_ffn",
                    "layer": layer,
                    "gate": f"blk.{layer}.ffn_gate.weight",
                    "up": f"blk.{layer}.ffn_up.weight",
                    "down": f"blk.{layer}.ffn_down.weight",
                },
                {"op": "residual"},
            ]
        )
    plan.extend(
        [
            {"op": "rms_norm", "tensor": "token_embd_norm.weight"},
            {"op": "lm_head_tied", "tensor": "token_embd.weight", "rows": 65536},
            {"op": "argmax", "tie_break": "lowest_token_id"},
        ]
    )
    referenced = {
        value
        for operation in plan
        for key, value in operation.items()
        if key not in {"op", "layer", "token", "rows", "tie_break"}
        and isinstance(value, str)
        and "." in value
    }
    missing = sorted(referenced - tensor_names)
    if missing:
        raise ValueError(f"execution plan references missing tensors: {missing}")
    return plan
