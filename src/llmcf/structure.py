from __future__ import annotations

from pathlib import Path
from typing import Sequence

from .nbt import (
    TAG_COMPOUND,
    TAG_DOUBLE,
    TAG_FLOAT,
    TAG_INT,
    TAG_STRING,
    Double,
    Float,
    Int,
    IntArray,
    ListTag,
    Long,
    write_gzip_nbt,
)
from .q2k import Q2KBlock, Q3KBlock, Q6KBlock, Q40BlockGroup


DEFAULT_DATA_VERSION = 4440


def quant_chunk_structure(
    tensor: str,
    first_block: int,
    blocks: Sequence[Q2KBlock | Q3KBlock | Q6KBlock | Q40BlockGroup],
    data_version: int = DEFAULT_DATA_VERSION,
) -> dict:
    block_tags = [quant_block_tag(block) for block in reversed(blocks)]
    marker_data = {
        "schema": Int(1),
        "tensor": tensor,
        "first_block": Long(first_block),
        "block_count": Int(len(block_tags)),
        "blocks": ListTag(TAG_COMPOUND, block_tags),
    }
    return marker_structure(marker_data, first_block, data_version)


def marker_structure(data: dict, uuid_seed: int, data_version: int = DEFAULT_DATA_VERSION) -> dict:
    entity_nbt = {
        "id": "minecraft:marker",
        "Tags": ListTag(TAG_STRING, ["lfm.weight_chunk"]),
        "UUID": IntArray((0x4C464D00, uuid_seed & 0x7FFFFFFF, 0, 1)),
        "data": data,
    }
    return {
        "DataVersion": Int(data_version),
        "size": ListTag(TAG_INT, [Int(1), Int(1), Int(1)]),
        "palette": ListTag(TAG_COMPOUND, []),
        "blocks": ListTag(TAG_COMPOUND, []),
        "entities": ListTag(
            TAG_COMPOUND,
            [
                {
                    "pos": ListTag(TAG_DOUBLE, [Double(0.5), Double(0.0), Double(0.5)]),
                    "blockPos": ListTag(TAG_INT, [Int(0), Int(0), Int(0)]),
                    "nbt": entity_nbt,
                }
            ],
        ),
    }


def write_float_tensor(
    path: str | Path,
    tensor: str,
    values: Sequence[float],
    uuid_seed: int,
    data_version: int = DEFAULT_DATA_VERSION,
) -> None:
    data = {
        "schema": Int(1),
        "tensor": tensor,
        "value_count": Int(len(values)),
        "values": ListTag(5, [Float(value) for value in values]),
    }
    write_gzip_nbt(path, marker_structure(data, uuid_seed, data_version))


def write_string_table(
    path: str | Path,
    name: str,
    values: Sequence[str],
    uuid_seed: int,
    data_version: int = DEFAULT_DATA_VERSION,
) -> None:
    data = {
        "schema": Int(1),
        "name": name,
        "value_count": Int(len(values)),
        "values": ListTag(TAG_STRING, list(values)),
    }
    write_gzip_nbt(path, marker_structure(data, uuid_seed, data_version))


def write_character_token_table(
    path: str | Path,
    values: dict[str, Sequence[int]],
    uuid_seed: int,
    data_version: int = DEFAULT_DATA_VERSION,
) -> None:
    table = {
        char: {
            "n": Int(len(token_ids)),
            **{f"b{index}": Int(token_id) for index, token_id in enumerate(token_ids)},
        }
        for char, token_ids in values.items()
    }
    data = {
        "schema": Int(1),
        "name": "minecraft_input_utf8_byte_tokens",
        "value_count": Int(len(table)),
        "values": table,
    }
    write_gzip_nbt(path, marker_structure(data, uuid_seed, data_version))


def write_rope_table(
    path: str | Path,
    cos_rows: Sequence[Sequence[float]],
    sin_rows: Sequence[Sequence[float]],
    uuid_seed: int,
    data_version: int = DEFAULT_DATA_VERSION,
) -> None:
    if len(cos_rows) != len(sin_rows) or not cos_rows:
        raise ValueError("RoPE cosine and sine tables must have equal non-zero length")
    width = len(cos_rows[0])
    if width == 0 or any(len(row) != width for row in cos_rows) or any(len(row) != width for row in sin_rows):
        raise ValueError("RoPE rows must have a consistent non-zero width")
    positions = [
        {
            "cos": ListTag(TAG_FLOAT, [Float(value) for value in cos_row]),
            "sin": ListTag(TAG_FLOAT, [Float(value) for value in sin_row]),
        }
        for cos_row, sin_row in zip(cos_rows, sin_rows)
    ]
    data = {
        "schema": Int(1),
        "position_count": Int(len(positions)),
        "half_width": Int(width),
        "positions": ListTag(TAG_COMPOUND, positions),
    }
    write_gzip_nbt(path, marker_structure(data, uuid_seed, data_version))


def quant_block_tag(block: Q2KBlock | Q3KBlock | Q6KBlock | Q40BlockGroup) -> dict:
    if isinstance(block, Q40BlockGroup):
        return {
            "d": ListTag(TAG_FLOAT, [Float(value) for value in block.d]),
            "qs": IntArray(block.packed_qs()),
        }
    if isinstance(block, Q2KBlock):
        return {
            "scales": IntArray(block.packed_scales()),
            "qs": IntArray(block.packed_qs()),
            "d": Float(block.d),
            "dmin": Float(block.dmin),
        }
    if isinstance(block, Q3KBlock):
        return {
            "hmask": IntArray(block.packed_hmask()),
            "qs": IntArray(block.packed_qs()),
            "scales": IntArray(block.scales),
            "d": Float(block.d),
        }
    if isinstance(block, Q6KBlock):
        return {
            "ql": IntArray(block.packed_ql()),
            "qh": IntArray(block.packed_qh()),
            "scales": IntArray(block.scales),
            "d": Float(block.d),
        }
    raise TypeError(type(block))


def write_quant_chunk(
    path: str | Path,
    tensor: str,
    first_block: int,
    blocks: Sequence[Q2KBlock | Q3KBlock | Q6KBlock | Q40BlockGroup],
    data_version: int = DEFAULT_DATA_VERSION,
) -> None:
    write_gzip_nbt(
        path,
        quant_chunk_structure(tensor, first_block, blocks, data_version),
    )


def q2k_chunk_structure(
    tensor: str,
    first_block: int,
    blocks: Sequence[Q2KBlock],
    data_version: int = DEFAULT_DATA_VERSION,
) -> dict:
    return quant_chunk_structure(tensor, first_block, blocks, data_version)


def write_q2k_chunk(
    path: str | Path,
    tensor: str,
    first_block: int,
    blocks: Sequence[Q2KBlock],
    data_version: int = DEFAULT_DATA_VERSION,
) -> None:
    write_quant_chunk(path, tensor, first_block, blocks, data_version)
