from __future__ import annotations

from typing import Any


Provider = dict[str, Any] | float | int | str


def int_storage(path: str) -> dict[str, Any]:
    return {
        "type": "minecraft:storage",
        "storage": "lfm:scratch",
        "path": path,
        "fallback": 0,
    }


def float_storage(path: str) -> dict[str, Any]:
    return {
        "type": "minecraft:storage",
        "storage": "lfm:scratch",
        "path": path,
        "fallback": 0.0,
    }


def floor_div(left: Provider, right: Provider) -> dict[str, Any]:
    return {"type": "minecraft:floor_div", "left": left, "right": right}


def floor_mod(left: Provider, right: Provider) -> dict[str, Any]:
    return {"type": "minecraft:floor_mod", "left": left, "right": right}


def int_add(*values: Provider) -> dict[str, Any]:
    return {"type": "minecraft:add", "inputs": list(values)}


def int_mul(*values: Provider) -> dict[str, Any]:
    return {"type": "minecraft:mul", "inputs": list(values)}


def int_sub(left: Provider, right: Provider) -> dict[str, Any]:
    return {"type": "minecraft:sub", "left": left, "right": right}


def from_int(value: Provider) -> dict[str, Any]:
    return {"type": "minecraft:from_int", "input": value}


def mul(*values: Provider) -> dict[str, Any]:
    return {"type": "minecraft:mul", "inputs": list(values)}


def add(*values: Provider) -> dict[str, Any]:
    return {"type": "minecraft:add", "inputs": list(values)}


def sub(left: Provider, right: Provider) -> dict[str, Any]:
    return {"type": "minecraft:sub", "left": left, "right": right}


def div(left: Provider, right: Provider) -> dict[str, Any]:
    return {"type": "minecraft:div", "left": left, "right": right}


def sqrt(value: Provider) -> dict[str, Any]:
    return {"type": "minecraft:sqrt", "input": value}


def negate(value: Provider) -> dict[str, Any]:
    return {"type": "minecraft:negate", "input": value}


def pow_value(base: Provider, exponent: Provider) -> dict[str, Any]:
    return {"type": "minecraft:pow", "base": base, "exponent": exponent}


def length(*values: Provider) -> dict[str, Any]:
    return {"type": "minecraft:length", "inputs": list(values)}


def packed_byte(path: str, byte_index: int) -> dict[str, Any]:
    word_index, byte_offset = divmod(byte_index, 4)
    word = int_storage(f"{path}[{word_index}]")
    shifted = floor_div(word, 1 << (8 * byte_offset)) if byte_offset else word
    return floor_mod(shifted, 256)


def q4_0_term(
    position: int,
    activation_path: str = "activation",
    activation_offset: int = 0,
) -> dict[str, Any]:
    if not 0 <= position < 256:
        raise ValueError(position)
    block, local = divmod(position, 32)
    lane = local % 16
    packed = packed_byte("weight.qs", block * 16 + lane)
    nibble = floor_mod(floor_div(packed, 16) if local >= 16 else packed, 16)
    quant = int_sub(nibble, 8)
    weight = mul(float_storage(f"weight.d[{block}]"), from_int(quant))
    return mul(float_storage(f"{activation_path}[{activation_offset + position}]"), weight)


def q2k_quant(position: int) -> dict[str, Any]:
    if not 0 <= position < 256:
        raise ValueError(position)
    group, lane = divmod(position, 16)
    local_group = group % 8
    q_index = (group // 8) * 32 + (local_group % 2) * 16 + lane
    shift = (local_group // 2) * 2
    return floor_mod(floor_div(packed_byte("weight.qs", q_index), 1 << shift), 4)


def q2k_scale_nibble(group: int, high: bool) -> dict[str, Any]:
    scale_byte = packed_byte("weight.scales", group)
    return floor_div(scale_byte, 16) if high else floor_mod(scale_byte, 16)


def q2k_term(
    position: int,
    activation_path: str = "activation",
    activation_offset: int = 0,
) -> dict[str, Any]:
    group = position // 16
    quant = from_int(q2k_quant(position))
    scale = from_int(q2k_scale_nibble(group, high=False))
    minimum = from_int(q2k_scale_nibble(group, high=True))
    weight = sub(
        mul(float_storage("weight.d"), scale, quant),
        mul(float_storage("weight.dmin"), minimum),
    )
    return mul(float_storage(f"{activation_path}[{activation_offset + position}]"), weight)


def q2k_dot_provider(
    width: int = 256,
    include_accumulator: bool = True,
    activation_path: str = "activation",
    activation_offset: int = 0,
) -> dict[str, Any]:
    if width not in (32, 64, 128, 256):
        raise ValueError("kernel width must be 32, 64, 128, or 256")
    terms = [q2k_term(position, activation_path, activation_offset) for position in range(width)]
    partials = [add(*terms[index : index + 16]) for index in range(0, len(terms), 16)]
    if include_accumulator:
        return add(float_storage("row_acc"), *partials)
    return add(*partials)


def q3k_term(position: int, activation_path: str = "activation", activation_offset: int = 0) -> dict[str, Any]:
    group, lane = divmod(position, 16)
    local_group = group % 8
    q_index = (group // 8) * 32 + (local_group % 2) * 16 + lane
    shift = (local_group // 2) * 2
    low = floor_mod(floor_div(packed_byte("weight.qs", q_index), 1 << shift), 4)
    mask = 1 << (group // 2)
    high = floor_mod(
        floor_div(packed_byte("weight.hmask", (local_group % 2) * 16 + lane), mask),
        2,
    )
    quant = int_sub(int_add(low, int_mul(high, 4)), 4)
    weight = mul(
        float_storage("weight.d"),
        from_int(int_storage(f"weight.scales[{group}]")),
        from_int(quant),
    )
    return mul(float_storage(f"{activation_path}[{activation_offset + position}]"), weight)


def q6k_value(position: int) -> dict[str, Any]:
    if not 0 <= position < 256:
        raise ValueError(position)
    half, local = divmod(position, 128)
    segment, lane = divmod(local, 32)
    ql_index = half * 64 + lane + (32 if segment in (1, 3) else 0)
    ql_byte = packed_byte("weight.ql", ql_index)
    low = floor_mod(floor_div(ql_byte, 16 if segment >= 2 else 1), 16)
    qh_byte = packed_byte("weight.qh", half * 32 + lane)
    high = floor_mod(floor_div(qh_byte, 1 << (segment * 2)), 4)
    quant = int_sub(int_add(low, int_mul(high, 16)), 32)
    scale_index = half * 8 + lane // 16 + segment * 2
    return mul(
        float_storage("weight.d"),
        from_int(int_storage(f"weight.scales[{scale_index}]")),
        from_int(quant),
    )


def q6k_term(position: int, activation_path: str = "activation", activation_offset: int = 0) -> dict[str, Any]:
    return mul(float_storage(f"{activation_path}[{activation_offset + position}]"), q6k_value(position))


def quant_dot_provider(kind: str, width: int = 256, include_accumulator: bool = True, activation_path: str = "activation", activation_offset: int = 0) -> dict[str, Any]:
    if width not in (32, 64, 128, 256):
        raise ValueError("kernel width must be 32, 64, 128, or 256")
    term_factory = {
        "Q4_0": q4_0_term,
        "Q2_K": q2k_term,
        "Q3_K": q3k_term,
        "Q6_K": q6k_term,
    }.get(kind)
    if term_factory is None:
        raise ValueError(f"unsupported quant type: {kind}")
    terms = [term_factory(position, activation_path, activation_offset) for position in range(width)]
    partials = [add(*terms[index : index + 16]) for index in range(0, len(terms), 16)]
    if include_accumulator:
        return add(float_storage("row_acc"), *partials)
    return add(*partials)


def rms_l2_provider(width: int = 2048, input_path: str = "activation") -> dict[str, Any]:
    if width <= 0:
        raise ValueError("RMS width must be positive")
    return length(*(float_storage(f"{input_path}[{index}]") for index in range(width)))


def rms_scale_provider(width: int = 2048, epsilon: float = 1e-5, l2_path: str = "rms0.l2") -> dict[str, Any]:
    if width <= 0 or epsilon <= 0:
        raise ValueError("invalid RMSNorm parameters")
    l2 = float_storage(l2_path)
    mean_square = div(mul(l2, l2), float(width))
    return div(1.0, sqrt(add(mean_square, float(epsilon))))


def rms_apply_provider(
    index: int,
    input_path: str = "activation",
    scale_path: str = "rms0.scale",
    weight_path: str = "rms0.weight.values",
) -> dict[str, Any]:
    if index < 0:
        raise ValueError(index)
    return mul(
        float_storage(f"{input_path}[{index}]"),
        float_storage(scale_path),
        float_storage(f"{weight_path}[{index}]"),
    )


def residual_provider(index: int) -> dict[str, Any]:
    if index < 0:
        raise ValueError(index)
    return add(float_storage(f"activation[{index}]"), float_storage(f"shortconv0.output[{index}]"))


def swiglu_provider(index: int, gate_path: str = "ffn0.gate", up_path: str = "ffn0.up") -> dict[str, Any]:
    if index < 0:
        raise ValueError(index)
    gate = float_storage(f"{gate_path}[{index}]")
    silu = div(gate, add(1.0, pow_value(2.718281828459045, negate(gate))))
    return mul(silu, float_storage(f"{up_path}[{index}]"))


def add_paths_provider(index: int, left_path: str, right_path: str) -> dict[str, Any]:
    if index < 0:
        raise ValueError(index)
    return add(float_storage(f"{left_path}[{index}]"), float_storage(f"{right_path}[{index}]"))


def shortconv_state_provider(index: int, width: int = 2048) -> dict[str, Any]:
    if not 0 <= index < width:
        raise ValueError(index)
    return mul(
        float_storage(f"shortconv0.in_proj[{index}]"),
        float_storage(f"shortconv0.in_proj[{2 * width + index}]"),
    )


def shortconv_mix_provider(index: int, width: int = 2048, kernel_size: int = 3) -> dict[str, Any]:
    if not 0 <= index < width or kernel_size <= 0:
        raise ValueError(index)
    current = float_storage(f"shortconv0.conv_state[{index}]")
    convolved = add(
        mul(
            float_storage(f"shortconv0.prev2[{index}]"),
            float_storage(f"shortconv0.conv_weight.values[{index * kernel_size}]"),
        ),
        mul(
            float_storage(f"shortconv0.prev1[{index}]"),
            float_storage(f"shortconv0.conv_weight.values[{index * kernel_size + 1}]"),
        ),
        mul(
            current,
            float_storage(f"shortconv0.conv_weight.values[{index * kernel_size + kernel_size - 1}]"),
        ),
    )
    return mul(
        float_storage(f"shortconv0.in_proj[{width + index}]"),
        convolved,
    )


def rope_apply_provider(index: int, head_width: int = 64) -> dict[str, Any]:
    if head_width % 2 or not 0 <= index < head_width:
        raise ValueError(index)
    half = head_width // 2
    if index < half:
        return sub(
            mul(float_storage(f"attn.head_input[{index}]"), float_storage(f"attn.rope.cos[{index}]")),
            mul(float_storage(f"attn.head_input[{index + half}]"), float_storage(f"attn.rope.sin[{index}]")),
        )
    lane = index - half
    return add(
        mul(float_storage(f"attn.head_input[{index}]"), float_storage(f"attn.rope.cos[{lane}]")),
        mul(float_storage(f"attn.head_input[{lane}]"), float_storage(f"attn.rope.sin[{lane}]")),
    )


def attention_score_provider(head_width: int = 64) -> dict[str, Any]:
    if head_width <= 0:
        raise ValueError(head_width)
    terms = [
        mul(float_storage(f"attn.head_query[{index}]"), float_storage(f"attn.head_key[{index}]"))
        for index in range(head_width)
    ]
    partials = [add(*terms[index : index + 16]) for index in range(0, head_width, 16)]
    return mul(head_width ** -0.5, add(*partials))


def attention_exp_provider() -> dict[str, Any]:
    return pow_value(2.718281828459045, sub(float_storage("attn.selected_score"), float_storage("attn.score_max")))


def attention_sum_provider() -> dict[str, Any]:
    return add(float_storage("attn.softmax_sum"), float_storage("attn.temp_exp"))


def attention_weight_provider() -> dict[str, Any]:
    return div(float_storage("attn.temp_exp"), float_storage("attn.softmax_sum"))


def attention_weighted_value_provider(index: int, head_width: int = 64) -> dict[str, Any]:
    if not 0 <= index < head_width:
        raise ValueError(index)
    return add(
        float_storage(f"attn.head_output[{index}]"),
        mul(float_storage("attn.temp_weight"), float_storage(f"attn.selected_value[{index}]")),
    )
