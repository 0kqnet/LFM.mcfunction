from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import shutil
import struct
from typing import Any

from .providers import (
    q6k_value,
    quant_dot_provider,
    rms_apply_provider,
    rms_l2_provider,
    rms_scale_provider,
    shortconv_mix_provider,
    shortconv_state_provider,
    residual_provider,
    swiglu_provider,
    add_paths_provider,
    rope_apply_provider,
    attention_score_provider,
    attention_exp_provider,
    attention_sum_provider,
    attention_weight_provider,
    attention_weighted_value_provider,
)
from .q2k import Q40BlockGroup, Q2KBlock, Q3KBlock, Q6KBlock
from .structure import write_quant_chunk


PACK_FORMAT = 119
DEFAULT_STEPS_PER_TICK = 32


@dataclass(frozen=True)
class PackPaths:
    root: Path

    @property
    def namespace(self) -> Path:
        return self.root / "data" / "lfm"

    @property
    def functions(self) -> Path:
        return self.namespace / "function"

    @property
    def float_providers(self) -> Path:
        return self.namespace / "context_float_provider"

    @property
    def int_providers(self) -> Path:
        return self.namespace / "context_int_provider"

    @property
    def structures(self) -> Path:
        return self.namespace / "structure"


def generate_kernel_pack(
    destination: str | Path,
    clean: bool = False,
    kernel_widths: tuple[int, ...] = (32, 64, 128, 256),
    steps_per_tick: int = DEFAULT_STEPS_PER_TICK,
) -> PackPaths:
    if steps_per_tick < 1:
        raise ValueError("steps_per_tick must be positive")
    paths = PackPaths(Path(destination))
    if clean and paths.root.exists():
        shutil.rmtree(paths.root)
    paths.root.mkdir(parents=True, exist_ok=True)
    _json(
        paths.root / "pack.mcmeta",
        {
            "pack": {
                "description": "LFM2.5 quantized inference for Minecraft 26.3",
                "min_format": [PACK_FORMAT, 0],
                "max_format": [PACK_FORMAT, 0],
            }
        },
    )
    _json(
        paths.root / "data" / "minecraft" / "tags" / "function" / "load.json",
        {"values": ["lfm:load"]},
    )
    _json(
        paths.root / "data" / "minecraft" / "tags" / "function" / "tick.json",
        {"values": ["lfm:tick"]},
    )
    _json_dimension(paths)
    for kind in ("Q4_0", "Q2_K", "Q3_K", "Q6_K"):
        for width in kernel_widths:
            _json(
                paths.float_providers / "kernel" / f"{kind.lower().replace('_', '')}_{width}.json",
                quant_dot_provider(kind, width),
            )

    _text(
        paths.functions / "load.mcfunction",
        "\n".join(
            [
                "scoreboard objectives add lfm.ui dummy",
                'execute unless data storage lfm:ui {initialized:1b} run function lfm:ui/setup',
                "execute in lfm:runtime run forceload add 0 0",
                "execute in lfm:runtime run kill @e[type=minecraft:marker,tag=lfm.weight_chunk]",
                'execute unless data storage lfm:job state run data merge storage lfm:job {model_sha:"",state:"idle",phase:"",layer:0,tensor:"",row:0,block:0,lane:0,processed_weights:0L,total_weights:0L,progress_ppm:0,elapsed_ticks:0L,input_token:1,result:{token_id:0,piece:"",logit:0.0f,verified:0b}}',
                'execute if data storage lfm:job {state:"running"} run data modify storage lfm:job state set value "recovered"',
                'tellraw @a[tag=lfm.operator] {"text":"[LFM] datapack loaded; use /function lfm:api/start_bos","color":"aqua"}',
            ]
        ),
    )
    _text(
        paths.functions / "tick.mcfunction",
        "\n".join(
            [
                "function lfm:ui/tick",
                'execute if data storage lfm:job {state:"running"} run function lfm:runtime/tick',
                'execute if data storage lfm:job {state:"running"} run function lfm:runtime/classify_batch',
            ]
            + [
                'execute if data storage lfm:job {state:"running",batchable:1b} run function lfm:runtime/tick_fast'
            ]
            * (steps_per_tick - 1)
        ),
    )
    _text(
        paths.functions / "runtime" / "tick.mcfunction",
        "\n".join(
            [
                "data modify storage lfm:job elapsed_ticks set value 0L",
                "execute store result storage lfm:job elapsed_ticks long 1 run time query gametime",
                'execute if data storage lfm:job {phase:"kernel_test"} run function lfm:test/run_kernel',
            ]
        ),
    )
    _api_functions(paths)
    _kernel_test(paths)
    _progress_ui(paths)
    return paths


def _progress_ui(paths: PackPaths) -> None:
    _text(
        paths.functions / "ui" / "setup.mcfunction",
        "\n".join(
            [
                'bossbar add lfm:inference {"text":"LFM inference","color":"light_purple"}',
                "bossbar set lfm:inference color purple",
                "bossbar set lfm:inference style progress",
                "bossbar set lfm:inference max 16",
                "bossbar set lfm:inference value 0",
                "bossbar set lfm:inference players",
                "data modify storage lfm:ui initialized set value 1b",
            ]
        ),
    )
    _text(
        paths.functions / "ui" / "tick.mcfunction",
        "\n".join(
            [
                "bossbar set lfm:inference players",
                'execute if data storage lfm:job {state:"running"} run bossbar set lfm:inference players @a[tag=lfm.operator]',
                'execute if data storage lfm:job {state:"paused"} run bossbar set lfm:inference players @a[tag=lfm.operator]',
                'execute if data storage lfm:job {state:"recovered"} run bossbar set lfm:inference players @a[tag=lfm.operator]',
                'execute unless data storage lfm:job {state:"running"} unless data storage lfm:job {state:"paused"} unless data storage lfm:job {state:"recovered"} run return 0',
                "execute store result score #position lfm.ui run data get storage lfm:cache position",
                "execute store result score #queue lfm.ui run data get storage lfm:prefill queue",
                "execute store result score #remaining lfm.ui run data get storage lfm:generation remaining",
                "scoreboard players operation #total lfm.ui = #position lfm.ui",
                "scoreboard players operation #total lfm.ui += #queue lfm.ui",
                "scoreboard players operation #total lfm.ui += #remaining lfm.ui",
                "execute if score #total lfm.ui matches ..0 run scoreboard players set #total lfm.ui 1",
                "scoreboard players operation #current lfm.ui = #position lfm.ui",
                "execute if score #current lfm.ui matches ..0 run scoreboard players set #current lfm.ui 1",
                "scoreboard players operation #done lfm.ui = #current lfm.ui",
                "scoreboard players remove #done lfm.ui 1",
                "execute if score #done lfm.ui matches ..-1 run scoreboard players set #done lfm.ui 0",
                "execute store result score #layer lfm.ui run data get storage lfm:job layer",
                "scoreboard players add #layer lfm.ui 1",
                "execute if score #layer lfm.ui matches 17.. run scoreboard players set #layer lfm.ui 16",
                "scoreboard players set #layers lfm.ui 16",
                "scoreboard players operation #progress lfm.ui = #done lfm.ui",
                "scoreboard players operation #progress lfm.ui *= #layers lfm.ui",
                "scoreboard players operation #progress lfm.ui += #layer lfm.ui",
                "scoreboard players operation #maximum lfm.ui = #total lfm.ui",
                "scoreboard players operation #maximum lfm.ui *= #layers lfm.ui",
                "execute store result bossbar lfm:inference value run scoreboard players get #progress lfm.ui",
                "execute store result bossbar lfm:inference max run scoreboard players get #maximum lfm.ui",
                'bossbar set lfm:inference name [{"text":"LFM  Token ","color":"light_purple"},{"score":{"name":"#current","objective":"lfm.ui"}},{"text":"/","color":"gray"},{"score":{"name":"#total","objective":"lfm.ui"}},{"text":"  ·  Layer ","color":"aqua"},{"score":{"name":"#layer","objective":"lfm.ui"}},{"text":"/16","color":"gray"}]',
                'execute if data storage lfm:job {phase:"lm_head"} run function lfm:ui/lm_head',
            ]
        ),
    )
    _text(
        paths.functions / "ui" / "lm_head.mcfunction",
        "\n".join(
            [
                "scoreboard players operation #output_token lfm.ui = #current lfm.ui",
                "scoreboard players add #output_token lfm.ui 1",
                "execute if score #output_token lfm.ui > #total lfm.ui run scoreboard players operation #output_token lfm.ui = #total lfm.ui",
                "execute store result score #vocab_row lfm.ui run data get storage lfm:job row",
                "scoreboard players operation #lm_units lfm.ui = #vocab_row lfm.ui",
                "scoreboard players set #vocab_per_unit lfm.ui 4096",
                "scoreboard players operation #lm_units lfm.ui /= #vocab_per_unit lfm.ui",
                "scoreboard players operation #progress lfm.ui += #lm_units lfm.ui",
                "execute store result bossbar lfm:inference value run scoreboard players get #progress lfm.ui",
                'bossbar set lfm:inference name [{"text":"LFM  Output token ","color":"light_purple"},{"score":{"name":"#output_token","objective":"lfm.ui"}},{"text":"/","color":"gray"},{"score":{"name":"#total","objective":"lfm.ui"}},{"text":"  ·  Vocab ","color":"gold"},{"score":{"name":"#vocab_row","objective":"lfm.ui"}},{"text":"/65536","color":"gray"}]',
            ]
        ),
    )


def configure_bos_embedding_runtime(
    paths: PackPaths,
    token: int = 1,
    width: int = 2048,
    total_work: int | None = None,
    next_phase: str | None = None,
    model_sha: str = "",
) -> None:
    if width % 256:
        raise ValueError("embedding width must be Q6_K block aligned")
    block_count = width // 256
    total_work = width if total_work is None else total_work
    if total_work < width:
        raise ValueError("total work cannot be smaller than embedding width")
    for lane in range(256):
        _json(
            paths.float_providers / "kernel" / "q6k_value" / f"{lane:03d}.json",
            q6k_value(lane),
        )

    append_commands: list[str] = []
    for lane in range(256):
        append_commands.extend(
            [
                f"data modify storage lfm:scratch scalar set compute default float lfm:kernel/q6k_value/{lane:03d}",
                "data modify storage lfm:scratch activation append from storage lfm:scratch scalar",
            ]
        )
    _text(paths.functions / "runtime" / "embedding" / "append_block.mcfunction", "\n".join(append_commands))

    for block in range(block_count):
        commands = [
            f"data modify storage lfm:scratch weight set from storage lfm:scratch embedding.blocks[-{block + 1}]",
            "function lfm:runtime/embedding/append_block",
            f"data modify storage lfm:job block set value {block + 1}",
            f"data modify storage lfm:job processed_weights set value {(block + 1) * 256}L",
            f"data modify storage lfm:job progress_ppm set value {((block + 1) * 256 * 1_000_000) // total_work}",
        ]
        if block + 1 == block_count:
            if next_phase is None:
                commands.extend(
                    [
                        'data modify storage lfm:job phase set value "bos_embedding_done"',
                        'data modify storage lfm:job state set value "done"',
                        'data modify storage lfm:job result set value {token_id:1,piece:"<bos-embedding>",logit:0.0f,verified:0b}',
                        'tellraw @a[tag=lfm.operator] [{"text":"[LFM] BOS embedding ready: 2048 real Q6_K values","color":"green"}]',
                    ]
                )
            else:
                commands.extend(
                    [
                        f'data modify storage lfm:job phase set value "{next_phase}"',
                        "data modify storage lfm:job block set value 0",
                        'tellraw @a[tag=lfm.operator] [{"text":"[LFM] BOS embedding ready; entering RMSNorm","color":"aqua"}]',
                    ]
                )
        _text(paths.functions / "runtime" / "embedding" / f"block_{block}.mcfunction", "\n".join(commands))

    dispatch = [
        f"execute if data storage lfm:job {{block:{block}}} run function lfm:runtime/embedding/block_{block}"
        for block in range(block_count)
    ]
    _text(paths.functions / "runtime" / "embedding" / "step.mcfunction", "\n".join(dispatch))
    _text(
        paths.functions / "runtime" / "embedding" / "load.mcfunction",
        "\n".join(
            [
                "execute in lfm:runtime positioned 0 0 0 run place template lfm:runtime/bos_embedding",
                "execute in lfm:runtime positioned 0 0 0 run data modify storage lfm:scratch embedding set from entity @e[type=minecraft:marker,tag=lfm.weight_chunk,limit=1,sort=nearest] data",
                "execute in lfm:runtime run kill @e[type=minecraft:marker,tag=lfm.weight_chunk]",
                "data modify storage lfm:scratch activation set value []",
                "data modify storage lfm:job block set value 0",
                'data modify storage lfm:job phase set value "bos_embedding"',
            ]
        ),
    )
    _text(
        paths.functions / "runtime" / "tick.mcfunction",
        "\n".join(
            [
                "data modify storage lfm:job elapsed_ticks set value 0L",
                "execute store result storage lfm:job elapsed_ticks long 1 run time query gametime",
                'execute if data storage lfm:job {phase:"bos_embedding_load"} run return run function lfm:runtime/embedding/load',
                'execute if data storage lfm:job {phase:"bos_embedding"} run return run function lfm:runtime/embedding/step',
            ]
        ),
    )
    _text(
        paths.functions / "api" / "start_bos.mcfunction",
        "\n".join(
            [
                'execute unless data storage lfm:job {state:"idle"} run return fail',
                "data remove storage lfm:scratch embedding",
                "data remove storage lfm:scratch activation",
                "data remove storage lfm:scratch weight",
                "data remove storage lfm:cache shortconv",
                "data remove storage lfm:cache attention",
                "data modify storage lfm:cache position set value 0",
                f"data modify storage lfm:cache tokens set value [{token}]",
                f'data merge storage lfm:job {{model_sha:"{model_sha}",state:"running",phase:"bos_embedding_load",position:0,input_token:{token},row:0,chunk:0,chunk_row:0,block:0,processed_weights:0L,total_weights:{total_work}L,progress_ppm:0,result:{{token_id:0,piece:"",logit:0.0f,verified:0b}}}}',
                "function lfm:runtime/rope/select with storage lfm:job",
                'tellraw @s {"text":"[LFM] loading real BOS embedding","color":"green"}',
            ]
        ),
    )


def configure_token_embedding_runtime(
    paths: PackPaths,
    vocab_size: int = 65536,
    width: int = 2048,
    rows_per_chunk: int = 256,
    next_phase: str | None = None,
) -> None:
    if width != 2048 or vocab_size % rows_per_chunk or rows_per_chunk * (width // 256) != 2048:
        raise ValueError("token embedding runtime requires 256 rows of width 2048 per structure")
    job_value = lambda path: {
        "type": "minecraft:storage",
        "storage": "lfm:job",
        "path": path,
        "fallback": 0,
    }
    _json(paths.int_providers / "token_embedding" / "chunk.json", {"type": "minecraft:floor_div", "left": job_value("input_token"), "right": rows_per_chunk})
    _json(paths.int_providers / "token_embedding" / "chunk_row.json", {"type": "minecraft:floor_mod", "left": job_value("input_token"), "right": rows_per_chunk})
    _json(
        paths.int_providers / "token_embedding" / "block_index.json",
        {
            "type": "minecraft:sub",
            "left": 2047,
            "right": {
                "type": "minecraft:add",
                "inputs": [
                    {"type": "minecraft:mul", "inputs": [job_value("chunk_row"), 8]},
                    job_value("block"),
                ],
            },
        },
    )
    _json(paths.int_providers / "token_embedding" / "increment_block.json", {"type": "minecraft:add", "inputs": [job_value("block"), 1]})
    _text(
        paths.functions / "runtime" / "token_embedding" / "prepare.mcfunction",
        "\n".join(
            [
                "data modify storage lfm:job chunk set compute default integer lfm:token_embedding/chunk",
                "data modify storage lfm:job chunk_row set compute default integer lfm:token_embedding/chunk_row",
                "data modify storage lfm:job block set value 0",
                "data modify storage lfm:scratch activation set value []",
                'data modify storage lfm:job phase set value "token_embedding_load"',
            ]
        ),
    )
    chunk_count = vocab_size // rows_per_chunk
    for chunk in range(chunk_count):
        _text(
            paths.functions / "runtime" / "token_embedding" / f"load_{chunk:03d}.mcfunction",
            "\n".join(
                [
                    f"execute in lfm:runtime positioned 0 0 0 run place template lfm:runtime/lm_head/{chunk:03d}",
                    "execute in lfm:runtime positioned 0 0 0 run data modify storage lfm:scratch embedding set from entity @e[type=minecraft:marker,tag=lfm.weight_chunk,limit=1,sort=nearest] data",
                    "execute in lfm:runtime run kill @e[type=minecraft:marker,tag=lfm.weight_chunk]",
                    'data modify storage lfm:job phase set value "token_embedding_apply"',
                ]
            ),
        )
    _text(
        paths.functions / "runtime" / "token_embedding" / "load.mcfunction",
        "\n".join(
            f"execute if data storage lfm:job {{chunk:{chunk}}} run return run function lfm:runtime/token_embedding/load_{chunk:03d}"
            for chunk in range(chunk_count)
        ),
    )
    _text(
        paths.functions / "runtime" / "token_embedding" / "append_selected.mcfunction",
        "\n".join(
            [
                "$data modify storage lfm:scratch weight set from storage lfm:scratch embedding.blocks[$(index)]",
                "function lfm:runtime/embedding/append_block",
            ]
        ),
    )
    _text(
        paths.functions / "runtime" / "token_embedding" / "step.mcfunction",
        "\n".join(
            [
                "data modify storage lfm:scratch embedding_args.index set compute default integer lfm:token_embedding/block_index",
                "function lfm:runtime/token_embedding/append_selected with storage lfm:scratch embedding_args",
                "data modify storage lfm:job block set compute default integer lfm:token_embedding/increment_block",
                "execute if data storage lfm:job {block:8} run return run function lfm:runtime/token_embedding/done",
            ]
        ),
    )
    if next_phase is None:
        done_commands = [
            "data modify storage lfm:job result.token_id set from storage lfm:job input_token",
            "function lfm:runtime/tokenizer/lookup with storage lfm:job result",
            'data modify storage lfm:job phase set value "token_embedding_done"',
            'data modify storage lfm:job state set value "done"',
            'tellraw @a[tag=lfm.operator] [{"text":"[LFM] token embedding ready: ","color":"green"},{"nbt":"result.piece","storage":"lfm:job"},{"text":" (id ","color":"gray"},{"nbt":"result.token_id","storage":"lfm:job"},{"text":")","color":"gray"}]',
        ]
    else:
        done_commands = [
            f'data modify storage lfm:job phase set value "{next_phase}"',
            "data modify storage lfm:job layer set value 0",
            "data modify storage lfm:job block set value 0",
            "data modify storage lfm:job row set value 0",
            "data modify storage lfm:job chunk set value 0",
            "data modify storage lfm:job chunk_row set value 0",
        ]
    _text(paths.functions / "runtime" / "token_embedding" / "done.mcfunction", "\n".join(done_commands))
    _text(
        paths.functions / "runtime" / "token_embedding" / "start.mcfunction",
        "\n".join(
            [
                'data merge storage lfm:job {state:"running",phase:"token_embedding_prepare",row:0,chunk:0,chunk_row:0,block:0,result:{token_id:0,piece:"",logit:0.0f,verified:0b}}',
                "data modify storage lfm:job position set from storage lfm:cache position",
                "function lfm:runtime/rope/select with storage lfm:job",
            ]
        ),
    )
    _text(
        paths.functions / "api" / "start_token.mcfunction",
        "\n".join(
            [
                'execute if data storage lfm:job {state:"running"} run return fail',
                "$data modify storage lfm:job input_token set value $(token)",
                "function lfm:runtime/token_embedding/start",
            ]
        ),
    )
    _text(
        paths.functions / "api" / "continue.mcfunction",
        "\n".join(
            [
                'execute unless data storage lfm:job {state:"done"} run return fail',
                "data modify storage lfm:job input_token set from storage lfm:job result.token_id",
                "function lfm:runtime/token_embedding/start",
            ]
        ),
    )
    _append_tick_dispatch(
        paths,
        [
            'execute if data storage lfm:job {phase:"token_embedding_prepare"} run return run function lfm:runtime/token_embedding/prepare',
            'execute if data storage lfm:job {phase:"token_embedding_load"} run return run function lfm:runtime/token_embedding/load',
            'execute if data storage lfm:job {phase:"token_embedding_apply"} run return run function lfm:runtime/token_embedding/step',
        ],
    )


def configure_rms0_runtime(
    paths: PackPaths,
    width: int = 2048,
    epsilon: float = 1e-5,
    work_offset: int = 2048,
    total_work: int = 4096,
    next_phase: str | None = None,
) -> None:
    if width % 256:
        raise ValueError("RMS width must be divisible by 256")
    _json(paths.float_providers / "rms0" / "l2.json", rms_l2_provider(width))
    _json(paths.float_providers / "rms0" / "scale.json", rms_scale_provider(width, epsilon))
    for index in range(width):
        _json(
            paths.float_providers / "rms0" / "apply" / f"{index:04d}.json",
            rms_apply_provider(index),
        )

    _text(
        paths.functions / "runtime" / "rms0" / "load.mcfunction",
        "\n".join(
            [
                "execute in lfm:runtime positioned 0 0 0 run place template lfm:runtime/rms0_weight",
                "execute in lfm:runtime positioned 0 0 0 run data modify storage lfm:scratch rms0.weight set from entity @e[type=minecraft:marker,tag=lfm.weight_chunk,limit=1,sort=nearest] data",
                "execute in lfm:runtime run kill @e[type=minecraft:marker,tag=lfm.weight_chunk]",
                "data modify storage lfm:scratch rms0.l2 set compute default float lfm:rms0/l2",
                "data modify storage lfm:scratch rms0.scale set compute default float lfm:rms0/scale",
                "data modify storage lfm:scratch normalized set value []",
                "data modify storage lfm:job block set value 0",
                'data modify storage lfm:job phase set value "rms0_apply"',
            ]
        ),
    )
    block_count = width // 256
    for block in range(block_count):
        commands: list[str] = []
        for index in range(block * 256, (block + 1) * 256):
            commands.extend(
                [
                    f"data modify storage lfm:scratch scalar set compute default float lfm:rms0/apply/{index:04d}",
                    "data modify storage lfm:scratch normalized append from storage lfm:scratch scalar",
                ]
            )
        processed = work_offset + (block + 1) * 256
        commands.extend(
            [
                f"data modify storage lfm:job block set value {block + 1}",
                f"data modify storage lfm:job processed_weights set value {processed}L",
                f"data modify storage lfm:job progress_ppm set value {(processed * 1_000_000) // total_work}",
            ]
        )
        if block + 1 == block_count:
            if next_phase is None:
                commands.extend(
                    [
                        'data modify storage lfm:job phase set value "rms0_done"',
                        'data modify storage lfm:job state set value "done"',
                        'data modify storage lfm:job result set value {token_id:1,piece:"<layer0-rmsnorm>",logit:0.0f,verified:0b}',
                        'tellraw @a[tag=lfm.operator] [{"text":"[LFM] Layer 0 RMSNorm ready: 2048 values","color":"green"}]',
                    ]
                )
            else:
                commands.extend(
                    [
                        f'data modify storage lfm:job phase set value "{next_phase}"',
                        "data modify storage lfm:job block set value 0",
                        "data modify storage lfm:job row set value 0",
                        "data modify storage lfm:job chunk set value 0",
                        "data modify storage lfm:job chunk_row set value 0",
                        'tellraw @a[tag=lfm.operator] [{"text":"[LFM] Layer 0 RMSNorm ready; entering ShortConv projection","color":"aqua"}]',
                    ]
                )
        _text(paths.functions / "runtime" / "rms0" / f"block_{block}.mcfunction", "\n".join(commands))
    _text(
        paths.functions / "runtime" / "rms0" / "step.mcfunction",
        "\n".join(
            f"execute if data storage lfm:job {{block:{block}}} run function lfm:runtime/rms0/block_{block}"
            for block in range(block_count)
        ),
    )
    tick = paths.functions / "runtime" / "tick.mcfunction"
    lines = tick.read_text(encoding="utf-8").rstrip().splitlines()
    lines.extend(
        [
            'execute if data storage lfm:job {phase:"rms0_load"} run return run function lfm:runtime/rms0/load',
            'execute if data storage lfm:job {phase:"rms0_apply"} run return run function lfm:runtime/rms0/step',
        ]
    )
    _text(tick, "\n".join(lines))


def configure_shortconv0_in_runtime(
    paths: PackPaths,
    input_width: int = 2048,
    rows: int = 6144,
    rows_per_chunk: int = 256,
    work_offset: int = 4096,
    total_work: int | None = None,
    next_phase: str | None = None,
    kind: str = "Q2_K",
) -> None:
    if input_width != 2048 or rows % rows_per_chunk:
        raise ValueError("unsupported ShortConv projection shape")
    chunk_count = rows // rows_per_chunk
    projection_work = work_offset + input_width * rows
    total_work = projection_work if total_work is None else total_work
    for block in range(input_width // 256):
        _json(
            paths.float_providers / "shortconv0_in" / f"block_{block}.json",
            quant_dot_provider(
                kind,
                256,
                activation_path="normalized",
                activation_offset=block * 256,
            ),
        )
    _json(
        paths.int_providers / "shortconv0_in" / "increment_row.json",
        {
            "type": "minecraft:add",
            "inputs": [
                {"type": "minecraft:storage", "storage": "lfm:job", "path": "row", "fallback": 0},
                1,
            ],
        },
    )
    _json(
        paths.int_providers / "shortconv0_in" / "increment_chunk_row.json",
        {
            "type": "minecraft:add",
            "inputs": [
                {"type": "minecraft:storage", "storage": "lfm:job", "path": "chunk_row", "fallback": 0},
                1,
            ],
        },
    )
    _json(
        paths.int_providers / "shortconv0_in" / "increment_chunk.json",
        {
            "type": "minecraft:add",
            "inputs": [
                {"type": "minecraft:storage", "storage": "lfm:job", "path": "chunk", "fallback": 0},
                1,
            ],
        },
    )
    _json(
        paths.int_providers / "shortconv0_in" / "processed.json",
        {
            "type": "minecraft:add",
            "inputs": [
                work_offset,
                {
                    "type": "minecraft:mul",
                    "inputs": [
                        {"type": "minecraft:storage", "storage": "lfm:job", "path": "row", "fallback": 0},
                        input_width,
                    ],
                },
            ],
        },
    )

    for chunk in range(chunk_count):
        _text(
            paths.functions / "runtime" / "shortconv0_in" / f"load_{chunk:02d}.mcfunction",
            "\n".join(
                [
                    f"execute in lfm:runtime positioned 0 0 0 run place template lfm:runtime/shortconv0_in/{chunk:03d}",
                    "execute in lfm:runtime positioned 0 0 0 run data modify storage lfm:scratch matrix set from entity @e[type=minecraft:marker,tag=lfm.weight_chunk,limit=1,sort=nearest] data",
                    "execute in lfm:runtime run kill @e[type=minecraft:marker,tag=lfm.weight_chunk]",
                    "data modify storage lfm:job chunk_row set value 0",
                    'data modify storage lfm:job phase set value "shortconv0_in"',
                ]
            ),
        )
    _text(
        paths.functions / "runtime" / "shortconv0_in" / "load.mcfunction",
        "\n".join(
            ["execute if data storage lfm:job {chunk:0} run data modify storage lfm:scratch shortconv0.in_proj set value []"]
            + [
                f"execute if data storage lfm:job {{chunk:{chunk}}} run return run function lfm:runtime/shortconv0_in/load_{chunk:02d}"
                for chunk in range(chunk_count)
            ]
        ),
    )
    row_commands = ["data modify storage lfm:scratch row_acc set value 0.0f"]
    for block in range(input_width // 256):
        row_commands.extend(
            [
                "data modify storage lfm:scratch weight set from storage lfm:scratch matrix.blocks[-1]",
                "data remove storage lfm:scratch matrix.blocks[-1]",
                f"data modify storage lfm:scratch row_acc set compute default float lfm:shortconv0_in/block_{block}",
            ]
        )
    row_commands.extend(
        [
            "data modify storage lfm:scratch shortconv0.in_proj append from storage lfm:scratch row_acc",
            "data modify storage lfm:job row set compute default integer lfm:shortconv0_in/increment_row",
            "data modify storage lfm:job chunk_row set compute default integer lfm:shortconv0_in/increment_chunk_row",
            "data modify storage lfm:job processed_weights set compute default integer lfm:shortconv0_in/processed",
            f"execute if data storage lfm:job {{row:{rows}}} run return run function lfm:runtime/shortconv0_in/done",
            f"execute if data storage lfm:job {{chunk_row:{rows_per_chunk}}} run data modify storage lfm:job chunk set compute default integer lfm:shortconv0_in/increment_chunk",
            f"execute if data storage lfm:job {{chunk_row:{rows_per_chunk}}} run data modify storage lfm:job phase set value \"shortconv0_in_load\"",
        ]
    )
    _text(paths.functions / "runtime" / "shortconv0_in" / "row.mcfunction", "\n".join(row_commands))
    done_commands = [f"data modify storage lfm:job processed_weights set value {projection_work}"]
    if next_phase is None:
        done_commands.extend(
            [
                "data modify storage lfm:job progress_ppm set value 1000000",
                'data modify storage lfm:job phase set value "shortconv0_in_done"',
                'data modify storage lfm:job state set value "done"',
                'data modify storage lfm:job result set value {token_id:1,piece:"<layer0-shortconv-in>",logit:0.0f,verified:0b}',
                'tellraw @a[tag=lfm.operator] [{"text":"[LFM] Layer 0 ShortConv input projection ready: 6144 values","color":"green"}]',
            ]
        )
    else:
        done_commands.extend(
            [
                f'data modify storage lfm:job phase set value "{next_phase}"',
                "data modify storage lfm:job block set value 0",
                'tellraw @a[tag=lfm.operator] [{"text":"[LFM] ShortConv projection ready; applying causal convolution","color":"aqua"}]',
            ]
        )
    _text(paths.functions / "runtime" / "shortconv0_in" / "done.mcfunction", "\n".join(done_commands))
    tick = paths.functions / "runtime" / "tick.mcfunction"
    lines = tick.read_text(encoding="utf-8").rstrip().splitlines()
    lines.extend(
        [
            'execute if data storage lfm:job {phase:"shortconv0_in_load"} run return run function lfm:runtime/shortconv0_in/load',
            'execute if data storage lfm:job {phase:"shortconv0_in"} run return run function lfm:runtime/shortconv0_in/row',
        ]
    )
    _text(tick, "\n".join(lines))


def configure_shortconv0_mix_runtime(
    paths: PackPaths,
    width: int = 2048,
    work_offset: int = 4096 + 2048 * 6144,
    total_work: int | None = None,
    next_phase: str | None = None,
) -> None:
    if width % 256:
        raise ValueError("ShortConv width must be divisible by 256")
    total_work = work_offset + width if total_work is None else total_work
    for index in range(width):
        _json(paths.float_providers / "shortconv0_mix" / "state" / f"{index:04d}.json", shortconv_state_provider(index, width))
        _json(paths.float_providers / "shortconv0_mix" / "output" / f"{index:04d}.json", shortconv_mix_provider(index, width))
    _text(
        paths.functions / "runtime" / "shortconv0_mix" / "load.mcfunction",
        "\n".join(
            [
                "execute in lfm:runtime positioned 0 0 0 run place template lfm:runtime/shortconv0_conv",
                "execute in lfm:runtime positioned 0 0 0 run data modify storage lfm:scratch shortconv0.conv_weight set from entity @e[type=minecraft:marker,tag=lfm.weight_chunk,limit=1,sort=nearest] data",
                "execute in lfm:runtime run kill @e[type=minecraft:marker,tag=lfm.weight_chunk]",
                "data remove storage lfm:scratch shortconv0.prev2",
                "data remove storage lfm:scratch shortconv0.prev1",
                "data modify storage lfm:scratch shortconv0.prev2 set from storage lfm:cache shortconv.layer0.prev2",
                "data modify storage lfm:scratch shortconv0.prev1 set from storage lfm:cache shortconv.layer0.prev1",
                "data modify storage lfm:scratch shortconv0.conv_state set value []",
                "data modify storage lfm:scratch shortconv0.mixed set value []",
                "data modify storage lfm:job block set value 0",
                'data modify storage lfm:job phase set value "shortconv0_mix"',
            ]
        ),
    )
    block_count = width // 256
    for block in range(block_count):
        commands: list[str] = []
        for index in range(block * 256, (block + 1) * 256):
            commands.extend(
                [
                    f"data modify storage lfm:scratch scalar set compute default float lfm:shortconv0_mix/state/{index:04d}",
                    "data modify storage lfm:scratch shortconv0.conv_state append from storage lfm:scratch scalar",
                    f"data modify storage lfm:scratch scalar set compute default float lfm:shortconv0_mix/output/{index:04d}",
                    "data modify storage lfm:scratch shortconv0.mixed append from storage lfm:scratch scalar",
                ]
            )
        processed = work_offset + (block + 1) * 256
        commands.extend([f"data modify storage lfm:job block set value {block + 1}", f"data modify storage lfm:job processed_weights set value {processed}"])
        if block + 1 == block_count:
            commands.extend(
                [
                    "data remove storage lfm:cache shortconv.layer0.prev2",
                    "data modify storage lfm:cache shortconv.layer0.prev2 set from storage lfm:scratch shortconv0.prev1",
                    "data modify storage lfm:cache shortconv.layer0.prev1 set from storage lfm:scratch shortconv0.conv_state",
                ]
            )
            if next_phase is None:
                commands.extend(["data modify storage lfm:job progress_ppm set value 1000000", 'data modify storage lfm:job phase set value "shortconv0_mix_done"', 'data modify storage lfm:job state set value "done"'])
            else:
                commands.extend([f'data modify storage lfm:job phase set value "{next_phase}"', "data modify storage lfm:job row set value 0", "data modify storage lfm:job chunk set value 0", "data modify storage lfm:job chunk_row set value 0"])
        _text(paths.functions / "runtime" / "shortconv0_mix" / f"block_{block}.mcfunction", "\n".join(commands))
    _text(paths.functions / "runtime" / "shortconv0_mix" / "step.mcfunction", "\n".join(f"execute if data storage lfm:job {{block:{block}}} run function lfm:runtime/shortconv0_mix/block_{block}" for block in range(block_count)))
    tick = paths.functions / "runtime" / "tick.mcfunction"
    lines = tick.read_text(encoding="utf-8").rstrip().splitlines()
    lines.extend(
        [
            'execute if data storage lfm:job {phase:"shortconv0_mix_load"} run return run function lfm:runtime/shortconv0_mix/load',
            'execute if data storage lfm:job {phase:"shortconv0_mix"} run return run function lfm:runtime/shortconv0_mix/step',
        ]
    )
    _text(tick, "\n".join(lines))


def configure_shortconv0_out_runtime(paths: PackPaths, work_offset: int = 4096 + 2048 * 6144 + 2048, next_phase: str | None = None, kind: str = "Q2_K") -> None:
    width, rows, rows_per_chunk = 2048, 2048, 256
    total_work = work_offset + width * rows
    for block in range(8):
        _json(paths.float_providers / "shortconv0_out" / f"block_{block}.json", quant_dot_provider(kind, 256, activation_path="shortconv0.mixed", activation_offset=block * 256))
    for name, path in (("increment_row", "row"), ("increment_chunk_row", "chunk_row"), ("increment_chunk", "chunk")):
        _json(paths.int_providers / "shortconv0_out" / f"{name}.json", {"type": "minecraft:add", "inputs": [{"type": "minecraft:storage", "storage": "lfm:job", "path": path, "fallback": 0}, 1]})
    for chunk in range(8):
        _text(paths.functions / "runtime" / "shortconv0_out" / f"load_{chunk}.mcfunction", "\n".join([f"execute in lfm:runtime positioned 0 0 0 run place template lfm:runtime/shortconv0_out/{chunk:03d}", "execute in lfm:runtime positioned 0 0 0 run data modify storage lfm:scratch matrix set from entity @e[type=minecraft:marker,tag=lfm.weight_chunk,limit=1,sort=nearest] data", "execute in lfm:runtime run kill @e[type=minecraft:marker,tag=lfm.weight_chunk]", "data modify storage lfm:job chunk_row set value 0", 'data modify storage lfm:job phase set value "shortconv0_out"']))
    _text(paths.functions / "runtime" / "shortconv0_out" / "load.mcfunction", "\n".join(["execute if data storage lfm:job {chunk:0} run data modify storage lfm:scratch shortconv0.output set value []"] + [f"execute if data storage lfm:job {{chunk:{chunk}}} run return run function lfm:runtime/shortconv0_out/load_{chunk}" for chunk in range(8)]))
    commands = ["data modify storage lfm:scratch row_acc set value 0.0f"]
    for block in range(8):
        commands.extend(["data modify storage lfm:scratch weight set from storage lfm:scratch matrix.blocks[-1]", "data remove storage lfm:scratch matrix.blocks[-1]", f"data modify storage lfm:scratch row_acc set compute default float lfm:shortconv0_out/block_{block}"])
    commands.extend(["data modify storage lfm:scratch shortconv0.output append from storage lfm:scratch row_acc", "data modify storage lfm:job row set compute default integer lfm:shortconv0_out/increment_row", "data modify storage lfm:job chunk_row set compute default integer lfm:shortconv0_out/increment_chunk_row", f"execute if data storage lfm:job {{row:{rows}}} run return run function lfm:runtime/shortconv0_out/done", f"execute if data storage lfm:job {{chunk_row:{rows_per_chunk}}} run data modify storage lfm:job chunk set compute default integer lfm:shortconv0_out/increment_chunk", f'execute if data storage lfm:job {{chunk_row:{rows_per_chunk}}} run data modify storage lfm:job phase set value "shortconv0_out_load"'])
    _text(paths.functions / "runtime" / "shortconv0_out" / "row.mcfunction", "\n".join(commands))
    done = [f"data modify storage lfm:job processed_weights set value {total_work}"]
    if next_phase is None:
        done.extend(["data modify storage lfm:job progress_ppm set value 1000000", 'data modify storage lfm:job phase set value "shortconv0_out_done"', 'data modify storage lfm:job state set value "done"'])
    else:
        done.extend([f'data modify storage lfm:job phase set value "{next_phase}"', "data modify storage lfm:job block set value 0"])
    _text(paths.functions / "runtime" / "shortconv0_out" / "done.mcfunction", "\n".join(done))
    tick = paths.functions / "runtime" / "tick.mcfunction"
    lines = tick.read_text(encoding="utf-8").rstrip().splitlines()
    lines.extend(['execute if data storage lfm:job {phase:"shortconv0_out_load"} run return run function lfm:runtime/shortconv0_out/load', 'execute if data storage lfm:job {phase:"shortconv0_out"} run return run function lfm:runtime/shortconv0_out/row'])
    _text(tick, "\n".join(lines))


def configure_residual_ffn_norm0_runtime(
    paths: PackPaths,
    width: int = 2048,
    epsilon: float = 1e-5,
    next_phase: str | None = None,
) -> None:
    for index in range(width):
        _json(paths.float_providers / "residual0" / f"{index:04d}.json", residual_provider(index))
        _json(paths.float_providers / "ffn_norm0" / "apply" / f"{index:04d}.json", rms_apply_provider(index, "hidden0", "rms1.scale", "rms1.weight.values"))
    _json(paths.float_providers / "ffn_norm0" / "l2.json", rms_l2_provider(width, "hidden0"))
    _json(paths.float_providers / "ffn_norm0" / "scale.json", rms_scale_provider(width, epsilon, "rms1.l2"))
    for stage, provider_root, output in (("residual0", "residual0", "hidden0"), ("ffn_norm0", "ffn_norm0/apply", "ffn_input0")):
        for block in range(8):
            commands: list[str] = []
            for index in range(block * 256, (block + 1) * 256):
                commands.extend([f"data modify storage lfm:scratch scalar set compute default float lfm:{provider_root}/{index:04d}", f"data modify storage lfm:scratch {output} append from storage lfm:scratch scalar"])
            commands.append(f"data modify storage lfm:job block set value {block + 1}")
            if block == 7:
                if stage == "residual0":
                    commands.extend(['data modify storage lfm:job phase set value "ffn_norm0_load"', "data modify storage lfm:job block set value 0"])
                else:
                    if next_phase is None:
                        commands.extend(["data modify storage lfm:job progress_ppm set value 1000000", 'data modify storage lfm:job phase set value "ffn_norm0_done"', 'data modify storage lfm:job state set value "done"', 'data modify storage lfm:job result set value {token_id:1,piece:"<layer0-ffn-norm>",logit:0.0f,verified:0b}'])
                    else:
                        commands.extend([f'data modify storage lfm:job phase set value "{next_phase}"', "data modify storage lfm:job row set value 0", "data modify storage lfm:job chunk set value 0", "data modify storage lfm:job chunk_row set value 0", "data modify storage lfm:job block set value 0"])
            _text(paths.functions / "runtime" / stage / f"block_{block}.mcfunction", "\n".join(commands))
        _text(paths.functions / "runtime" / stage / "step.mcfunction", "\n".join(f"execute if data storage lfm:job {{block:{block}}} run function lfm:runtime/{stage}/block_{block}" for block in range(8)))
    _text(paths.functions / "runtime" / "residual0" / "load.mcfunction", "\n".join(["data modify storage lfm:scratch hidden0 set value []", "data modify storage lfm:job block set value 0", 'data modify storage lfm:job phase set value "residual0"']))
    _text(paths.functions / "runtime" / "ffn_norm0" / "load.mcfunction", "\n".join(["execute in lfm:runtime positioned 0 0 0 run place template lfm:runtime/ffn_norm0_weight", "execute in lfm:runtime positioned 0 0 0 run data modify storage lfm:scratch rms1.weight set from entity @e[type=minecraft:marker,tag=lfm.weight_chunk,limit=1,sort=nearest] data", "execute in lfm:runtime run kill @e[type=minecraft:marker,tag=lfm.weight_chunk]", "data modify storage lfm:scratch rms1.l2 set compute default float lfm:ffn_norm0/l2", "data modify storage lfm:scratch rms1.scale set compute default float lfm:ffn_norm0/scale", "data modify storage lfm:scratch ffn_input0 set value []", "data modify storage lfm:job block set value 0", 'data modify storage lfm:job phase set value "ffn_norm0"']))
    tick = paths.functions / "runtime" / "tick.mcfunction"
    lines = tick.read_text(encoding="utf-8").rstrip().splitlines()
    lines.extend(['execute if data storage lfm:job {phase:"residual0_load"} run return run function lfm:runtime/residual0/load', 'execute if data storage lfm:job {phase:"residual0"} run return run function lfm:runtime/residual0/step', 'execute if data storage lfm:job {phase:"ffn_norm0_load"} run return run function lfm:runtime/ffn_norm0/load', 'execute if data storage lfm:job {phase:"ffn_norm0"} run return run function lfm:runtime/ffn_norm0/step'])
    _text(tick, "\n".join(lines))


def configure_quant_matvec_runtime(
    paths: PackPaths,
    name: str,
    kind: str,
    input_path: str,
    output_path: str,
    input_width: int,
    rows: int,
    next_phase: str,
    provider_name: str | None = None,
    structure_name: str | None = None,
    write_float_providers: bool = True,
    write_output: bool = True,
    online_argmax: bool = False,
) -> None:
    provider_name = name if provider_name is None else provider_name
    structure_name = name if structure_name is None else structure_name
    blocks_per_row = input_width // 256
    if input_width % 256 or 2048 % blocks_per_row or rows % (2048 // blocks_per_row):
        raise ValueError(f"unsupported matrix paging: {name}")
    rows_per_chunk = 2048 // blocks_per_row
    chunk_count = rows // rows_per_chunk
    if write_float_providers:
        for block in range(blocks_per_row):
            _json(paths.float_providers / provider_name / f"block_{block}.json", quant_dot_provider(kind, 256, activation_path=input_path, activation_offset=block * 256))
    for counter_provider, field in (("increment_row", "row"), ("increment_chunk_row", "chunk_row"), ("increment_chunk", "chunk")):
        _json(paths.int_providers / name / f"{counter_provider}.json", {"type": "minecraft:add", "inputs": [{"type": "minecraft:storage", "storage": "lfm:job", "path": field, "fallback": 0}, 1]})
    for chunk in range(chunk_count):
        _text(paths.functions / "runtime" / name / f"load_{chunk:02d}.mcfunction", "\n".join([f"execute in lfm:runtime positioned 0 0 0 run place template lfm:runtime/{structure_name}/{chunk:03d}", "execute in lfm:runtime positioned 0 0 0 run data modify storage lfm:scratch matrix set from entity @e[type=minecraft:marker,tag=lfm.weight_chunk,limit=1,sort=nearest] data", "execute in lfm:runtime run kill @e[type=minecraft:marker,tag=lfm.weight_chunk]", "data modify storage lfm:job chunk_row set value 0", f'data modify storage lfm:job phase set value "{name}"']))
    load_commands: list[str] = []
    if write_output:
        load_commands.append(f"execute if data storage lfm:job {{chunk:0}} run data modify storage lfm:scratch {output_path} set value []")
    if online_argmax:
        load_commands.extend(
            [
                "execute if data storage lfm:job {chunk:0} run scoreboard players set #best lfm.argmax -2147483648",
                "execute if data storage lfm:job {chunk:0} run scoreboard players set #current lfm.argmax -2147483648",
                "execute if data storage lfm:job {chunk:0} run data modify storage lfm:job result.token_id set value 0",
                "execute if data storage lfm:job {chunk:0} run data modify storage lfm:job result.logit set value -1000000.0f",
            ]
        )
    load_commands.extend(f"execute if data storage lfm:job {{chunk:{chunk}}} run return run function lfm:runtime/{name}/load_{chunk:02d}" for chunk in range(chunk_count))
    _text(paths.functions / "runtime" / name / "load.mcfunction", "\n".join(load_commands))
    commands = ["data modify storage lfm:scratch row_acc set value 0.0f"]
    for block in range(blocks_per_row):
        commands.extend(["data modify storage lfm:scratch weight set from storage lfm:scratch matrix.blocks[-1]", "data remove storage lfm:scratch matrix.blocks[-1]", f"data modify storage lfm:scratch row_acc set compute default float lfm:{provider_name}/block_{block}"])
    if write_output:
        commands.append(f"data modify storage lfm:scratch {output_path} append from storage lfm:scratch row_acc")
    if online_argmax:
        commands.extend(
            [
                "execute store result score #current lfm.argmax run data get storage lfm:scratch row_acc 10000000",
                "execute if score #current lfm.argmax > #best lfm.argmax run data modify storage lfm:job result.logit set from storage lfm:scratch row_acc",
                "execute if score #current lfm.argmax > #best lfm.argmax run data modify storage lfm:job result.token_id set from storage lfm:job row",
                "execute if score #current lfm.argmax > #best lfm.argmax run scoreboard players operation #best lfm.argmax = #current lfm.argmax",
            ]
        )
    commands.extend([f"data modify storage lfm:job row set compute default integer lfm:{name}/increment_row", f"data modify storage lfm:job chunk_row set compute default integer lfm:{name}/increment_chunk_row", f"execute if data storage lfm:job {{row:{rows}}} run return run function lfm:runtime/{name}/done", f"execute if data storage lfm:job {{chunk_row:{rows_per_chunk}}} run data modify storage lfm:job chunk set compute default integer lfm:{name}/increment_chunk", f'execute if data storage lfm:job {{chunk_row:{rows_per_chunk}}} run data modify storage lfm:job phase set value "{name}_load"'])
    _text(paths.functions / "runtime" / name / "row.mcfunction", "\n".join(commands))
    _text(paths.functions / "runtime" / name / "done.mcfunction", "\n".join([f'data modify storage lfm:job phase set value "{next_phase}"', "data modify storage lfm:job row set value 0", "data modify storage lfm:job chunk set value 0", "data modify storage lfm:job chunk_row set value 0", "data modify storage lfm:job block set value 0"]))
    tick = paths.functions / "runtime" / "tick.mcfunction"
    lines = tick.read_text(encoding="utf-8").rstrip().splitlines()
    lines.extend([f'execute if data storage lfm:job {{phase:"{name}_load"}} run return run function lfm:runtime/{name}/load', f'execute if data storage lfm:job {{phase:"{name}"}} run return run function lfm:runtime/{name}/row'])
    _text(tick, "\n".join(lines))


def configure_final_norm_runtime(paths: PackPaths) -> None:
    configure_reused_rms_runtime(paths, "final_norm", "final_norm", "lm_head_load")


def configure_rope_table_runtime(paths: PackPaths) -> None:
    _text(
        paths.functions / "runtime" / "rope" / "load.mcfunction",
        "\n".join(
            [
                "execute in lfm:runtime positioned 0 0 0 run place template lfm:runtime/rope",
                "execute in lfm:runtime positioned 0 0 0 run data modify storage lfm:rope positions set from entity @e[type=minecraft:marker,tag=lfm.weight_chunk,limit=1,sort=nearest] data.positions",
                "execute in lfm:runtime run kill @e[type=minecraft:marker,tag=lfm.weight_chunk]",
            ]
        ),
    )
    _text(
        paths.functions / "runtime" / "rope" / "select.mcfunction",
        "$data modify storage lfm:scratch attn.rope set from storage lfm:rope positions[$(position)]",
    )
    load_function = paths.functions / "load.mcfunction"
    load_lines = load_function.read_text(encoding="utf-8").rstrip().splitlines()
    rope_load = "function lfm:runtime/rope/load"
    if rope_load not in load_lines:
        load_lines.append(rope_load)
        _text(load_function, "\n".join(load_lines))


def configure_tokenizer_runtime(paths: PackPaths) -> None:
    _text(
        paths.functions / "runtime" / "tokenizer" / "load.mcfunction",
        "\n".join(
            [
                "execute in lfm:runtime positioned 0 0 0 run place template lfm:runtime/tokenizer",
                "execute in lfm:runtime positioned 0 0 0 run data modify storage lfm:tokenizer tokens set from entity @e[type=minecraft:marker,tag=lfm.weight_chunk,limit=1,sort=nearest] data.values",
                "execute in lfm:runtime run kill @e[type=minecraft:marker,tag=lfm.weight_chunk]",
            ]
        ),
    )
    _text(
        paths.functions / "runtime" / "tokenizer" / "lookup.mcfunction",
        "$data modify storage lfm:job result.piece set from storage lfm:tokenizer tokens[$(token_id)]",
    )
    load_function = paths.functions / "load.mcfunction"
    load_lines = load_function.read_text(encoding="utf-8").rstrip().splitlines()
    tokenizer_load = "function lfm:runtime/tokenizer/load"
    if tokenizer_load not in load_lines:
        load_lines.append(tokenizer_load)
        _text(load_function, "\n".join(load_lines))


def configure_minecraft_input_runtime(
    paths: PackPaths,
    prefix_tokens: list[int],
    suffix_tokens: list[int],
) -> None:
    prefix = "[" + ",".join(map(str, prefix_tokens)) + "]"
    suffix_commands = [
        f"data modify storage lfm:input tokens append value {token_id}"
        for token_id in suffix_tokens
    ]
    _text(
        paths.functions / "runtime" / "input" / "load.mcfunction",
        "\n".join(
            [
                "execute in lfm:runtime positioned 0 0 0 run place template lfm:runtime/input_chars",
                "execute in lfm:runtime positioned 0 0 0 run data modify storage lfm:input_lookup chars set from entity @e[type=minecraft:marker,tag=lfm.weight_chunk,limit=1,sort=nearest] data.values",
                "execute in lfm:runtime run kill @e[type=minecraft:marker,tag=lfm.weight_chunk]",
            ]
        ),
    )
    _text(
        paths.functions / "api" / "chat.mcfunction",
        "\n".join(
            [
                'execute unless data storage lfm:job {state:"idle"} unless data storage lfm:job {state:"done"} unless data storage lfm:job {state:"cancelled"} run return run function lfm:runtime/input/busy',
                '$function lfm:runtime/input/start {message:"$(message)",max_tokens:$(max_tokens)}',
            ]
        ),
    )
    _text(paths.functions / "api" / "input_status.mcfunction", "data get storage lfm:input")
    _text(
        paths.functions / "runtime" / "input" / "start.mcfunction",
        "\n".join(
            [
                "function lfm:api/reset",
                '$data modify storage lfm:input message set value "$(message)"',
                '$data modify storage lfm:input remaining set value "$(message)"',
                "$data modify storage lfm:input max_tokens set value $(max_tokens)",
                f"data modify storage lfm:input tokens set value {prefix}",
                'data modify storage lfm:input state set value "tokenizing"',
                "execute store result score #input_chars lfm.argmax run data get storage lfm:input remaining",
                "execute unless score #input_chars lfm.argmax matches 1..64 run return run function lfm:runtime/input/invalid_length",
                "function lfm:runtime/input/next",
            ]
        ),
    )
    _text(
        paths.functions / "runtime" / "input" / "next.mcfunction",
        "\n".join(
            [
                'execute if data storage lfm:input {remaining:""} run return run function lfm:runtime/input/finish',
                "data modify storage lfm:input char set string storage lfm:input remaining 0 1",
                "data modify storage lfm:input remaining set string storage lfm:input remaining 1",
                "data remove storage lfm:input current",
                "execute if data storage lfm:input {char:'\"'} run return run function lfm:runtime/input/unsupported",
                "execute if data storage lfm:input {char:'\\\\'} run return run function lfm:runtime/input/unsupported",
                "function lfm:runtime/input/lookup with storage lfm:input",
            ]
        ),
    )
    _text(
        paths.functions / "runtime" / "input" / "lookup.mcfunction",
        "\n".join(
            [
                '$data modify storage lfm:input current set from storage lfm:input_lookup chars."$(char)"',
                "execute unless data storage lfm:input current.n run return run function lfm:runtime/input/unsupported",
                "data modify storage lfm:input tokens append from storage lfm:input current.b0",
                "execute if data storage lfm:input {current:{n:2}} run data modify storage lfm:input tokens append from storage lfm:input current.b1",
                "execute if data storage lfm:input {current:{n:3}} run data modify storage lfm:input tokens append from storage lfm:input current.b1",
                "execute if data storage lfm:input {current:{n:3}} run data modify storage lfm:input tokens append from storage lfm:input current.b2",
                "function lfm:runtime/input/next",
            ]
        ),
    )
    _text(
        paths.functions / "runtime" / "input" / "finish.mcfunction",
        "\n".join(
            suffix_commands
            + [
                "execute store result score #input_tokens lfm.argmax run data get storage lfm:input tokens",
                "execute store result score #generation_tokens lfm.argmax run data get storage lfm:input max_tokens",
                "scoreboard players operation #input_total lfm.argmax = #input_tokens lfm.argmax",
                "scoreboard players operation #input_total lfm.argmax += #generation_tokens lfm.argmax",
                "execute unless score #generation_tokens lfm.argmax matches 1..64 run return run function lfm:runtime/input/invalid_generation_length",
                "execute unless score #input_total lfm.argmax matches ..256 run return run function lfm:runtime/input/too_long",
                "function lfm:runtime/input/launch",
            ]
        ),
    )
    _text(
        paths.functions / "runtime" / "input" / "launch.mcfunction",
        "\n".join(
            [
                "data modify storage lfm:prefill queue set from storage lfm:input tokens",
                "data modify storage lfm:generation remaining set from storage lfm:input max_tokens",
                'data modify storage lfm:prefill active set value 1b',
                'data modify storage lfm:generation active set value 1b',
                "data modify storage lfm:generation token_ids set value []",
                "data modify storage lfm:generation pieces set value []",
                'data modify storage lfm:input state set value "running"',
                'tellraw @s [{"text":"[LFM] Minecraft tokenizer accepted: ","color":"green"},{"nbt":"message","storage":"lfm:input"},{"text":" (byte tokens: ","color":"gray"},{"score":{"name":"#input_tokens","objective":"lfm.argmax"}},{"text":")"}]',
                "function lfm:api/start_bos",
            ]
        ),
    )
    errors = {
        "busy": "An inference job is already active; pause/reset it before starting a new chat.",
        "invalid_length": "Message length must be between 1 and 64 characters.",
        "invalid_generation_length": "max_tokens must be between 1 and 64.",
        "too_long": "The byte-token prompt plus output exceeds the 256-token context limit.",
        "unsupported": "This character is not supported by the in-Minecraft tokenizer.",
    }
    for name, message in errors.items():
        _text(
            paths.functions / "runtime" / "input" / f"{name}.mcfunction",
            "\n".join(
                [
                    f'data modify storage lfm:input state set value "error:{name}"',
                    f'tellraw @s {{"text":"[LFM] {message}","color":"red"}}',
                    "return fail",
                ]
            ),
        )
    load_function = paths.functions / "load.mcfunction"
    load_lines = load_function.read_text(encoding="utf-8").rstrip().splitlines()
    input_load = "function lfm:runtime/input/load"
    if input_load not in load_lines:
        load_lines.append(input_load)
        _text(load_function, "\n".join(load_lines))


def configure_lm_head_runtime(paths: PackPaths, rows: int = 65536) -> None:
    load_function = paths.functions / "load.mcfunction"
    load_lines = load_function.read_text(encoding="utf-8").rstrip().splitlines()
    if "scoreboard objectives add lfm.argmax dummy" not in load_lines:
        load_lines.insert(0, "scoreboard objectives add lfm.argmax dummy")
        _text(load_function, "\n".join(load_lines))
    _json(
        paths.int_providers / "cache" / "increment_position.json",
        {
            "type": "minecraft:add",
            "inputs": [
                {"type": "minecraft:storage", "storage": "lfm:cache", "path": "position", "fallback": 0},
                1,
            ],
        },
    )
    configure_quant_matvec_runtime(
        paths,
        "lm_head",
        "Q6_K",
        "normalized",
        "logits",
        2048,
        rows,
        "lm_head_done",
        write_output=False,
        online_argmax=True,
    )
    _text(
        paths.functions / "runtime" / "lm_head" / "done.mcfunction",
        "\n".join(
            [
                "data modify storage lfm:job progress_ppm set value 1000000",
                'data modify storage lfm:job phase set value "lm_head_done"',
                'data modify storage lfm:job state set value "done"',
                "function lfm:runtime/tokenizer/lookup with storage lfm:job result",
                "data modify storage lfm:cache tokens append from storage lfm:job result.token_id",
                "data modify storage lfm:cache position set compute default integer lfm:cache/increment_position",
                'tellraw @a[tag=lfm.operator] [{"text":"[LFM] next token: ","color":"green"},{"nbt":"result.piece","storage":"lfm:job"},{"text":"  id: ","color":"gray"},{"nbt":"result.token_id","storage":"lfm:job"},{"text":"  logit: ","color":"gray"},{"nbt":"result.logit","storage":"lfm:job"}]',
                "function lfm:runtime/generation/on_token",
            ]
        ),
    )


def configure_generation_runtime(paths: PackPaths) -> None:
    _json(
        paths.int_providers / "generation" / "decrement_remaining.json",
        {
            "type": "minecraft:sub",
            "left": {
                "type": "minecraft:storage",
                "storage": "lfm:generation",
                "path": "remaining",
                "fallback": 0,
            },
            "right": 1,
        },
    )
    _text(
        paths.functions / "runtime" / "generation" / "on_token.mcfunction",
        "\n".join(
            [
                'execute unless data storage lfm:generation {active:1b} run return 0',
                "data modify storage lfm:generation token_ids append from storage lfm:job result.token_id",
                "data modify storage lfm:generation pieces append from storage lfm:job result.piece",
                "data modify storage lfm:generation remaining set compute default integer lfm:generation/decrement_remaining",
                'execute if data storage lfm:job {result:{token_id:7}} run data modify storage lfm:generation active set value 0b',
                'execute if data storage lfm:generation {remaining:0} run data modify storage lfm:generation active set value 0b',
                'execute if data storage lfm:generation {active:1b} run schedule function lfm:runtime/generation/continue 1t replace',
                'execute unless data storage lfm:generation {active:1b} run tellraw @a[tag=lfm.operator] [{"text":"[LFM] generation complete: ","color":"aqua"},{"nbt":"token_ids","storage":"lfm:generation"}]',
            ]
        ),
    )
    _text(
        paths.functions / "runtime" / "generation" / "continue.mcfunction",
        "\n".join(
            [
                'execute unless data storage lfm:generation {active:1b} run return 0',
                "function lfm:api/continue",
            ]
        ),
    )
    _text(
        paths.functions / "api" / "start_generate_bos.mcfunction",
        "\n".join(
            [
                "function lfm:api/reset",
                "$data modify storage lfm:generation remaining set value $(max_tokens)",
                "execute store result score #generation_remaining lfm.argmax run data get storage lfm:generation remaining",
                "execute unless score #generation_remaining lfm.argmax matches 1..256 run return fail",
                'data modify storage lfm:generation active set value 1b',
                "data modify storage lfm:generation token_ids set value []",
                "data modify storage lfm:generation pieces set value []",
                "function lfm:api/start_bos",
            ]
        ),
    )
    _text(
        paths.functions / "api" / "generate_more.mcfunction",
        "\n".join(
            [
                'execute unless data storage lfm:job {state:"done"} run return fail',
                "$data modify storage lfm:generation remaining set value $(max_tokens)",
                "execute store result score #generation_remaining lfm.argmax run data get storage lfm:generation remaining",
                "execute store result score #generation_position lfm.argmax run data get storage lfm:cache position",
                "scoreboard players operation #generation_position lfm.argmax += #generation_remaining lfm.argmax",
                "execute unless score #generation_remaining lfm.argmax matches 1..256 run return fail",
                "execute unless score #generation_position lfm.argmax matches ..256 run return fail",
                'data modify storage lfm:generation active set value 1b',
                "data modify storage lfm:generation token_ids set value []",
                "data modify storage lfm:generation pieces set value []",
                "function lfm:api/continue",
            ]
        ),
    )


def configure_prefill_runtime(paths: PackPaths) -> None:
    _text(
        paths.functions / "runtime" / "prefill" / "token_complete.mcfunction",
        "\n".join(
            [
                'execute if data storage lfm:prefill {active:1b} if data storage lfm:prefill queue[0] run return run function lfm:runtime/prefill/next',
                'data modify storage lfm:prefill active set value 0b',
                'data modify storage lfm:job phase set value "final_norm_load"',
                "data modify storage lfm:job row set value 0",
                "data modify storage lfm:job chunk set value 0",
                "data modify storage lfm:job chunk_row set value 0",
                "data modify storage lfm:job block set value 0",
            ]
        ),
    )
    _text(
        paths.functions / "runtime" / "prefill" / "next.mcfunction",
        "\n".join(
            [
                "data modify storage lfm:cache position set compute default integer lfm:cache/increment_position",
                "data modify storage lfm:job input_token set from storage lfm:prefill queue[0]",
                "data modify storage lfm:cache tokens append from storage lfm:job input_token",
                "data remove storage lfm:prefill queue[0]",
                "function lfm:runtime/token_embedding/start",
            ]
        ),
    )
    _text(
        paths.functions / "api" / "start_prompt.mcfunction",
        "\n".join(
            [
                "function lfm:api/reset",
                "$data modify storage lfm:prefill queue set value $(tokens)",
                "$data modify storage lfm:generation remaining set value $(max_tokens)",
                "execute store result score #prefill_length lfm.argmax run data get storage lfm:prefill queue",
                "execute store result score #generation_remaining lfm.argmax run data get storage lfm:generation remaining",
                "scoreboard players operation #prefill_length lfm.argmax += #generation_remaining lfm.argmax",
                "execute unless score #generation_remaining lfm.argmax matches 1..256 run return fail",
                "execute unless score #prefill_length lfm.argmax matches ..256 run return fail",
                'data modify storage lfm:prefill active set value 1b',
                'data modify storage lfm:generation active set value 1b',
                "data modify storage lfm:generation token_ids set value []",
                "data modify storage lfm:generation pieces set value []",
                "function lfm:api/start_bos",
            ]
        ),
    )
    _text(
        paths.functions / "api" / "append_prompt.mcfunction",
        "\n".join(
            [
                'execute unless data storage lfm:job {state:"done"} run return fail',
                "$data modify storage lfm:prefill queue set value $(tokens)",
                "$data modify storage lfm:generation remaining set value $(max_tokens)",
                "execute store result score #prefill_length lfm.argmax run data get storage lfm:prefill queue",
                "execute store result score #generation_remaining lfm.argmax run data get storage lfm:generation remaining",
                "execute store result score #prefill_position lfm.argmax run data get storage lfm:cache position",
                "scoreboard players operation #prefill_length lfm.argmax += #generation_remaining lfm.argmax",
                "scoreboard players operation #prefill_length lfm.argmax += #prefill_position lfm.argmax",
                "execute unless score #generation_remaining lfm.argmax matches 1..256 run return fail",
                "execute unless score #prefill_length lfm.argmax matches ..256 run return fail",
                'data modify storage lfm:prefill active set value 1b',
                'data modify storage lfm:generation active set value 1b',
                "data modify storage lfm:generation token_ids set value []",
                "data modify storage lfm:generation pieces set value []",
                "function lfm:api/continue",
            ]
        ),
    )
    _text(
        paths.functions / "api" / "prefill_status.mcfunction",
        "data get storage lfm:prefill",
    )
    _append_tick_dispatch(
        paths,
        ['execute if data storage lfm:job {phase:"prefill_token_complete"} run return run function lfm:runtime/prefill/token_complete'],
    )
    _text(
        paths.functions / "api" / "generation_status.mcfunction",
        "data get storage lfm:generation",
    )


def configure_ffn0_runtime(
    paths: PackPaths,
    next_phase: str | None = None,
    gate_kind: str = "Q2_K",
    up_kind: str = "Q2_K",
    down_kind: str = "Q3_K",
) -> None:
    configure_quant_matvec_runtime(paths, "ffn_gate0", gate_kind, "ffn_input0", "ffn0.gate", 2048, 8192, "ffn_up0_load")
    configure_quant_matvec_runtime(paths, "ffn_up0", up_kind, "ffn_input0", "ffn0.up", 2048, 8192, "swiglu0")
    for index in range(8192):
        _json(paths.float_providers / "swiglu0" / f"{index:04d}.json", swiglu_provider(index))
    for block in range(32):
        commands: list[str] = []
        for index in range(block * 256, (block + 1) * 256):
            commands.extend([f"data modify storage lfm:scratch scalar set compute default float lfm:swiglu0/{index:04d}", "data modify storage lfm:scratch ffn0.activated append from storage lfm:scratch scalar"])
        commands.append(f"data modify storage lfm:job block set value {block + 1}")
        if block == 31:
            commands.extend(['data modify storage lfm:job phase set value "ffn_down0_load"', "data modify storage lfm:job row set value 0", "data modify storage lfm:job chunk set value 0", "data modify storage lfm:job chunk_row set value 0"])
        _text(paths.functions / "runtime" / "swiglu0" / f"block_{block}.mcfunction", "\n".join(commands))
    _text(paths.functions / "runtime" / "swiglu0" / "load.mcfunction", "\n".join(["data modify storage lfm:scratch ffn0.activated set value []", "data modify storage lfm:job block set value 0", 'data modify storage lfm:job phase set value "swiglu0_apply"']))
    _text(paths.functions / "runtime" / "swiglu0" / "step.mcfunction", "\n".join(f"execute if data storage lfm:job {{block:{block}}} run function lfm:runtime/swiglu0/block_{block}" for block in range(32)))
    tick = paths.functions / "runtime" / "tick.mcfunction"
    lines = tick.read_text(encoding="utf-8").rstrip().splitlines()
    lines.extend(['execute if data storage lfm:job {phase:"swiglu0"} run return run function lfm:runtime/swiglu0/load', 'execute if data storage lfm:job {phase:"swiglu0_apply"} run return run function lfm:runtime/swiglu0/step'])
    _text(tick, "\n".join(lines))
    configure_quant_matvec_runtime(paths, "ffn_down0", down_kind, "ffn0.activated", "ffn0.down", 8192, 2048, "ffn_residual0")
    for index in range(2048):
        _json(paths.float_providers / "ffn_residual0" / f"{index:04d}.json", add_paths_provider(index, "hidden0", "ffn0.down"))
    for block in range(8):
        commands = []
        for index in range(block * 256, (block + 1) * 256):
            commands.extend([f"data modify storage lfm:scratch scalar set compute default float lfm:ffn_residual0/{index:04d}", "data modify storage lfm:scratch layer0_output append from storage lfm:scratch scalar"])
        commands.append(f"data modify storage lfm:job block set value {block + 1}")
        if block == 7:
            if next_phase is None:
                commands.extend(["data modify storage lfm:job progress_ppm set value 1000000", 'data modify storage lfm:job phase set value "layer0_done"', 'data modify storage lfm:job state set value "done"', 'data modify storage lfm:job result set value {token_id:1,piece:"<layer0-output>",logit:0.0f,verified:0b}'])
            else:
                commands.extend(["data modify storage lfm:scratch activation set from storage lfm:scratch layer0_output", "data modify storage lfm:job layer set value 1", f'data modify storage lfm:job phase set value "{next_phase}"', "data modify storage lfm:job block set value 0"])
        _text(paths.functions / "runtime" / "ffn_residual0" / f"block_{block}.mcfunction", "\n".join(commands))
    _text(paths.functions / "runtime" / "ffn_residual0" / "load.mcfunction", "\n".join(["data modify storage lfm:scratch layer0_output set value []", "data modify storage lfm:job block set value 0", 'data modify storage lfm:job phase set value "ffn_residual0_apply"']))
    _text(paths.functions / "runtime" / "ffn_residual0" / "step.mcfunction", "\n".join(f"execute if data storage lfm:job {{block:{block}}} run function lfm:runtime/ffn_residual0/block_{block}" for block in range(8)))
    tick = paths.functions / "runtime" / "tick.mcfunction"
    lines = tick.read_text(encoding="utf-8").rstrip().splitlines()
    lines.extend(['execute if data storage lfm:job {phase:"ffn_residual0"} run return run function lfm:runtime/ffn_residual0/load', 'execute if data storage lfm:job {phase:"ffn_residual0_apply"} run return run function lfm:runtime/ffn_residual0/step'])
    _text(tick, "\n".join(lines))


def configure_reused_rms_runtime(
    paths: PackPaths,
    name: str,
    weight_structure: str,
    next_phase: str,
    width: int = 2048,
) -> None:
    if width != 2048:
        raise ValueError("the reusable RMS runtime currently requires width 2048")
    _text(
        paths.functions / "runtime" / name / "load.mcfunction",
        "\n".join(
            [
                f"execute in lfm:runtime positioned 0 0 0 run place template lfm:runtime/{weight_structure}",
                "execute in lfm:runtime positioned 0 0 0 run data modify storage lfm:scratch rms0.weight set from entity @e[type=minecraft:marker,tag=lfm.weight_chunk,limit=1,sort=nearest] data",
                "execute in lfm:runtime run kill @e[type=minecraft:marker,tag=lfm.weight_chunk]",
                "data modify storage lfm:scratch rms0.l2 set compute default float lfm:rms0/l2",
                "data modify storage lfm:scratch rms0.scale set compute default float lfm:rms0/scale",
                "data modify storage lfm:scratch normalized set value []",
                "data modify storage lfm:job block set value 0",
                f'data modify storage lfm:job phase set value "{name}_apply"',
            ]
        ),
    )
    for block in range(8):
        commands: list[str] = []
        for index in range(block * 256, (block + 1) * 256):
            commands.extend(
                [
                    f"data modify storage lfm:scratch scalar set compute default float lfm:rms0/apply/{index:04d}",
                    "data modify storage lfm:scratch normalized append from storage lfm:scratch scalar",
                ]
            )
        commands.append(f"data modify storage lfm:job block set value {block + 1}")
        if block == 7:
            commands.extend([f'data modify storage lfm:job phase set value "{next_phase}"', "data modify storage lfm:job block set value 0", "data modify storage lfm:job row set value 0", "data modify storage lfm:job chunk set value 0", "data modify storage lfm:job chunk_row set value 0"])
        _text(paths.functions / "runtime" / name / f"block_{block}.mcfunction", "\n".join(commands))
    _text(paths.functions / "runtime" / name / "step.mcfunction", "\n".join(f"execute if data storage lfm:job {{block:{block}}} run function lfm:runtime/{name}/block_{block}" for block in range(8)))
    _append_tick_dispatch(paths, [f'execute if data storage lfm:job {{phase:"{name}_load"}} run return run function lfm:runtime/{name}/load', f'execute if data storage lfm:job {{phase:"{name}_apply"}} run return run function lfm:runtime/{name}/step'])


def configure_reused_shortconv_mix_runtime(
    paths: PackPaths,
    name: str,
    conv_structure: str,
    next_phase: str,
    cache_layer: int,
) -> None:
    _text(
        paths.functions / "runtime" / name / "load.mcfunction",
        "\n".join(
            [
                f"execute in lfm:runtime positioned 0 0 0 run place template lfm:runtime/{conv_structure}",
                "execute in lfm:runtime positioned 0 0 0 run data modify storage lfm:scratch shortconv0.conv_weight set from entity @e[type=minecraft:marker,tag=lfm.weight_chunk,limit=1,sort=nearest] data",
                "execute in lfm:runtime run kill @e[type=minecraft:marker,tag=lfm.weight_chunk]",
                "data remove storage lfm:scratch shortconv0.prev2",
                "data remove storage lfm:scratch shortconv0.prev1",
                f"data modify storage lfm:scratch shortconv0.prev2 set from storage lfm:cache shortconv.layer{cache_layer}.prev2",
                f"data modify storage lfm:scratch shortconv0.prev1 set from storage lfm:cache shortconv.layer{cache_layer}.prev1",
                "data modify storage lfm:scratch shortconv0.conv_state set value []",
                "data modify storage lfm:scratch shortconv0.mixed set value []",
                "data modify storage lfm:job block set value 0",
                f'data modify storage lfm:job phase set value "{name}_apply"',
            ]
        ),
    )
    for block in range(8):
        commands: list[str] = []
        for index in range(block * 256, (block + 1) * 256):
            commands.extend(
                [
                    f"data modify storage lfm:scratch scalar set compute default float lfm:shortconv0_mix/state/{index:04d}",
                    "data modify storage lfm:scratch shortconv0.conv_state append from storage lfm:scratch scalar",
                    f"data modify storage lfm:scratch scalar set compute default float lfm:shortconv0_mix/output/{index:04d}",
                    "data modify storage lfm:scratch shortconv0.mixed append from storage lfm:scratch scalar",
                ]
            )
        commands.append(f"data modify storage lfm:job block set value {block + 1}")
        if block == 7:
            commands.extend(
                [
                    f"data remove storage lfm:cache shortconv.layer{cache_layer}.prev2",
                    f"data modify storage lfm:cache shortconv.layer{cache_layer}.prev2 set from storage lfm:scratch shortconv0.prev1",
                    f"data modify storage lfm:cache shortconv.layer{cache_layer}.prev1 set from storage lfm:scratch shortconv0.conv_state",
                    f'data modify storage lfm:job phase set value "{next_phase}"',
                    "data modify storage lfm:job row set value 0",
                    "data modify storage lfm:job chunk set value 0",
                    "data modify storage lfm:job chunk_row set value 0",
                ]
            )
        _text(paths.functions / "runtime" / name / f"block_{block}.mcfunction", "\n".join(commands))
    _text(paths.functions / "runtime" / name / "step.mcfunction", "\n".join(f"execute if data storage lfm:job {{block:{block}}} run function lfm:runtime/{name}/block_{block}" for block in range(8)))
    _append_tick_dispatch(paths, [f'execute if data storage lfm:job {{phase:"{name}_load"}} run return run function lfm:runtime/{name}/load', f'execute if data storage lfm:job {{phase:"{name}_apply"}} run return run function lfm:runtime/{name}/step'])


def configure_reused_residual_ffn_norm_runtime(
    paths: PackPaths,
    name: str,
    ffn_norm_structure: str,
    next_phase: str,
) -> None:
    residual_name = f"{name}_residual"
    norm_name = f"{name}_ffn_norm"
    for block in range(8):
        residual_commands: list[str] = []
        norm_commands: list[str] = []
        for index in range(block * 256, (block + 1) * 256):
            residual_commands.extend([f"data modify storage lfm:scratch scalar set compute default float lfm:residual0/{index:04d}", "data modify storage lfm:scratch hidden0 append from storage lfm:scratch scalar"])
            norm_commands.extend([f"data modify storage lfm:scratch scalar set compute default float lfm:ffn_norm0/apply/{index:04d}", "data modify storage lfm:scratch ffn_input0 append from storage lfm:scratch scalar"])
        residual_commands.append(f"data modify storage lfm:job block set value {block + 1}")
        norm_commands.append(f"data modify storage lfm:job block set value {block + 1}")
        if block == 7:
            residual_commands.extend([f'data modify storage lfm:job phase set value "{norm_name}_load"', "data modify storage lfm:job block set value 0"])
            norm_commands.extend([f'data modify storage lfm:job phase set value "{next_phase}"', "data modify storage lfm:job row set value 0", "data modify storage lfm:job chunk set value 0", "data modify storage lfm:job chunk_row set value 0", "data modify storage lfm:job block set value 0"])
        _text(paths.functions / "runtime" / residual_name / f"block_{block}.mcfunction", "\n".join(residual_commands))
        _text(paths.functions / "runtime" / norm_name / f"block_{block}.mcfunction", "\n".join(norm_commands))
    _text(paths.functions / "runtime" / residual_name / "load.mcfunction", "\n".join(["data modify storage lfm:scratch hidden0 set value []", "data modify storage lfm:job block set value 0", f'data modify storage lfm:job phase set value "{residual_name}_apply"']))
    _text(paths.functions / "runtime" / residual_name / "step.mcfunction", "\n".join(f"execute if data storage lfm:job {{block:{block}}} run function lfm:runtime/{residual_name}/block_{block}" for block in range(8)))
    _text(
        paths.functions / "runtime" / norm_name / "load.mcfunction",
        "\n".join(
            [
                f"execute in lfm:runtime positioned 0 0 0 run place template lfm:runtime/{ffn_norm_structure}",
                "execute in lfm:runtime positioned 0 0 0 run data modify storage lfm:scratch rms1.weight set from entity @e[type=minecraft:marker,tag=lfm.weight_chunk,limit=1,sort=nearest] data",
                "execute in lfm:runtime run kill @e[type=minecraft:marker,tag=lfm.weight_chunk]",
                "data modify storage lfm:scratch rms1.l2 set compute default float lfm:ffn_norm0/l2",
                "data modify storage lfm:scratch rms1.scale set compute default float lfm:ffn_norm0/scale",
                "data modify storage lfm:scratch ffn_input0 set value []",
                "data modify storage lfm:job block set value 0",
                f'data modify storage lfm:job phase set value "{norm_name}_apply"',
            ]
        ),
    )
    _text(paths.functions / "runtime" / norm_name / "step.mcfunction", "\n".join(f"execute if data storage lfm:job {{block:{block}}} run function lfm:runtime/{norm_name}/block_{block}" for block in range(8)))
    _append_tick_dispatch(paths, [f'execute if data storage lfm:job {{phase:"{residual_name}_load"}} run return run function lfm:runtime/{residual_name}/load', f'execute if data storage lfm:job {{phase:"{residual_name}_apply"}} run return run function lfm:runtime/{residual_name}/step', f'execute if data storage lfm:job {{phase:"{norm_name}_load"}} run return run function lfm:runtime/{norm_name}/load', f'execute if data storage lfm:job {{phase:"{norm_name}_apply"}} run return run function lfm:runtime/{norm_name}/step'])


def configure_reused_swiglu_runtime(paths: PackPaths, name: str, next_phase: str) -> None:
    _text(paths.functions / "runtime" / name / "load.mcfunction", "\n".join(["data modify storage lfm:scratch ffn0.activated set value []", "data modify storage lfm:job block set value 0", f'data modify storage lfm:job phase set value "{name}_apply"']))
    for block in range(32):
        commands: list[str] = []
        for index in range(block * 256, (block + 1) * 256):
            commands.extend([f"data modify storage lfm:scratch scalar set compute default float lfm:swiglu0/{index:04d}", "data modify storage lfm:scratch ffn0.activated append from storage lfm:scratch scalar"])
        commands.append(f"data modify storage lfm:job block set value {block + 1}")
        if block == 31:
            commands.extend([f'data modify storage lfm:job phase set value "{next_phase}"', "data modify storage lfm:job row set value 0", "data modify storage lfm:job chunk set value 0", "data modify storage lfm:job chunk_row set value 0"])
        _text(paths.functions / "runtime" / name / f"block_{block}.mcfunction", "\n".join(commands))
    _text(paths.functions / "runtime" / name / "step.mcfunction", "\n".join(f"execute if data storage lfm:job {{block:{block}}} run function lfm:runtime/{name}/block_{block}" for block in range(32)))
    _append_tick_dispatch(paths, [f'execute if data storage lfm:job {{phase:"{name}_load"}} run return run function lfm:runtime/{name}/load', f'execute if data storage lfm:job {{phase:"{name}_apply"}} run return run function lfm:runtime/{name}/step'])


def configure_reused_final_residual_runtime(
    paths: PackPaths,
    name: str,
    output_path: str,
    layer: int,
    next_phase: str | None = None,
) -> None:
    _text(paths.functions / "runtime" / name / "load.mcfunction", "\n".join([f"data modify storage lfm:scratch {output_path} set value []", "data modify storage lfm:job block set value 0", f'data modify storage lfm:job phase set value "{name}_apply"']))
    for block in range(8):
        commands: list[str] = []
        for index in range(block * 256, (block + 1) * 256):
            commands.extend([f"data modify storage lfm:scratch scalar set compute default float lfm:ffn_residual0/{index:04d}", f"data modify storage lfm:scratch {output_path} append from storage lfm:scratch scalar"])
        commands.append(f"data modify storage lfm:job block set value {block + 1}")
        if block == 7:
            if next_phase is None:
                commands.extend(["data modify storage lfm:job progress_ppm set value 1000000", f'data modify storage lfm:job phase set value "layer{layer}_done"', 'data modify storage lfm:job state set value "done"', f'data modify storage lfm:job result set value {{token_id:1,piece:"<layer{layer}-output>",logit:0.0f,verified:0b}}'])
            else:
                commands.extend([f"data modify storage lfm:scratch activation set from storage lfm:scratch {output_path}", f"data modify storage lfm:job layer set value {layer + 1}", f'data modify storage lfm:job phase set value "{next_phase}"', "data modify storage lfm:job block set value 0"])
        _text(paths.functions / "runtime" / name / f"block_{block}.mcfunction", "\n".join(commands))
    _text(paths.functions / "runtime" / name / "step.mcfunction", "\n".join(f"execute if data storage lfm:job {{block:{block}}} run function lfm:runtime/{name}/block_{block}" for block in range(8)))
    _append_tick_dispatch(paths, [f'execute if data storage lfm:job {{phase:"{name}_load"}} run return run function lfm:runtime/{name}/load', f'execute if data storage lfm:job {{phase:"{name}_apply"}} run return run function lfm:runtime/{name}/step'])


def configure_shortconv_layer_runtime(
    paths: PackPaths,
    layer: int,
    next_phase: str | None = None,
    short_in_kind: str = "Q2_K",
    short_out_kind: str = "Q2_K",
    gate_kind: str = "Q2_K",
    up_kind: str = "Q2_K",
    down_kind: str = "Q3_K",
) -> None:
    if layer <= 0:
        raise ValueError("the reusable layer runtime is for layers after Layer 0")
    prefix = f"layer{layer}"
    structures = f"layers/{layer:03d}"
    configure_reused_rms_runtime(paths, f"{prefix}_attn_norm", f"{structures}/attn_norm", f"{prefix}_shortconv_in_load")
    configure_quant_matvec_runtime(paths, f"{prefix}_shortconv_in", short_in_kind, "normalized", "shortconv0.in_proj", 2048, 6144, f"{prefix}_shortconv_mix_load", provider_name="shortconv0_in", structure_name=f"{structures}/shortconv_in", write_float_providers=False)
    configure_reused_shortconv_mix_runtime(paths, f"{prefix}_shortconv_mix", f"{structures}/shortconv_conv", f"{prefix}_shortconv_out_load", cache_layer=layer)
    configure_quant_matvec_runtime(paths, f"{prefix}_shortconv_out", short_out_kind, "shortconv0.mixed", "shortconv0.output", 2048, 2048, f"{prefix}_residual_load", provider_name="shortconv0_out", structure_name=f"{structures}/shortconv_out", write_float_providers=False)
    configure_reused_residual_ffn_norm_runtime(paths, prefix, f"{structures}/ffn_norm", f"{prefix}_ffn_gate_load")
    configure_quant_matvec_runtime(paths, f"{prefix}_ffn_gate", gate_kind, "ffn_input0", "ffn0.gate", 2048, 8192, f"{prefix}_ffn_up_load", provider_name="ffn_gate0", structure_name=f"{structures}/ffn_gate", write_float_providers=False)
    configure_quant_matvec_runtime(paths, f"{prefix}_ffn_up", up_kind, "ffn_input0", "ffn0.up", 2048, 8192, f"{prefix}_swiglu_load", provider_name="ffn_up0", structure_name=f"{structures}/ffn_up", write_float_providers=False)
    configure_reused_swiglu_runtime(paths, f"{prefix}_swiglu", f"{prefix}_ffn_down_load")
    configure_quant_matvec_runtime(paths, f"{prefix}_ffn_down", down_kind, "ffn0.activated", "ffn0.down", 8192, 2048, f"{prefix}_ffn_residual_load", provider_name="ffn_down0", structure_name=f"{structures}/ffn_down", write_float_providers=False)
    configure_reused_final_residual_runtime(paths, f"{prefix}_ffn_residual", f"layer{layer}_output", layer, next_phase)


def configure_attention_head_norm_runtime(
    paths: PackPaths,
    name: str,
    input_path: str,
    output_path: str,
    heads: int,
    weight_structure: str,
    next_phase: str,
    write_providers: bool = True,
) -> None:
    if heads <= 0:
        raise ValueError("attention head count must be positive")
    if write_providers:
        _json(paths.float_providers / "attention_head_norm" / "l2.json", rms_l2_provider(64, "attn.head_input"))
        _json(paths.float_providers / "attention_head_norm" / "scale.json", rms_scale_provider(64, 1e-5, "attn.head_l2"))
        for index in range(64):
            _json(
                paths.float_providers / "attention_head_norm" / "apply" / f"{index:02d}.json",
                rms_apply_provider(index, "attn.head_input", "attn.head_scale", "attn.head_norm_weight.values"),
            )
    _text(
        paths.functions / "runtime" / name / "load.mcfunction",
        "\n".join(
            [
                f"execute in lfm:runtime positioned 0 0 0 run place template lfm:runtime/{weight_structure}",
                "execute in lfm:runtime positioned 0 0 0 run data modify storage lfm:scratch attn.head_norm_weight set from entity @e[type=minecraft:marker,tag=lfm.weight_chunk,limit=1,sort=nearest] data",
                "execute in lfm:runtime run kill @e[type=minecraft:marker,tag=lfm.weight_chunk]",
                f"data modify storage lfm:scratch {output_path} set value []",
                "data modify storage lfm:job block set value 0",
                f'data modify storage lfm:job phase set value "{name}_apply"',
            ]
        ),
    )
    for head in range(heads):
        commands = ["data modify storage lfm:scratch attn.head_input set value []"]
        for index in range(64):
            commands.append(f"data modify storage lfm:scratch attn.head_input append from storage lfm:scratch {input_path}[{head * 64 + index}]")
        commands.extend(
            [
                "data modify storage lfm:scratch attn.head_l2 set compute default float lfm:attention_head_norm/l2",
                "data modify storage lfm:scratch attn.head_scale set compute default float lfm:attention_head_norm/scale",
            ]
        )
        for index in range(64):
            commands.extend(
                [
                    f"data modify storage lfm:scratch scalar set compute default float lfm:attention_head_norm/apply/{index:02d}",
                    f"data modify storage lfm:scratch {output_path} append from storage lfm:scratch scalar",
                ]
            )
        commands.append(f"data modify storage lfm:job block set value {head + 1}")
        if head + 1 == heads:
            commands.extend(
                [
                    f'data modify storage lfm:job phase set value "{next_phase}"',
                    "data modify storage lfm:job row set value 0",
                    "data modify storage lfm:job chunk set value 0",
                    "data modify storage lfm:job chunk_row set value 0",
                    "data modify storage lfm:job block set value 0",
                ]
            )
        _text(paths.functions / "runtime" / name / f"head_{head:02d}.mcfunction", "\n".join(commands))
    _text(
        paths.functions / "runtime" / name / "step.mcfunction",
        "\n".join(
            f"execute if data storage lfm:job {{block:{head}}} run return run function lfm:runtime/{name}/head_{head:02d}"
            for head in range(heads)
        ),
    )
    _append_tick_dispatch(
        paths,
        [
            f'execute if data storage lfm:job {{phase:"{name}_load"}} run return run function lfm:runtime/{name}/load',
            f'execute if data storage lfm:job {{phase:"{name}_apply"}} run return run function lfm:runtime/{name}/step',
        ],
    )


def configure_attention_rope_runtime(
    paths: PackPaths,
    name: str,
    input_path: str,
    output_path: str,
    heads: int,
    next_phase: str,
    write_providers: bool = True,
) -> None:
    if write_providers:
        for index in range(64):
            _json(paths.float_providers / "attention_rope" / f"{index:02d}.json", rope_apply_provider(index))
    _text(
        paths.functions / "runtime" / name / "load.mcfunction",
        "\n".join(
            [
                f"data modify storage lfm:scratch {output_path} set value []",
                "data modify storage lfm:job block set value 0",
                f'data modify storage lfm:job phase set value "{name}_apply"',
            ]
        ),
    )
    for head in range(heads):
        commands = ["data modify storage lfm:scratch attn.head_input set value []"]
        for index in range(64):
            commands.append(f"data modify storage lfm:scratch attn.head_input append from storage lfm:scratch {input_path}[{head * 64 + index}]")
        for index in range(64):
            commands.extend(
                [
                    f"data modify storage lfm:scratch scalar set compute default float lfm:attention_rope/{index:02d}",
                    f"data modify storage lfm:scratch {output_path} append from storage lfm:scratch scalar",
                ]
            )
        commands.append(f"data modify storage lfm:job block set value {head + 1}")
        if head + 1 == heads:
            commands.extend(
                [
                    f'data modify storage lfm:job phase set value "{next_phase}"',
                    "data modify storage lfm:job row set value 0",
                    "data modify storage lfm:job chunk set value 0",
                    "data modify storage lfm:job chunk_row set value 0",
                    "data modify storage lfm:job block set value 0",
                ]
            )
        _text(paths.functions / "runtime" / name / f"head_{head:02d}.mcfunction", "\n".join(commands))
    _text(
        paths.functions / "runtime" / name / "step.mcfunction",
        "\n".join(
            f"execute if data storage lfm:job {{block:{head}}} run return run function lfm:runtime/{name}/head_{head:02d}"
            for head in range(heads)
        ),
    )
    _append_tick_dispatch(
        paths,
        [
            f'execute if data storage lfm:job {{phase:"{name}_load"}} run return run function lfm:runtime/{name}/load',
            f'execute if data storage lfm:job {{phase:"{name}_apply"}} run return run function lfm:runtime/{name}/step',
        ],
    )


def configure_cached_attention_runtime(
    paths: PackPaths,
    name: str,
    layer: int,
    next_phase: str,
    write_providers: bool = True,
) -> None:
    if write_providers:
        _json(paths.float_providers / "cached_attention" / "score.json", attention_score_provider())
        _json(paths.float_providers / "cached_attention" / "exp.json", attention_exp_provider())
        _json(paths.float_providers / "cached_attention" / "sum.json", attention_sum_provider())
        _json(paths.float_providers / "cached_attention" / "weight.json", attention_weight_provider())
        for index in range(64):
            _json(paths.float_providers / "cached_attention" / "weighted" / f"{index:02d}.json", attention_weighted_value_provider(index))
    job_value = lambda path: {
        "type": "minecraft:storage",
        "storage": "lfm:job",
        "path": path,
        "fallback": 0,
    }
    for provider_name, field in (("increment_context", "context_pos"), ("increment_head", "block")):
        _json(
            paths.int_providers / name / f"{provider_name}.json",
            {"type": "minecraft:add", "inputs": [job_value(field), 1]},
        )
    _json(
        paths.int_providers / name / "context_length.json",
        {"type": "minecraft:add", "inputs": [job_value("position"), 1]},
    )
    _text(
        paths.functions / "runtime" / name / "load.mcfunction",
        "\n".join(
            [
                f"execute unless data storage lfm:cache attention.layer{layer}.keys run data modify storage lfm:cache attention.layer{layer}.keys set value []",
                f"execute unless data storage lfm:cache attention.layer{layer}.values run data modify storage lfm:cache attention.layer{layer}.values set value []",
                f"data modify storage lfm:cache attention.layer{layer}.keys append from storage lfm:scratch attn.k_rope",
                f"data modify storage lfm:cache attention.layer{layer}.values append from storage lfm:scratch attn.v",
                f"data modify storage lfm:job context_length set compute default integer lfm:{name}/context_length",
                "data modify storage lfm:scratch attn.context_output set value []",
                "data modify storage lfm:scratch attn.args.layer set value " + str(layer),
                "data modify storage lfm:job block set value 0",
                f'data modify storage lfm:job phase set value "{name}_prepare_head"',
            ]
        ),
    )
    zero_head = "[" + ",".join("0.0f" for _ in range(64)) + "]"
    for head in range(32):
        prepare = [
            "data modify storage lfm:scratch attn.head_query set value []",
        ]
        for index in range(64):
            prepare.append(f"data modify storage lfm:scratch attn.head_query append from storage lfm:scratch attn.q_rope[{head * 64 + index}]")
        prepare.extend(
            [
                "data modify storage lfm:scratch attn.scores set value []",
                "data modify storage lfm:scratch attn.score_max set value -1000000.0f",
                "scoreboard players set #attn_max lfm.argmax -2147483648",
                "data modify storage lfm:job context_pos set value 0",
                f'data modify storage lfm:job phase set value "{name}_score"',
            ]
        )
        _text(paths.functions / "runtime" / name / f"prepare_head_{head:02d}.mcfunction", "\n".join(prepare))

        kv_base = (head // 4) * 64
        select_key = ["data modify storage lfm:scratch attn.head_key set value []"]
        select_value = ["data modify storage lfm:scratch attn.selected_value set value []"]
        for index in range(64):
            select_key.append(
                f"$data modify storage lfm:scratch attn.head_key append from storage lfm:cache attention.layer$(layer).keys[$(pos)][{kv_base + index}]"
            )
            select_value.append(
                f"$data modify storage lfm:scratch attn.selected_value append from storage lfm:cache attention.layer$(layer).values[$(pos)][{kv_base + index}]"
            )
        _text(paths.functions / "runtime" / name / f"select_key_{head:02d}.mcfunction", "\n".join(select_key))
        _text(paths.functions / "runtime" / name / f"select_value_{head:02d}.mcfunction", "\n".join(select_value))
        _text(
            paths.functions / "runtime" / name / f"score_head_{head:02d}.mcfunction",
            "\n".join(
                [
                    f"function lfm:runtime/{name}/select_key_{head:02d} with storage lfm:scratch attn.args",
                    f"function lfm:runtime/{name}/score_selected",
                ]
            ),
        )
        _text(
            paths.functions / "runtime" / name / f"weighted_head_{head:02d}.mcfunction",
            "\n".join(
                [
                    f"function lfm:runtime/{name}/select_value_{head:02d} with storage lfm:scratch attn.args",
                    f"function lfm:runtime/{name}/weighted_selected",
                ]
            ),
        )

    _text(
        paths.functions / "runtime" / name / "prepare_head.mcfunction",
        "\n".join(
            f"execute if data storage lfm:job {{block:{head}}} run return run function lfm:runtime/{name}/prepare_head_{head:02d}"
            for head in range(32)
        ),
    )
    _text(
        paths.functions / "runtime" / name / "score.mcfunction",
        "\n".join(
            [
                "data modify storage lfm:scratch attn.args.pos set from storage lfm:job context_pos",
                *[
                    f"execute if data storage lfm:job {{block:{head}}} run return run function lfm:runtime/{name}/score_head_{head:02d}"
                    for head in range(32)
                ],
            ]
        ),
    )
    _text(
        paths.functions / "runtime" / name / "score_selected.mcfunction",
        "\n".join(
            [
                "data modify storage lfm:scratch attn.temp_score set compute default float lfm:cached_attention/score",
                "data modify storage lfm:scratch attn.scores append from storage lfm:scratch attn.temp_score",
                "execute store result score #attn_current lfm.argmax run data get storage lfm:scratch attn.temp_score 10000000",
                "execute if score #attn_current lfm.argmax > #attn_max lfm.argmax run data modify storage lfm:scratch attn.score_max set from storage lfm:scratch attn.temp_score",
                "execute if score #attn_current lfm.argmax > #attn_max lfm.argmax run scoreboard players operation #attn_max lfm.argmax = #attn_current lfm.argmax",
                f"data modify storage lfm:job context_pos set compute default integer lfm:{name}/increment_context",
                f"execute if data storage lfm:job {{context_pos:$(context_length)}} run return 0",
            ]
        ).replace("$(context_length)", "0"),
    )
    # Dynamic loop bounds are checked by macros because context_length changes per generated token.
    _text(
        paths.functions / "runtime" / name / "score_done_check.mcfunction",
        "$execute if data storage lfm:job {context_pos:$(context_length)} run return run function lfm:runtime/" + name + "/begin_exp",
    )
    score_selected = paths.functions / "runtime" / name / "score_selected.mcfunction"
    score_lines = score_selected.read_text(encoding="utf-8").rstrip().splitlines()
    score_lines[-1] = f"function lfm:runtime/{name}/score_done_check with storage lfm:job"
    _text(score_selected, "\n".join(score_lines))
    _text(
        paths.functions / "runtime" / name / "begin_exp.mcfunction",
        "\n".join(
            [
                "data modify storage lfm:scratch attn.softmax_sum set value 0.0f",
                "data modify storage lfm:job context_pos set value 0",
                f'data modify storage lfm:job phase set value "{name}_exp"',
            ]
        ),
    )
    _text(
        paths.functions / "runtime" / name / "select_score.mcfunction",
        "$data modify storage lfm:scratch attn.selected_score set from storage lfm:scratch attn.scores[$(pos)]",
    )
    _text(
        paths.functions / "runtime" / name / "exp.mcfunction",
        "\n".join(
            [
                "data modify storage lfm:scratch attn.args.pos set from storage lfm:job context_pos",
                f"function lfm:runtime/{name}/select_score with storage lfm:scratch attn.args",
                "data modify storage lfm:scratch attn.temp_exp set compute default float lfm:cached_attention/exp",
                "data modify storage lfm:scratch attn.softmax_sum set compute default float lfm:cached_attention/sum",
                f"data modify storage lfm:job context_pos set compute default integer lfm:{name}/increment_context",
                f"function lfm:runtime/{name}/exp_done_check with storage lfm:job",
            ]
        ),
    )
    _text(
        paths.functions / "runtime" / name / "exp_done_check.mcfunction",
        "$execute if data storage lfm:job {context_pos:$(context_length)} run return run function lfm:runtime/" + name + "/begin_weighted",
    )
    _text(
        paths.functions / "runtime" / name / "begin_weighted.mcfunction",
        "\n".join(
            [
                f"data modify storage lfm:scratch attn.head_output set value {zero_head}",
                "data modify storage lfm:job context_pos set value 0",
                f'data modify storage lfm:job phase set value "{name}_weighted"',
            ]
        ),
    )
    _text(
        paths.functions / "runtime" / name / "weighted.mcfunction",
        "\n".join(
            [
                "data modify storage lfm:scratch attn.args.pos set from storage lfm:job context_pos",
                f"function lfm:runtime/{name}/select_score with storage lfm:scratch attn.args",
                *[
                    f"execute if data storage lfm:job {{block:{head}}} run return run function lfm:runtime/{name}/weighted_head_{head:02d}"
                    for head in range(32)
                ],
            ]
        ),
    )
    weighted_commands = [
        "data modify storage lfm:scratch attn.temp_exp set compute default float lfm:cached_attention/exp",
        "data modify storage lfm:scratch attn.temp_weight set compute default float lfm:cached_attention/weight",
    ]
    for index in range(64):
        weighted_commands.append(
            f"data modify storage lfm:scratch attn.head_output[{index}] set compute default float lfm:cached_attention/weighted/{index:02d}"
        )
    weighted_commands.extend(
        [
            f"data modify storage lfm:job context_pos set compute default integer lfm:{name}/increment_context",
            f"function lfm:runtime/{name}/weighted_done_check with storage lfm:job",
        ]
    )
    _text(paths.functions / "runtime" / name / "weighted_selected.mcfunction", "\n".join(weighted_commands))
    _text(
        paths.functions / "runtime" / name / "weighted_done_check.mcfunction",
        "$execute if data storage lfm:job {context_pos:$(context_length)} run return run function lfm:runtime/" + name + "/head_done",
    )
    head_done = []
    for index in range(64):
        head_done.append(f"data modify storage lfm:scratch attn.context_output append from storage lfm:scratch attn.head_output[{index}]")
    head_done.extend(
        [
            f"data modify storage lfm:job block set compute default integer lfm:{name}/increment_head",
            f"execute if data storage lfm:job {{block:32}} run return run function lfm:runtime/{name}/done",
            f'data modify storage lfm:job phase set value "{name}_prepare_head"',
        ]
    )
    _text(paths.functions / "runtime" / name / "head_done.mcfunction", "\n".join(head_done))
    _text(
        paths.functions / "runtime" / name / "done.mcfunction",
        "\n".join(
            [
                f'data modify storage lfm:job phase set value "{next_phase}"',
                "data modify storage lfm:job row set value 0",
                "data modify storage lfm:job chunk set value 0",
                "data modify storage lfm:job chunk_row set value 0",
                "data modify storage lfm:job block set value 0",
            ]
        ),
    )
    _append_tick_dispatch(
        paths,
        [
            f'execute if data storage lfm:job {{phase:"{name}_load"}} run return run function lfm:runtime/{name}/load',
            f'execute if data storage lfm:job {{phase:"{name}_prepare_head"}} run return run function lfm:runtime/{name}/prepare_head',
            f'execute if data storage lfm:job {{phase:"{name}_score"}} run return run function lfm:runtime/{name}/score',
            f'execute if data storage lfm:job {{phase:"{name}_exp"}} run return run function lfm:runtime/{name}/exp',
            f'execute if data storage lfm:job {{phase:"{name}_weighted"}} run return run function lfm:runtime/{name}/weighted',
        ],
    )


def configure_attention_expand_runtime(paths: PackPaths, name: str, next_phase: str, cache_layer: int) -> None:
    _text(
        paths.functions / "runtime" / name / "load.mcfunction",
        "\n".join(
            [
                f"execute unless data storage lfm:cache attention.layer{cache_layer}.keys run data modify storage lfm:cache attention.layer{cache_layer}.keys set value []",
                f"execute unless data storage lfm:cache attention.layer{cache_layer}.values run data modify storage lfm:cache attention.layer{cache_layer}.values set value []",
                f"data modify storage lfm:cache attention.layer{cache_layer}.keys append from storage lfm:scratch attn.k_norm",
                f"data modify storage lfm:cache attention.layer{cache_layer}.values append from storage lfm:scratch attn.v",
                "data modify storage lfm:scratch attn.expanded set value []",
                "data modify storage lfm:job block set value 0",
                f'data modify storage lfm:job phase set value "{name}_apply"',
            ]
        ),
    )
    for block in range(8):
        commands: list[str] = []
        for output_index in range(block * 256, (block + 1) * 256):
            query_head, lane = divmod(output_index, 64)
            kv_head = query_head // 4
            source_index = kv_head * 64 + lane
            commands.append(f"data modify storage lfm:scratch attn.expanded append from storage lfm:scratch attn.v[{source_index}]")
        commands.append(f"data modify storage lfm:job block set value {block + 1}")
        if block == 7:
            commands.extend([f'data modify storage lfm:job phase set value "{next_phase}"', "data modify storage lfm:job row set value 0", "data modify storage lfm:job chunk set value 0", "data modify storage lfm:job chunk_row set value 0"])
        _text(paths.functions / "runtime" / name / f"block_{block}.mcfunction", "\n".join(commands))
    _text(paths.functions / "runtime" / name / "step.mcfunction", "\n".join(f"execute if data storage lfm:job {{block:{block}}} run function lfm:runtime/{name}/block_{block}" for block in range(8)))
    _append_tick_dispatch(paths, [f'execute if data storage lfm:job {{phase:"{name}_load"}} run return run function lfm:runtime/{name}/load', f'execute if data storage lfm:job {{phase:"{name}_apply"}} run return run function lfm:runtime/{name}/step'])


def configure_attention_layer_runtime(
    paths: PackPaths,
    layer: int,
    next_phase: str | None = None,
    write_attention_providers: bool = True,
    key_kind: str = "Q2_K",
    value_kind: str = "Q3_K",
    query_kind: str = "Q2_K",
    output_kind: str = "Q3_K",
    gate_kind: str = "Q2_K",
    up_kind: str = "Q2_K",
    down_kind: str = "Q3_K",
) -> None:
    if layer <= 0:
        raise ValueError("the reusable attention runtime is for layers after Layer 0")
    prefix = f"layer{layer}"
    structures = f"layers/{layer:03d}"
    configure_reused_rms_runtime(paths, f"{prefix}_attn_norm", f"{structures}/attn_norm", f"{prefix}_attn_k_load")
    configure_quant_matvec_runtime(paths, f"{prefix}_attn_k", key_kind, "normalized", "attn.k", 2048, 512, f"{prefix}_attn_k_norm_load", provider_name="attention_k", structure_name=f"{structures}/attn_k", write_float_providers=write_attention_providers)
    configure_attention_head_norm_runtime(paths, f"{prefix}_attn_k_norm", "attn.k", "attn.k_norm", 8, f"{structures}/attn_k_norm", f"{prefix}_attn_k_rope_load", write_providers=write_attention_providers)
    configure_attention_rope_runtime(paths, f"{prefix}_attn_k_rope", "attn.k_norm", "attn.k_rope", 8, f"{prefix}_attn_v_load", write_providers=write_attention_providers)
    configure_quant_matvec_runtime(paths, f"{prefix}_attn_v", value_kind, "normalized", "attn.v", 2048, 512, f"{prefix}_attn_q_load", provider_name="attention_v", structure_name=f"{structures}/attn_v", write_float_providers=write_attention_providers)
    configure_quant_matvec_runtime(paths, f"{prefix}_attn_q", query_kind, "normalized", "attn.q", 2048, 2048, f"{prefix}_attn_q_norm_load", provider_name="attention_k", structure_name=f"{structures}/attn_q", write_float_providers=False)
    configure_attention_head_norm_runtime(paths, f"{prefix}_attn_q_norm", "attn.q", "attn.q_norm", 32, f"{structures}/attn_q_norm", f"{prefix}_attn_q_rope_load", write_providers=False)
    configure_attention_rope_runtime(paths, f"{prefix}_attn_q_rope", "attn.q_norm", "attn.q_rope", 32, f"{prefix}_attn_context_load", write_providers=False)
    configure_cached_attention_runtime(paths, f"{prefix}_attn_context", layer, f"{prefix}_attn_out_load", write_providers=write_attention_providers)
    configure_quant_matvec_runtime(paths, f"{prefix}_attn_out", output_kind, "attn.context_output", "shortconv0.output", 2048, 2048, f"{prefix}_residual_load", provider_name="attention_out", structure_name=f"{structures}/attn_out", write_float_providers=write_attention_providers)
    configure_reused_residual_ffn_norm_runtime(paths, prefix, f"{structures}/ffn_norm", f"{prefix}_ffn_gate_load")
    configure_quant_matvec_runtime(paths, f"{prefix}_ffn_gate", gate_kind, "ffn_input0", "ffn0.gate", 2048, 8192, f"{prefix}_ffn_up_load", provider_name="ffn_gate0", structure_name=f"{structures}/ffn_gate", write_float_providers=False)
    configure_quant_matvec_runtime(paths, f"{prefix}_ffn_up", up_kind, "ffn_input0", "ffn0.up", 2048, 8192, f"{prefix}_swiglu_load", provider_name="ffn_up0", structure_name=f"{structures}/ffn_up", write_float_providers=False)
    configure_reused_swiglu_runtime(paths, f"{prefix}_swiglu", f"{prefix}_ffn_down_load")
    configure_quant_matvec_runtime(paths, f"{prefix}_ffn_down", down_kind, "ffn0.activated", "ffn0.down", 8192, 2048, f"{prefix}_ffn_residual_load", provider_name="ffn_down0", structure_name=f"{structures}/ffn_down", write_float_providers=False)
    configure_reused_final_residual_runtime(paths, f"{prefix}_ffn_residual", f"layer{layer}_output", layer, next_phase)


def _append_tick_dispatch(paths: PackPaths, dispatch: list[str]) -> None:
    tick = paths.functions / "runtime" / "tick.mcfunction"
    lines = tick.read_text(encoding="utf-8").rstrip().splitlines()
    lines.extend(dispatch)
    _text(tick, "\n".join(lines))


def _api_functions(paths: PackPaths) -> None:
    api = paths.functions / "api"
    _text(
        api / "start_bos.mcfunction",
        "\n".join(
            [
                'execute unless data storage lfm:job {state:"idle"} run return fail',
                'data merge storage lfm:job {state:"running",phase:"kernel_test",input_token:1,result:{token_id:0,piece:"",logit:0.0f,verified:0b}}',
                'tellraw @s {"text":"[LFM] BOS inference started (kernel proof stage)","color":"green"}',
            ]
        ),
    )
    _text(api / "pause.mcfunction", 'execute if data storage lfm:job {state:"running"} run data modify storage lfm:job state set value "paused"')
    _text(api / "resume.mcfunction", 'execute if data storage lfm:job {state:"paused"} run data modify storage lfm:job state set value "running"\nexecute if data storage lfm:job {state:"recovered"} run data modify storage lfm:job state set value "running"')
    _text(api / "cancel.mcfunction", 'execute if data storage lfm:job {state:"running"} run data modify storage lfm:job state set value "cancelled"\ndata remove storage lfm:scratch chunk\ndata remove storage lfm:scratch weight\ndata remove storage lfm:scratch activation\ndata remove storage lfm:scratch row_acc')
    scratch_roots = ["embedding", "activation", "normalized", "weight", "matrix", "row_acc", "scalar", "rms0", "rms1", "shortconv0", "hidden0", "ffn_input0", "ffn0", "attn"] + [f"layer{layer}_output" for layer in range(16)]
    reset_commands = [f"data remove storage lfm:scratch {root}" for root in scratch_roots]
    reset_commands.extend(
        [
            "data remove storage lfm:cache shortconv",
            "data remove storage lfm:cache attention",
            "data remove storage lfm:cache tokens",
            "data modify storage lfm:cache position set value 0",
        ]
    )
    reset_commands.append('data merge storage lfm:job {model_sha:"",state:"idle",phase:"",layer:0,tensor:"",row:0,chunk:0,chunk_row:0,block:0,lane:0,processed_weights:0L,total_weights:0L,progress_ppm:0,elapsed_ticks:0L,input_token:1,result:{token_id:0,piece:"",logit:0.0f,verified:0b}}')
    _text(api / "reset.mcfunction", "\n".join(reset_commands))
    _text(api / "status.mcfunction", "data get storage lfm:job")


def _kernel_test(paths: PackPaths) -> None:
    raw = bytes(range(16)) + bytes((index * 37 + 11) & 0xFF for index in range(64))
    raw += bytes.fromhex("003c0038")  # d=1.0, dmin=0.5 in little-endian fp16
    block = Q2KBlock.from_bytes(raw)
    activations = [float((index % 17) - 8) / 8.0 for index in range(256)]
    q3_raw = bytes((index * 13 + 5) & 0xFF for index in range(108)) + bytes.fromhex("0028")
    q6_scales = bytes(((index * 11 + 3) & 0x7F) for index in range(16))
    q6_raw = bytes((index * 29 + 17) & 0xFF for index in range(192)) + q6_scales + bytes.fromhex("0024")
    q4_raw = b"".join(
        struct.pack("<e", (block_index + 1) / 8.0)
        + bytes((byte_index * 17 + block_index * 11 + 3) & 0xFF for byte_index in range(16))
        for block_index in range(8)
    )
    test_blocks = {
        "q40": Q40BlockGroup.from_bytes(q4_raw),
        "q2k": block,
        "q3k": Q3KBlock.from_bytes(q3_raw),
        "q6k": Q6KBlock.from_bytes(q6_raw),
    }
    for name, test_block in test_blocks.items():
        write_quant_chunk(
            paths.structures / "test" / f"{name}_block.nbt",
            f"synthetic.{name}",
            0,
            [test_block],
        )
        _kernel_test_function(paths, name, test_block, activations)
    _text(
        paths.functions / "test" / "run_all.mcfunction",
        "\n".join(
            [
                "function lfm:test/run_q40",
                "function lfm:test/run_q2k",
                "function lfm:test/run_q3k",
                "function lfm:test/run_q6k",
                'tellraw @a [{"text":"[LFM] quant kernel tests completed; inspect storage lfm:scratch tests","color":"aqua"}]',
            ]
        ),
    )
    _json(
        paths.root / "kernel_test.json",
        {
            name: {"expected": sum(w * a for w, a in zip(test_block.dequantize(), activations)), "width": 256}
            for name, test_block in test_blocks.items()
        },
    )


def _kernel_test_function(
    paths: PackPaths,
    name: str,
    block: Q40BlockGroup | Q2KBlock | Q3KBlock | Q6KBlock,
    activations: list[float],
) -> None:
    activation_snbt = "[" + ",".join(f"{value:.8g}f" for value in activations) + "]"
    expected = sum(weight * value for weight, value in zip(block.dequantize(), activations))
    _text(
        paths.functions / "test" / f"run_{name}.mcfunction",
        "\n".join(
            [
                f"execute in lfm:runtime positioned 0 0 0 run place template lfm:test/{name}_block",
                "execute in lfm:runtime positioned 0 0 0 run data modify storage lfm:scratch chunk set from entity @e[type=minecraft:marker,tag=lfm.weight_chunk,limit=1,sort=nearest] data",
                "execute in lfm:runtime run kill @e[type=minecraft:marker,tag=lfm.weight_chunk]",
                "data modify storage lfm:scratch weight set from storage lfm:scratch chunk.blocks[-1]",
                f"data modify storage lfm:scratch activation set value {activation_snbt}",
                "data modify storage lfm:scratch row_acc set value 0.0f",
                f"data modify storage lfm:scratch tests.{name}.actual set compute default float lfm:kernel/{name}_256",
                f"data modify storage lfm:scratch tests.{name}.expected set value {expected:.9g}f",
            ]
        ),
    )
    if name == "q2k":
        _text(
            paths.functions / "test" / "run_kernel.mcfunction",
            "\n".join(
                [
                    'data modify storage lfm:job state set value "paused"',
                    "function lfm:test/run_q2k",
                    "data modify storage lfm:scratch actual set from storage lfm:scratch tests.q2k.actual",
                    "data modify storage lfm:scratch expected set from storage lfm:scratch tests.q2k.expected",
                    'data merge storage lfm:job {state:"done",phase:"kernel_test_done",processed_weights:256L,total_weights:256L,progress_ppm:1000000,result:{token_id:0,piece:"<kernel-test>",logit:0.0f,verified:0b}}',
                    'tellraw @a[tag=lfm.operator] [{"text":"[LFM] Q2_K kernel finished. actual=","color":"aqua"},{"nbt":"actual","storage":"lfm:scratch"},{"text":" expected="},{"nbt":"expected","storage":"lfm:scratch"}]',
                ]
            ),
        )


def _json_dimension(paths: PackPaths) -> None:
    void_generator = {
        "type": "minecraft:flat",
        "settings": {
            "biome": "minecraft:the_void",
            "features": False,
            "lakes": False,
            "layers": [],
        },
    }
    _json(
        paths.namespace / "dimension" / "runtime.json",
        {
            "type": "minecraft:the_end",
            "generator": void_generator,
        },
    )
    _json(
        paths.namespace / "worldgen" / "world_preset" / "void.json",
        {
            "dimensions": {
                "minecraft:overworld": {
                    "type": "minecraft:overworld",
                    "generator": void_generator,
                }
            }
        },
    )


def _json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value.rstrip() + "\n", encoding="utf-8")
    if path.name == "tick.mcfunction" and path.parent.name == "runtime":
        _sync_batched_runtime(path, value)


def _sync_batched_runtime(tick_path: Path, value: str) -> None:
    row_dispatch = [
        line
        for line in value.rstrip().splitlines()
        if " run return run function " in line and line.endswith("/row")
    ]
    fast_path = tick_path.with_name("tick_fast.mcfunction")
    fast_path.write_text(("\n".join(row_dispatch) if row_dispatch else "return 0") + "\n", encoding="utf-8")

    classifier = ['data modify storage lfm:job batchable set value 0b']
    for line in row_dispatch:
        phase_match = line.split(" run return run function ", 1)[0].removeprefix("execute if data storage lfm:job ")
        classifier.append(
            f"execute if data storage lfm:job {phase_match} run data modify storage lfm:job batchable set value 1b"
        )
    tick_path.with_name("classify_batch.mcfunction").write_text(
        "\n".join(classifier) + "\n", encoding="utf-8"
    )
