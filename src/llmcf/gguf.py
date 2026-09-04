from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
import io
import json
import struct
from typing import Any, BinaryIO


GGUF_VALUE_TYPES = {
    0: "uint8",
    1: "int8",
    2: "uint16",
    3: "int16",
    4: "uint32",
    5: "int32",
    6: "float32",
    7: "bool",
    8: "string",
    9: "array",
    10: "uint64",
    11: "int64",
    12: "float64",
}

GGML_TYPES = {
    0: "F32",
    1: "F16",
    2: "Q4_0",
    3: "Q4_1",
    6: "Q5_0",
    7: "Q5_1",
    8: "Q8_0",
    9: "Q8_1",
    10: "Q2_K",
    11: "Q3_K",
    12: "Q4_K",
    13: "Q5_K",
    14: "Q6_K",
    15: "Q8_K",
    16: "IQ2_XXS",
    17: "IQ2_XS",
    18: "IQ3_XXS",
    19: "IQ1_S",
    20: "IQ4_NL",
    21: "IQ3_S",
    22: "IQ2_S",
    23: "IQ4_XS",
    24: "I8",
    25: "I16",
    26: "I32",
    27: "I64",
    28: "F64",
    30: "BF16",
}


class GGUFError(ValueError):
    pass


@dataclass(frozen=True)
class TensorInfo:
    name: str
    dimensions: tuple[int, ...]
    ggml_type: int
    offset: int
    nbytes: int = 0

    @property
    def type_name(self) -> str:
        return GGML_TYPES.get(self.ggml_type, f"UNKNOWN_{self.ggml_type}")

    @property
    def elements(self) -> int:
        product = 1
        for dimension in self.dimensions:
            product *= dimension
        return product


@dataclass(frozen=True)
class GGUFFile:
    path: Path
    version: int
    metadata: dict[str, Any]
    tensors: tuple[TensorInfo, ...]
    data_offset: int

    def tensor(self, name: str) -> TensorInfo:
        for tensor in self.tensors:
            if tensor.name == name:
                return tensor
        raise KeyError(name)

    def read_tensor_bytes(self, tensor: TensorInfo) -> bytes:
        with self.path.open("rb") as handle:
            handle.seek(self.data_offset + tensor.offset)
            return _read_exact(handle, tensor.nbytes)

    def manifest(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "data_offset": self.data_offset,
            "metadata": _json_safe(self.metadata),
            "tensor_count": len(self.tensors),
            "tensors": [
                {
                    "name": tensor.name,
                    "dimensions": list(tensor.dimensions),
                    "type": tensor.type_name,
                    "type_id": tensor.ggml_type,
                    "offset": tensor.offset,
                    "nbytes": tensor.nbytes,
                    "elements": tensor.elements,
                }
                for tensor in self.tensors
            ],
        }


def parse_gguf(path: str | Path) -> GGUFFile:
    source = Path(path)
    file_size = source.stat().st_size
    with source.open("rb") as handle:
        if _read_exact(handle, 4) != b"GGUF":
            raise GGUFError("not a GGUF file")
        version = _u32(handle)
        if version not in (2, 3):
            raise GGUFError(f"unsupported GGUF version: {version}")
        tensor_count = _u64(handle)
        metadata_count = _u64(handle)
        metadata: dict[str, Any] = {}
        for _ in range(metadata_count):
            key = _string(handle)
            value_type = _u32(handle)
            metadata[key] = _value(handle, value_type)

        raw_tensors: list[TensorInfo] = []
        for _ in range(tensor_count):
            name = _string(handle)
            rank = _u32(handle)
            dimensions = tuple(_u64(handle) for _ in range(rank))
            ggml_type = _u32(handle)
            offset = _u64(handle)
            raw_tensors.append(TensorInfo(name, dimensions, ggml_type, offset))

        alignment = int(metadata.get("general.alignment", 32))
        if alignment <= 0 or alignment & (alignment - 1):
            raise GGUFError(f"invalid alignment: {alignment}")
        data_offset = (handle.tell() + alignment - 1) // alignment * alignment

    ordered = sorted(enumerate(raw_tensors), key=lambda pair: pair[1].offset)
    sizes: dict[int, int] = {}
    for ordered_index, (original_index, tensor) in enumerate(ordered):
        next_offset = (
            ordered[ordered_index + 1][1].offset
            if ordered_index + 1 < len(ordered)
            else file_size - data_offset
        )
        nbytes = next_offset - tensor.offset
        if nbytes < 0:
            raise GGUFError(f"overlapping tensor offsets near {tensor.name}")
        sizes[original_index] = nbytes

    tensors = tuple(
        TensorInfo(t.name, t.dimensions, t.ggml_type, t.offset, sizes[index])
        for index, t in enumerate(raw_tensors)
    )
    return GGUFFile(source, version, metadata, tensors, data_offset)


def file_sha256(path: str | Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def write_manifest(model: GGUFFile, destination: str | Path) -> None:
    output = Path(destination)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(model.manifest(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _value(handle: BinaryIO, value_type: int) -> Any:
    if value_type == 0:
        return _unpack(handle, "B")
    if value_type == 1:
        return _unpack(handle, "b")
    if value_type == 2:
        return _unpack(handle, "H")
    if value_type == 3:
        return _unpack(handle, "h")
    if value_type == 4:
        return _u32(handle)
    if value_type == 5:
        return _unpack(handle, "i")
    if value_type == 6:
        return _unpack(handle, "f")
    if value_type == 7:
        return bool(_unpack(handle, "B"))
    if value_type == 8:
        return _string(handle)
    if value_type == 9:
        element_type = _u32(handle)
        length = _u64(handle)
        return [_value(handle, element_type) for _ in range(length)]
    if value_type == 10:
        return _u64(handle)
    if value_type == 11:
        return _unpack(handle, "q")
    if value_type == 12:
        return _unpack(handle, "d")
    raise GGUFError(
        f"unknown GGUF metadata value type {value_type} "
        f"({GGUF_VALUE_TYPES.get(value_type, 'unknown')})"
    )


def _string(handle: BinaryIO) -> str:
    length = _u64(handle)
    try:
        return _read_exact(handle, length).decode("utf-8")
    except UnicodeDecodeError as error:
        raise GGUFError("invalid UTF-8 in GGUF string") from error


def _u32(handle: BinaryIO) -> int:
    return _unpack(handle, "I")


def _u64(handle: BinaryIO) -> int:
    return _unpack(handle, "Q")


def _unpack(handle: BinaryIO, fmt: str) -> Any:
    return struct.unpack("<" + fmt, _read_exact(handle, struct.calcsize(fmt)))[0]


def _read_exact(handle: BinaryIO, length: int) -> bytes:
    data = handle.read(length)
    if len(data) != length:
        raise GGUFError(f"unexpected EOF: needed {length}, got {len(data)}")
    return data


def _json_safe(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.hex()
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    return value
