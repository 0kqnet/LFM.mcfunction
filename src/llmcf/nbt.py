from __future__ import annotations

from dataclasses import dataclass
import gzip
import io
from pathlib import Path
import struct
from typing import Any, BinaryIO, Mapping, Sequence


TAG_END = 0
TAG_BYTE = 1
TAG_SHORT = 2
TAG_INT = 3
TAG_LONG = 4
TAG_FLOAT = 5
TAG_DOUBLE = 6
TAG_BYTE_ARRAY = 7
TAG_STRING = 8
TAG_LIST = 9
TAG_COMPOUND = 10
TAG_INT_ARRAY = 11
TAG_LONG_ARRAY = 12


@dataclass(frozen=True)
class Byte:
    value: int


@dataclass(frozen=True)
class Short:
    value: int


@dataclass(frozen=True)
class Int:
    value: int


@dataclass(frozen=True)
class Long:
    value: int


@dataclass(frozen=True)
class Float:
    value: float


@dataclass(frozen=True)
class Double:
    value: float


@dataclass(frozen=True)
class ByteArray:
    value: bytes


@dataclass(frozen=True)
class IntArray:
    value: Sequence[int]


@dataclass(frozen=True)
class LongArray:
    value: Sequence[int]


@dataclass(frozen=True)
class ListTag:
    element_type: int
    value: Sequence[Any]


def write_gzip_nbt(
    path: str | Path,
    root: Mapping[str, Any],
    root_name: str = "",
    compresslevel: int = 1,
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as raw_handle:
        with gzip.GzipFile(fileobj=raw_handle, mode="wb", compresslevel=compresslevel, mtime=0) as handle:
            _write_u8(handle, TAG_COMPOUND)
            _write_string(handle, root_name)
            _write_payload(handle, TAG_COMPOUND, root)


def write_nbt(handle: BinaryIO, root: Mapping[str, Any], root_name: str = "") -> None:
    _write_u8(handle, TAG_COMPOUND)
    _write_string(handle, root_name)
    _write_payload(handle, TAG_COMPOUND, root)


def infer_type(value: Any) -> int:
    if isinstance(value, Byte):
        return TAG_BYTE
    if isinstance(value, Short):
        return TAG_SHORT
    if isinstance(value, Int):
        return TAG_INT
    if isinstance(value, Long):
        return TAG_LONG
    if isinstance(value, Float):
        return TAG_FLOAT
    if isinstance(value, Double):
        return TAG_DOUBLE
    if isinstance(value, ByteArray):
        return TAG_BYTE_ARRAY
    if isinstance(value, str):
        return TAG_STRING
    if isinstance(value, ListTag):
        return TAG_LIST
    if isinstance(value, Mapping):
        return TAG_COMPOUND
    if isinstance(value, IntArray):
        return TAG_INT_ARRAY
    if isinstance(value, LongArray):
        return TAG_LONG_ARRAY
    raise TypeError(f"NBT value requires an explicit wrapper: {value!r}")


def _write_payload(handle: BinaryIO, tag_type: int, value: Any) -> None:
    if tag_type == TAG_BYTE:
        _write(handle, ">b", value.value)
    elif tag_type == TAG_SHORT:
        _write(handle, ">h", value.value)
    elif tag_type == TAG_INT:
        _write(handle, ">i", value.value)
    elif tag_type == TAG_LONG:
        _write(handle, ">q", value.value)
    elif tag_type == TAG_FLOAT:
        _write(handle, ">f", value.value)
    elif tag_type == TAG_DOUBLE:
        _write(handle, ">d", value.value)
    elif tag_type == TAG_BYTE_ARRAY:
        _write(handle, ">i", len(value.value))
        handle.write(value.value)
    elif tag_type == TAG_STRING:
        _write_string(handle, value)
    elif tag_type == TAG_LIST:
        _write_u8(handle, value.element_type)
        _write(handle, ">i", len(value.value))
        for item in value.value:
            _write_payload(handle, value.element_type, item)
    elif tag_type == TAG_COMPOUND:
        for name, child in value.items():
            child_type = infer_type(child)
            _write_u8(handle, child_type)
            _write_string(handle, name)
            _write_payload(handle, child_type, child)
        _write_u8(handle, TAG_END)
    elif tag_type == TAG_INT_ARRAY:
        _write(handle, ">i", len(value.value))
        for item in value.value:
            _write(handle, ">i", item)
    elif tag_type == TAG_LONG_ARRAY:
        _write(handle, ">i", len(value.value))
        for item in value.value:
            _write(handle, ">q", item)
    else:
        raise TypeError(f"unsupported NBT tag type: {tag_type}")


def _write_string(handle: BinaryIO, value: str) -> None:
    encoded = _modified_utf8(value)
    if len(encoded) > 65535:
        raise ValueError("NBT string is too long")
    _write(handle, ">H", len(encoded))
    handle.write(encoded)


def _modified_utf8(value: str) -> bytes:
    """Encode the Java DataInput/DataOutput modified UTF-8 used by NBT."""
    result = bytearray()
    utf16 = value.encode("utf-16-be", errors="surrogatepass")
    for offset in range(0, len(utf16), 2):
        code_unit = (utf16[offset] << 8) | utf16[offset + 1]
        if 0x0001 <= code_unit <= 0x007F:
            result.append(code_unit)
        elif code_unit <= 0x07FF:
            result.extend((0xC0 | (code_unit >> 6), 0x80 | (code_unit & 0x3F)))
        else:
            result.extend(
                (
                    0xE0 | (code_unit >> 12),
                    0x80 | ((code_unit >> 6) & 0x3F),
                    0x80 | (code_unit & 0x3F),
                )
            )
    return bytes(result)


def _write_u8(handle: BinaryIO, value: int) -> None:
    _write(handle, ">B", value)


def _write(handle: BinaryIO, fmt: str, value: Any) -> None:
    handle.write(struct.pack(fmt, value))
