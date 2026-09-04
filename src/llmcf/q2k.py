from __future__ import annotations

from dataclasses import dataclass
import math
import struct
from typing import Iterable, Sequence


QK_K = 256
Q4_0_BLOCK_SIZE = 32
Q4_0_BLOCK_BYTES = 18
Q4_0_GROUP_BYTES = Q4_0_BLOCK_BYTES * (QK_K // Q4_0_BLOCK_SIZE)
Q2_K_BLOCK_BYTES = 84
Q3_K_BLOCK_BYTES = 110
Q6_K_BLOCK_BYTES = 210


@dataclass(frozen=True)
class Q40BlockGroup:
    """Eight consecutive GGML Q4_0 blocks represented as one 256-value page."""

    d: tuple[float, ...]
    qs: bytes

    @classmethod
    def from_bytes(cls, raw: bytes) -> "Q40BlockGroup":
        if len(raw) != Q4_0_GROUP_BYTES:
            raise ValueError(f"Q4_0 group must be {Q4_0_GROUP_BYTES} bytes")
        scales: list[float] = []
        quants = bytearray()
        for offset in range(0, len(raw), Q4_0_BLOCK_BYTES):
            scales.append(float(struct.unpack_from("<e", raw, offset)[0]))
            quants.extend(raw[offset + 2 : offset + Q4_0_BLOCK_BYTES])
        return cls(tuple(scales), bytes(quants))

    def dequantize(self) -> list[float]:
        if len(self.d) != 8 or len(self.qs) != 128:
            raise ValueError("invalid Q4_0 group arrays")
        result: list[float] = []
        for block, scale in enumerate(self.d):
            q_base = block * 16
            result.extend(scale * ((self.qs[q_base + lane] & 0x0F) - 8) for lane in range(16))
            result.extend(scale * ((self.qs[q_base + lane] >> 4) - 8) for lane in range(16))
        if len(result) != QK_K:
            raise AssertionError(len(result))
        return result

    def dot(self, activations: Sequence[float]) -> float:
        if len(activations) != QK_K:
            raise ValueError(f"activation block must have {QK_K} values")
        return math.fsum(weight * value for weight, value in zip(self.dequantize(), activations))

    def packed_qs(self) -> tuple[int, ...]:
        return pack_bytes_i32(self.qs)


@dataclass(frozen=True)
class Q2KBlock:
    scales: bytes
    qs: bytes
    d: float
    dmin: float

    @classmethod
    def from_bytes(cls, raw: bytes) -> "Q2KBlock":
        if len(raw) != Q2_K_BLOCK_BYTES:
            raise ValueError(f"Q2_K block must be {Q2_K_BLOCK_BYTES} bytes")
        scales = raw[:16]
        qs = raw[16:80]
        d = struct.unpack_from("<e", raw, 80)[0]
        dmin = struct.unpack_from("<e", raw, 82)[0]
        return cls(scales, qs, float(d), float(dmin))

    def to_bytes(self) -> bytes:
        if len(self.scales) != 16 or len(self.qs) != 64:
            raise ValueError("invalid Q2_K block arrays")
        return self.scales + self.qs + struct.pack("<ee", self.d, self.dmin)

    def dequantize(self) -> list[float]:
        result: list[float] = []
        scale_index = 0
        q_base = 0
        for _ in range(2):
            shift = 0
            for _ in range(4):
                for q_offset in (0, 16):
                    scale = self.scales[scale_index]
                    scale_index += 1
                    dl = self.d * (scale & 0x0F)
                    ml = self.dmin * (scale >> 4)
                    for lane in range(16):
                        quant = (self.qs[q_base + q_offset + lane] >> shift) & 0x03
                        result.append(dl * quant - ml)
                shift += 2
            q_base += 32
        if len(result) != QK_K:
            raise AssertionError(len(result))
        return result

    def dot(self, activations: Sequence[float]) -> float:
        if len(activations) != QK_K:
            raise ValueError(f"activation block must have {QK_K} values")
        return math.fsum(weight * value for weight, value in zip(self.dequantize(), activations))

    def packed_scales(self) -> tuple[int, ...]:
        return pack_bytes_i32(self.scales)

    def packed_qs(self) -> tuple[int, ...]:
        return pack_bytes_i32(self.qs)


@dataclass(frozen=True)
class Q3KBlock:
    hmask: bytes
    qs: bytes
    scales: tuple[int, ...]
    d: float

    @classmethod
    def from_bytes(cls, raw: bytes) -> "Q3KBlock":
        if len(raw) != Q3_K_BLOCK_BYTES:
            raise ValueError(f"Q3_K block must be {Q3_K_BLOCK_BYTES} bytes")
        return cls(
            raw[:32],
            raw[32:96],
            unpack_q3k_scales(raw[96:108]),
            float(struct.unpack_from("<e", raw, 108)[0]),
        )

    def dequantize(self) -> list[float]:
        result: list[float] = []
        scale_index = 0
        q_base = 0
        mask = 1
        for _ in range(2):
            shift = 0
            for _ in range(4):
                for q_offset in (0, 16):
                    dl = self.d * self.scales[scale_index]
                    scale_index += 1
                    for lane in range(16):
                        index = q_offset + lane
                        low = (self.qs[q_base + index] >> shift) & 3
                        quant = low - (0 if self.hmask[index] & mask else 4)
                        result.append(dl * quant)
                shift += 2
                mask <<= 1
            q_base += 32
        return result

    def dot(self, activations: Sequence[float]) -> float:
        if len(activations) != QK_K:
            raise ValueError(f"activation block must have {QK_K} values")
        return math.fsum(weight * value for weight, value in zip(self.dequantize(), activations))

    def packed_hmask(self) -> tuple[int, ...]:
        return pack_bytes_i32(self.hmask)

    def packed_qs(self) -> tuple[int, ...]:
        return pack_bytes_i32(self.qs)


@dataclass(frozen=True)
class Q6KBlock:
    ql: bytes
    qh: bytes
    scales: tuple[int, ...]
    d: float

    @classmethod
    def from_bytes(cls, raw: bytes) -> "Q6KBlock":
        if len(raw) != Q6_K_BLOCK_BYTES:
            raise ValueError(f"Q6_K block must be {Q6_K_BLOCK_BYTES} bytes")
        scales = tuple(struct.unpack("<16b", raw[192:208]))
        return cls(
            raw[:128],
            raw[128:192],
            scales,
            float(struct.unpack_from("<e", raw, 208)[0]),
        )

    def dequantize(self) -> list[float]:
        result = [0.0] * 256
        for half in range(2):
            ql_base = half * 64
            qh_base = half * 32
            scale_base = half * 8
            out_base = half * 128
            for lane in range(32):
                scale_lane = lane // 16
                low_a = self.ql[ql_base + lane]
                low_b = self.ql[ql_base + lane + 32]
                high = self.qh[qh_base + lane]
                quants = (
                    (low_a & 0xF) | (((high >> 0) & 3) << 4),
                    (low_b & 0xF) | (((high >> 2) & 3) << 4),
                    (low_a >> 4) | (((high >> 4) & 3) << 4),
                    (low_b >> 4) | (((high >> 6) & 3) << 4),
                )
                for segment, quant in enumerate(quants):
                    scale = self.scales[scale_base + scale_lane + segment * 2]
                    result[out_base + lane + segment * 32] = self.d * scale * (quant - 32)
        return result

    def dot(self, activations: Sequence[float]) -> float:
        if len(activations) != QK_K:
            raise ValueError(f"activation block must have {QK_K} values")
        return math.fsum(weight * value for weight, value in zip(self.dequantize(), activations))

    def packed_ql(self) -> tuple[int, ...]:
        return pack_bytes_i32(self.ql)

    def packed_qh(self) -> tuple[int, ...]:
        return pack_bytes_i32(self.qh)


def iter_q2k_blocks(raw: bytes) -> Iterable[Q2KBlock]:
    if len(raw) % Q2_K_BLOCK_BYTES:
        raise ValueError("Q2_K tensor byte length is not block-aligned")
    for offset in range(0, len(raw), Q2_K_BLOCK_BYTES):
        yield Q2KBlock.from_bytes(raw[offset : offset + Q2_K_BLOCK_BYTES])


def pack_bytes_i32(raw: bytes) -> tuple[int, ...]:
    if len(raw) % 4:
        raise ValueError("packed bytes must be a multiple of four")
    return tuple(struct.unpack(f"<{len(raw) // 4}i", raw))


def unpack_i32_bytes(words: Sequence[int]) -> bytes:
    return struct.pack(f"<{len(words)}i", *words)


def unpack_q3k_scales(raw: bytes) -> tuple[int, ...]:
    if len(raw) != 12:
        raise ValueError("Q3_K scales must be 12 bytes")
    aux = bytearray(16)
    aux[:12] = raw
    a0, a1, tmp = struct.unpack_from("<III", aux)
    mask2 = 0x0F0F0F0F
    mask1 = 0x03030303
    values = (
        (a0 & mask2) | (((tmp >> 0) & mask1) << 4),
        (a1 & mask2) | (((tmp >> 2) & mask1) << 4),
        ((a0 >> 4) & mask2) | (((tmp >> 4) & mask1) << 4),
        ((a1 >> 4) & mask2) | (((tmp >> 6) & mask1) << 4),
    )
    unpacked = struct.pack("<IIII", *values)
    return tuple(value - 32 for value in unpacked)
