from __future__ import annotations

from collections.abc import Iterable, Sequence


def gpt2_byte_encoder() -> dict[int, str]:
    """Return the reversible byte-to-Unicode alphabet used by GPT-2 BPE."""
    byte_values = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    codepoints = list(byte_values)
    extra = 0
    for value in range(256):
        if value not in byte_values:
            byte_values.append(value)
            codepoints.append(256 + extra)
            extra += 1
    return dict(zip(byte_values, map(chr, codepoints)))


def gpt2_byte_decoder() -> dict[str, int]:
    return {symbol: value for value, symbol in gpt2_byte_encoder().items()}


def byte_token_ids(vocabulary: Sequence[str]) -> tuple[int, ...]:
    token_by_piece = {piece: token_id for token_id, piece in enumerate(vocabulary)}
    result: list[int] = []
    for value in range(256):
        piece = gpt2_byte_encoder()[value]
        if piece not in token_by_piece:
            raise ValueError(f"vocabulary has no GPT-2 base token for byte {value}")
        result.append(token_by_piece[piece])
    return tuple(result)


def encode_utf8_as_byte_tokens(text: str, vocabulary: Sequence[str]) -> list[int]:
    ids = byte_token_ids(vocabulary)
    return [ids[value] for value in text.encode("utf-8")]


def decode_gpt2_piece(piece: str) -> str:
    decoder = gpt2_byte_decoder()
    if not piece or any(char not in decoder for char in piece):
        return piece
    raw = bytes(decoder[char] for char in piece)
    return raw.decode("utf-8", errors="replace")


def display_vocabulary(vocabulary: Sequence[str], token_types: Sequence[int]) -> list[str]:
    if len(vocabulary) != len(token_types):
        raise ValueError("token strings and types must have equal length")
    return [piece if token_type == 3 else decode_gpt2_piece(piece) for piece, token_type in zip(vocabulary, token_types)]


def supported_input_characters() -> Iterable[str]:
    """Characters accepted by the in-Minecraft byte tokenizer.

    The ranges cover normal chat punctuation, Latin text, Japanese kana,
    CJK ideographs (including Extension A), and full-width forms.
    """
    yield "\t"
    yield "\n"
    yield "\r"
    ranges = (
        (0x20, 0x024F),
        (0x2000, 0x206F),
        (0x3000, 0x30FF),
        (0x31F0, 0x31FF),
        (0x3400, 0x4DBF),
        (0x4E00, 0x9FFF),
        (0xF900, 0xFAFF),
        (0xFF00, 0xFFEF),
    )
    for first, last in ranges:
        for codepoint in range(first, last + 1):
            yield chr(codepoint)


def character_byte_token_table(vocabulary: Sequence[str]) -> dict[str, tuple[int, ...]]:
    ids = byte_token_ids(vocabulary)
    decoder = gpt2_byte_decoder()
    direct: dict[str, int] = {}
    for token_id, piece in enumerate(vocabulary):
        if not piece or any(char not in decoder for char in piece):
            continue
        raw = bytes(decoder[char] for char in piece)
        try:
            decoded = raw.decode("utf-8")
        except UnicodeDecodeError:
            continue
        if len(decoded) == 1:
            direct.setdefault(decoded, token_id)
    return {
        char: (direct[char],) if char in direct else tuple(ids[value] for value in char.encode("utf-8"))
        for char in supported_input_characters()
    }
