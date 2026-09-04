from pathlib import Path
import gzip
import struct
import tempfile
import unittest

from llmcf.q2k import Q2KBlock, Q40BlockGroup
from llmcf.nbt import _modified_utf8
from llmcf.structure import (
    write_character_token_table,
    write_q2k_chunk,
    write_quant_chunk,
    write_rope_table,
    write_string_table,
)


class StructureTests(unittest.TestCase):
    def test_modified_utf8_handles_nul_and_supplementary_characters(self):
        self.assertEqual(_modified_utf8("\0"), bytes.fromhex("c080"))
        self.assertEqual(_modified_utf8("😀"), bytes.fromhex("eda0bdedb880"))

    def test_writes_gzip_structure(self):
        block = Q2KBlock(bytes(16), bytes(64), 1.0, 0.5)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "chunk.nbt"
            write_q2k_chunk(path, "test.weight", 0, [block])
            decoded = gzip.decompress(path.read_bytes())
        self.assertEqual(decoded[0], 10)
        self.assertIn(b"minecraft:marker", decoded)
        self.assertIn(b"lfm.weight_chunk", decoded)

    def test_writes_string_table_structure(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tokens.nbt"
            write_string_table(path, "tokens", ["<bos>", "1", "Ġhello"], 7)
            decoded = gzip.decompress(path.read_bytes())
        self.assertIn(b"tokens", decoded)
        self.assertIn(b"<bos>", decoded)
        self.assertIn("Ġhello".encode("utf-8"), decoded)

    def test_writes_character_token_table(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "chars.nbt"
            write_character_token_table(path, {"こ": (11, 12, 13), "A": (14,)}, 9)
            decoded = gzip.decompress(path.read_bytes())
        self.assertIn("こ".encode("utf-8"), decoded)
        self.assertIn(b"minecraft_input_utf8_byte_tokens", decoded)

    def test_writes_q4_0_group_structure(self):
        block = Q40BlockGroup((1.0,) * 8, bytes(128))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "q4.nbt"
            write_quant_chunk(path, "q4.weight", 0, [block])
            decoded = gzip.decompress(path.read_bytes())
        self.assertIn(b"q4.weight", decoded)
        self.assertIn(b"qs", decoded)
        self.assertIn(b"d", decoded)

    def test_writes_rope_table_structure(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rope.nbt"
            write_rope_table(path, [[1.0, 0.5], [0.0, -0.5]], [[0.0, 0.5], [1.0, 0.5]], 8)
            decoded = gzip.decompress(path.read_bytes())
        self.assertIn(b"positions", decoded)
        self.assertIn(b"cos", decoded)
        self.assertIn(b"sin", decoded)


if __name__ == "__main__":
    unittest.main()
