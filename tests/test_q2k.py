import math
import random
import struct
import unittest

from llmcf.q2k import Q2KBlock, Q3KBlock, Q6KBlock, Q40BlockGroup, pack_bytes_i32, unpack_i32_bytes


class Q2KTests(unittest.TestCase):
    def test_pack_i32_round_trip(self):
        raw = bytes(range(256))
        self.assertEqual(unpack_i32_bytes(pack_bytes_i32(raw)), raw)

    def test_known_dequant_layout(self):
        scales = bytes([0x11] * 16)
        qs = bytes(index % 4 for index in range(64))
        block = Q2KBlock(scales, qs, 2.0, 0.5)
        values = block.dequantize()
        self.assertEqual(len(values), 256)
        self.assertEqual(values[:4], [-0.5, 1.5, 3.5, 5.5])

    def test_bytes_round_trip(self):
        rng = random.Random(7)
        raw = rng.randbytes(80) + struct.pack("<ee", 0.125, 0.25)
        block = Q2KBlock.from_bytes(raw)
        self.assertEqual(block.to_bytes(), raw)
        self.assertTrue(all(math.isfinite(value) for value in block.dequantize()))

    def test_q3k_all_zero_payload(self):
        raw = bytes(108) + struct.pack("<e", 1.0)
        block = Q3KBlock.from_bytes(raw)
        values = block.dequantize()
        self.assertEqual(values, [128.0] * 256)
        self.assertEqual(block.dot([1.0] * 256), 32768.0)

    def test_q6k_simple_payload(self):
        raw = bytes(192) + bytes([1] * 16) + struct.pack("<e", 1.0)
        block = Q6KBlock.from_bytes(raw)
        values = block.dequantize()
        self.assertEqual(values, [-32.0] * 256)
        self.assertEqual(block.dot([1.0] * 256), -8192.0)

    def test_q4_0_group_layout(self):
        raw = b"".join(
            struct.pack("<e", float(block + 1)) + bytes([0xF0] * 16)
            for block in range(8)
        )
        group = Q40BlockGroup.from_bytes(raw)
        values = group.dequantize()
        self.assertEqual(len(values), 256)
        self.assertEqual(values[:16], [-8.0] * 16)
        self.assertEqual(values[16:32], [7.0] * 16)
        self.assertEqual(values[224:240], [-64.0] * 16)
        self.assertEqual(values[240:], [56.0] * 16)
        self.assertEqual(group.dot([1.0] * 256), -576.0)


if __name__ == "__main__":
    unittest.main()
