import io
from pathlib import Path
import struct
import tempfile
import unittest

from llmcf.gguf import parse_gguf


def _string(value: str) -> bytes:
    encoded = value.encode("utf-8")
    return struct.pack("<Q", len(encoded)) + encoded


class GGUFTests(unittest.TestCase):
    def test_minimal_file(self):
        header = io.BytesIO()
        header.write(b"GGUF")
        header.write(struct.pack("<IQQ", 3, 1, 2))
        header.write(_string("general.architecture"))
        header.write(struct.pack("<I", 8))
        header.write(_string("lfm2"))
        header.write(_string("general.alignment"))
        header.write(struct.pack("<II", 4, 32))
        header.write(_string("token_embd.weight"))
        header.write(struct.pack("<IQQI", 2, 256, 1, 10))
        header.write(struct.pack("<Q", 0))
        padding = (-header.tell()) % 32
        raw = header.getvalue() + bytes(padding) + bytes(84)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tiny.gguf"
            path.write_bytes(raw)
            model = parse_gguf(path)
        self.assertEqual(model.version, 3)
        self.assertEqual(model.metadata["general.architecture"], "lfm2")
        self.assertEqual(model.data_offset % 32, 0)
        self.assertEqual(model.tensors[0].dimensions, (256, 1))
        self.assertEqual(model.tensors[0].type_name, "Q2_K")
        self.assertEqual(model.tensors[0].nbytes, 84)


if __name__ == "__main__":
    unittest.main()
