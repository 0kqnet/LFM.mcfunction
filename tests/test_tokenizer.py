import unittest
from pathlib import Path
import tempfile

from llmcf.generator import configure_minecraft_input_runtime, generate_kernel_pack
from llmcf.tokenizer import (
    byte_token_ids,
    character_byte_token_table,
    decode_gpt2_piece,
    encode_utf8_as_byte_tokens,
    gpt2_byte_encoder,
)


class TokenizerTests(unittest.TestCase):
    def test_byte_encoder_is_reversible_and_complete(self):
        encoder = gpt2_byte_encoder()
        self.assertEqual(len(encoder), 256)
        self.assertEqual(len(set(encoder.values())), 256)
        self.assertEqual(encoder[ord("A")], "A")

    def test_utf8_byte_tokens_round_trip_japanese(self):
        encoder = gpt2_byte_encoder()
        vocabulary = [encoder[index] for index in range(256)]
        ids = byte_token_ids(vocabulary)
        encoded = encode_utf8_as_byte_tokens("こんにちは", vocabulary)
        raw = bytes(ids.index(token_id) for token_id in encoded)
        self.assertEqual(raw.decode("utf-8"), "こんにちは")

    def test_decodes_byte_encoded_japanese_piece(self):
        encoder = gpt2_byte_encoder()
        piece = "".join(encoder[value] for value in "の".encode("utf-8"))
        self.assertEqual(decode_gpt2_piece(piece), "の")

    def test_character_table_prefers_a_direct_reversible_token(self):
        encoder = gpt2_byte_encoder()
        vocabulary = [encoder[index] for index in range(256)]
        vocabulary.append("".join(encoder[value] for value in "こ".encode("utf-8")))
        self.assertEqual(character_byte_token_table(vocabulary)["こ"], (256,))

    def test_generates_minecraft_only_chat_entrypoint(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = generate_kernel_pack(Path(directory) / "pack")
            configure_minecraft_input_runtime(paths, [6, 100], [7, 101])
            api = (paths.functions / "api" / "chat.mcfunction").read_text(encoding="utf-8")
            lookup = (paths.functions / "runtime" / "input" / "lookup.mcfunction").read_text(encoding="utf-8")
            load = (paths.functions / "load.mcfunction").read_text(encoding="utf-8")
        self.assertIn("$(message)", api)
        self.assertIn('chars."$(char)"', lookup)
        self.assertIn("runtime/input/load", load)


if __name__ == "__main__":
    unittest.main()
