import unittest
from unittest.mock import patch

from llmcf.chat import (
    appended_chat_prompt,
    chat_command,
    initial_chat_prompt,
    minecraft_prompt_command,
)


class ChatTests(unittest.TestCase):
    def test_initial_chat_template(self):
        prompt = initial_chat_prompt("Hello", "Be concise")
        self.assertEqual(
            prompt,
            "<|im_start|>system\nBe concise<|im_end|>\n"
            "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n",
        )

    def test_append_starts_after_previous_im_end(self):
        self.assertEqual(
            appended_chat_prompt("Again"),
            "\n<|im_start|>user\nAgain<|im_end|>\n<|im_start|>assistant\n",
        )

    def test_minecraft_prompt_command(self):
        self.assertEqual(
            minecraft_prompt_command([6, 6423, 708], 4),
            "function lfm:api/start_prompt {tokens:[6,6423,708],max_tokens:4}",
        )
        self.assertIn("append_prompt", minecraft_prompt_command([708], 2, append=True))

    @patch("llmcf.chat.tokenize_via_llama_server", return_value=[1, 6, 6423, 708])
    def test_chat_command_removes_bos_for_datapack_prefill(self, tokenize):
        command, tokens = chat_command("Hello", 3)
        self.assertEqual(tokens, [6, 6423, 708])
        self.assertIn("tokens:[6,6423,708]", command)
        tokenize.assert_called_once()


if __name__ == "__main__":
    unittest.main()
