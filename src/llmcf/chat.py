from __future__ import annotations

import json
from typing import Sequence
from urllib.request import Request, urlopen


BOS_TOKEN_ID = 1


def initial_chat_prompt(message: str, system: str | None = None) -> str:
    parts: list[str] = []
    if system:
        parts.append(f"<|im_start|>system\n{system}<|im_end|>\n")
    parts.append(f"<|im_start|>user\n{message}<|im_end|>\n<|im_start|>assistant\n")
    return "".join(parts)


def appended_chat_prompt(message: str) -> str:
    return f"\n<|im_start|>user\n{message}<|im_end|>\n<|im_start|>assistant\n"


def tokenize_via_llama_server(
    content: str,
    url: str = "http://127.0.0.1:8081/tokenize",
    add_special: bool = True,
    timeout: float = 30.0,
) -> list[int]:
    payload = json.dumps(
        {"content": content, "add_special": add_special, "with_pieces": False}
    ).encode("utf-8")
    request = Request(url, data=payload, headers={"Content-Type": "application/json"})
    with urlopen(request, timeout=timeout) as response:
        body = json.load(response)
    tokens = body.get("tokens")
    if not isinstance(tokens, list) or not all(isinstance(token, int) for token in tokens):
        raise ValueError("llama-server returned an invalid token list")
    return tokens


def minecraft_prompt_command(
    token_ids: Sequence[int],
    max_tokens: int,
    append: bool = False,
) -> str:
    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive")
    if not all(0 <= token_id < 65536 for token_id in token_ids):
        raise ValueError("token IDs must be in the LFM2 vocabulary")
    function = "append_prompt" if append else "start_prompt"
    tokens = json.dumps(list(token_ids), separators=(",", ":"))
    return f"function lfm:api/{function} {{tokens:{tokens},max_tokens:{max_tokens}}}"


def chat_command(
    message: str,
    max_tokens: int,
    system: str | None = None,
    append: bool = False,
    tokenizer_url: str = "http://127.0.0.1:8081/tokenize",
) -> tuple[str, list[int]]:
    prompt = appended_chat_prompt(message) if append else initial_chat_prompt(message, system)
    tokens = tokenize_via_llama_server(prompt, tokenizer_url, add_special=not append)
    if not append:
        if not tokens or tokens[0] != BOS_TOKEN_ID:
            raise ValueError("tokenizer did not prepend the expected BOS token")
        tokens = tokens[1:]
    return minecraft_prompt_command(tokens, max_tokens, append=append), tokens
