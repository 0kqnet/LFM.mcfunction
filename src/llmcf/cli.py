from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .compiler import compile_bos_pack, compile_model, compile_tokenizer_assets
from .chat import chat_command
from .generator import generate_kernel_pack
from .gguf import file_sha256, parse_gguf, write_manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="llmcf")
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser("inspect", help="inspect a GGUF file")
    inspect_parser.add_argument("gguf", type=Path)
    inspect_parser.add_argument("--manifest", type=Path)
    inspect_parser.add_argument("--sha256", action="store_true")

    pack_parser = subparsers.add_parser("generate-kernel-pack", help="generate the Q2_K kernel proof datapack")
    pack_parser.add_argument("destination", type=Path)
    pack_parser.add_argument("--clean", action="store_true")

    compile_parser = subparsers.add_parser("compile-model", help="compile the locked GGUF into a datapack")
    compile_parser.add_argument("gguf", type=Path)
    compile_parser.add_argument("destination", type=Path)
    compile_parser.add_argument("--lock", type=Path, default=Path("model.lock.json"))
    compile_parser.add_argument("--blocks-per-chunk", type=int, default=2048)

    bos_parser = subparsers.add_parser("compile-bos-pack", help="compile the real BOS embedding into a small validation datapack")
    bos_parser.add_argument("gguf", type=Path)
    bos_parser.add_argument("destination", type=Path)
    bos_parser.add_argument("--lock", type=Path, default=Path("model.lock.json"))

    chat_parser = subparsers.add_parser("chat-command", help="tokenize a chat turn and print its Minecraft prefill command")
    chat_parser.add_argument("message")
    chat_parser.add_argument("--system")
    chat_parser.add_argument("--max-tokens", type=int, default=32)
    chat_parser.add_argument("--append", action="store_true")
    chat_parser.add_argument("--tokenizer-url", default="http://127.0.0.1:8081/tokenize")

    input_parser = subparsers.add_parser(
        "compile-tokenizer-assets",
        help="add the in-Minecraft UTF-8 byte tokenizer to an existing datapack",
    )
    input_parser.add_argument("gguf", type=Path)
    input_parser.add_argument("destination", type=Path)
    input_parser.add_argument("--lock", type=Path, default=Path("model.lock.json"))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "inspect":
        model = parse_gguf(args.gguf)
        if args.manifest:
            write_manifest(model, args.manifest)
        summary = {
            "version": model.version,
            "tensor_count": len(model.tensors),
            "data_offset": model.data_offset,
            "architecture": model.metadata.get("general.architecture"),
            "name": model.metadata.get("general.name"),
            "types": _type_counts(model),
        }
        if args.sha256:
            summary["sha256"] = file_sha256(args.gguf)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0
    if args.command == "generate-kernel-pack":
        paths = generate_kernel_pack(args.destination, clean=args.clean)
        print(paths.root.resolve())
        return 0
    if args.command == "compile-model":
        compile_model(args.gguf, args.destination, args.lock, args.blocks_per_chunk)
        return 0
    if args.command == "compile-bos-pack":
        proof = compile_bos_pack(args.gguf, args.destination, args.lock)
        print(json.dumps(proof, indent=2))
        return 0
    if args.command == "chat-command":
        command, tokens = chat_command(
            args.message,
            args.max_tokens,
            system=args.system,
            append=args.append,
            tokenizer_url=args.tokenizer_url,
        )
        print(json.dumps({"token_count": len(tokens), "tokens": tokens}, ensure_ascii=False))
        print(command)
        return 0
    if args.command == "compile-tokenizer-assets":
        result = compile_tokenizer_assets(args.gguf, args.destination, args.lock)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    raise AssertionError(args.command)


def _type_counts(model) -> dict[str, int]:
    counts: dict[str, int] = {}
    for tensor in model.tensors:
        counts[tensor.type_name] = counts.get(tensor.type_name, 0) + 1
    return dict(sorted(counts.items()))


if __name__ == "__main__":
    sys.exit(main())
