#!/usr/bin/env python3
"""Encode a prompt to token IDs and write them to a binary file.
Usage: python encode_prompt.py "Your prompt here" > prompt_tokens.bin
       python encode_prompt.py --text "Your prompt" --output prompt_tokens.bin
       python encode_prompt.py --input-file prompt.txt --output prompt_tokens.bin
"""
import sys
import struct
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('text', nargs='?', help='Prompt text')
    parser.add_argument('--text', '-t', dest='text_flag', help='Prompt text (flag form)')
    parser.add_argument('--input-file', help='Read prompt text from a file')
    parser.add_argument('--output', '-o', default='prompt_tokens.bin')
    parser.add_argument('--model', default=None)
    args = parser.parse_args()

    text = args.text or args.text_flag
    if args.input_file:
        with open(args.input_file, 'r', encoding='utf-8') as f:
            text = f.read()
    if not text:
        print("Usage: encode_prompt.py \"Your prompt here\"", file=sys.stderr)
        sys.exit(1)

    model_path = args.model or (
        '/Users/danielwoods/.cache/huggingface/hub/'
        'models--mlx-community--Qwen3.5-397B-A17B-4bit/'
        'snapshots/39159bd8aa74f5c8446d2b2dc584f62bb51cb0d3'
    )

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    ids = tokenizer.encode(text)

    print(f"Prompt: {repr(text)}", file=sys.stderr)
    print(f"Tokens ({len(ids)}): {ids}", file=sys.stderr)

    with open(args.output, 'wb') as f:
        # Header: uint32 count
        f.write(struct.pack('<I', len(ids)))
        # Token IDs: uint32 each
        for tid in ids:
            f.write(struct.pack('<I', tid))

    print(f"Written to {args.output}", file=sys.stderr)

if __name__ == '__main__':
    main()
