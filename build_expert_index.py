#!/usr/bin/env python3
"""Build expert_index.json from a model's safetensors index.

Scans model.safetensors.index.json to find all expert (switch_mlp) weight tensors
and computes their byte offsets and strides for the repacking script.

Usage:
    python build_expert_index.py --model ~/.cache/huggingface/hub/models--mlx-community--Qwen3.5-35B-A3B-4bit
"""

import argparse
import json
import os
import struct
import sys
from pathlib import Path
from collections import defaultdict


def parse_safetensors_header(filepath):
    """Parse a safetensors file header. Returns (header_dict, data_start_offset)."""
    with open(filepath, 'rb') as f:
        header_len = struct.unpack('<Q', f.read(8))[0]
        header = json.loads(f.read(header_len))
        data_start = 8 + header_len
    return header, data_start


def main():
    parser = argparse.ArgumentParser(description='Build expert_index.json from safetensors')
    parser.add_argument('--model', type=str,
                        default=os.path.expanduser(
                            '~/.cache/huggingface/hub/models--mlx-community--Qwen3.5-35B-A3B-4bit'),
                        help='Path to model directory')
    parser.add_argument('--output', type=str, default='expert_index.json',
                        help='Output path for expert_index.json')
    args = parser.parse_args()

    model_path = Path(args.model)

    # Find the snapshot directory if it exists (HF cache layout)
    # Check for model.safetensors.index.json in the model path or its snapshots
    index_path = model_path / 'model.safetensors.index.json'
    if not index_path.exists():
        # Try snapshot subdirectories
        snapshots_dir = model_path / 'snapshots'
        if snapshots_dir.exists():
            for snap in sorted(snapshots_dir.iterdir()):
                candidate = snap / 'model.safetensors.index.json'
                if candidate.exists():
                    model_path = snap
                    index_path = candidate
                    break

    if not index_path.exists():
        print(f"ERROR: {index_path} not found", file=sys.stderr)
        sys.exit(1)

    print(f"Model path: {model_path}")
    print(f"Index: {index_path}")

    with open(index_path) as f:
        idx = json.load(f)

    weight_map = idx['weight_map']

    # Find all expert tensor names: pattern is
    #   model.layers.{L}.switch_mlp.{gate_proj|up_proj|down_proj}.{weight|scales|biases}
    # or with language_model. prefix
    import re
    expert_pattern = re.compile(
        r'(?:language_model\.)?model\.layers\.(\d+)\.(?:mlp\.)?switch_mlp\.'
        r'(gate_proj|up_proj|down_proj)\.(weight|scales|biases)$'
    )

    # Group expert tensors by layer and component
    layer_tensors = defaultdict(dict)  # layer_idx -> {component_key -> tensor_name}
    for name in weight_map:
        m = expert_pattern.match(name)
        if m:
            layer_idx = int(m.group(1))
            proj = m.group(2)  # gate_proj, up_proj, down_proj
            part = m.group(3)  # weight, scales, biases
            component_key = f"{proj}.{part}"
            layer_tensors[layer_idx][component_key] = name

    if not layer_tensors:
        print("ERROR: No expert tensors found in weight_map", file=sys.stderr)
        print("Sample tensor names:", list(weight_map.keys())[:20], file=sys.stderr)
        sys.exit(1)

    num_layers = len(layer_tensors)
    print(f"Found expert tensors in {num_layers} layers")

    # Parse safetensors headers to get exact offsets
    header_cache = {}
    expert_reads = {}

    for layer_idx in sorted(layer_tensors.keys()):
        components = layer_tensors[layer_idx]
        layer_reads = {}

        for comp_key, tensor_name in sorted(components.items()):
            shard_file = weight_map[tensor_name]
            filepath = model_path / shard_file

            if shard_file not in header_cache:
                header_cache[shard_file] = parse_safetensors_header(str(filepath))

            header, data_start = header_cache[shard_file]

            if tensor_name not in header:
                # Try without language_model prefix
                alt_name = tensor_name
                if alt_name.startswith("language_model."):
                    alt_name = alt_name[len("language_model."):]
                if alt_name not in header:
                    print(f"WARNING: {tensor_name} not in {shard_file} header, skipping")
                    continue
                tensor_name_in_header = alt_name
            else:
                tensor_name_in_header = tensor_name

            meta = header[tensor_name_in_header]
            offsets = meta['data_offsets']
            shape = meta['shape']
            total_size = offsets[1] - offsets[0]
            abs_offset = data_start + offsets[0]

            # For expert tensors, the first dimension is num_experts (256)
            # Expert stride = total_size / num_experts
            num_experts = shape[0]
            expert_size = total_size // num_experts

            layer_reads[comp_key] = {
                "file": shard_file,
                "abs_offset": abs_offset,
                "expert_stride": expert_size,
                "expert_size": expert_size,
                "total_size": total_size,
                "shape": shape,
                "dtype": meta['dtype'],
            }

        expert_reads[str(layer_idx)] = layer_reads
        if layer_idx == 0:
            print(f"\nLayer 0 components:")
            for k, v in sorted(layer_reads.items()):
                print(f"  {k}: shape={v['shape']}, expert_size={v['expert_size']}, "
                      f"file={v['file']}")

    # Build output
    output = {
        "model_path": str(model_path),
        "expert_reads": expert_reads,
    }

    with open(args.output, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nWrote {args.output}")
    print(f"Layers: {num_layers}")
    print(f"Components per layer: {len(next(iter(expert_reads.values())))}")


if __name__ == '__main__':
    main()
