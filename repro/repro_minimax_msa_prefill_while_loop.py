# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""Reproduce MiniMax MSA prefill query-loop capture without loading model weights."""

import argparse
from types import SimpleNamespace

import torch

from QEfficient.transformers.models.minimax_m3_vl import MiniMaxM3VLTextConfig
from QEfficient.transformers.models.minimax_m3_vl.modeling_minimax_m3_vl import QEffMiniMaxM3VLAttention


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence-length", type=int, default=32)
    parser.add_argument("--msa-q-chunk", type=int, default=8)
    args = parser.parse_args()

    config = MiniMaxM3VLTextConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=16,
        dense_intermediate_size=64,
        shared_intermediate_size=16,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=args.sequence_length,
        num_local_experts=4,
        num_experts_per_tok=2,
        routed_scaling_factor=1.0,
        layer_types=["full_attention", "minimax_m3_sparse"],
        mlp_layer_types=["dense", "sparse"],
        index_n_heads=2,
        index_head_dim=8,
        index_block_size=4,
        index_topk_blocks=2,
        index_local_blocks=1,
    )

    class PrefillAttentionModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.attention = QEffMiniMaxM3VLAttention(config, layer_idx=1).eval()
            self.blocking_config = SimpleNamespace(
                num_cores_per_device=4,
                msa_q_chunk=args.msa_q_chunk,
                msa_num_kv_blocks=1,
                num_kv_blocks=1,
                ctx_len=args.sequence_length,
                prefill_export_seq_len=args.sequence_length,
                prefill_compile_seq_len=None,
                msa_attn_dp=1,
                msa_attn_cp=1,
            )

        def forward(self, query, hidden, cos, sin, key, value, block_indices, block_valid, positions):
            return self.attention._msa_attention_prefill(
                query,
                hidden,
                cos,
                sin,
                key,
                value,
                block_indices,
                block_valid,
                positions,
                self.blocking_config,
            )

    batch = 1
    inputs = (
        torch.randn(batch, config.num_attention_heads, args.sequence_length, config.head_dim),
        torch.randn(batch, args.sequence_length, config.hidden_size),
        torch.ones(batch, args.sequence_length, config.head_dim),
        torch.zeros(batch, args.sequence_length, config.head_dim),
        torch.zeros(batch, config.num_key_value_heads, args.sequence_length, config.head_dim),
        torch.zeros(batch, config.num_key_value_heads, args.sequence_length, config.head_dim),
        torch.zeros(batch, config.num_key_value_heads, args.sequence_length, 2, dtype=torch.int32),
        torch.ones(batch, config.num_key_value_heads, args.sequence_length, 2, dtype=torch.bool),
        torch.arange(args.sequence_length).view(1, args.sequence_length),
    )
    module = PrefillAttentionModule()

    with torch.no_grad():
        expected = module(*inputs)
        exported = torch.export.export(module, inputs)
        actual = exported.module()(*inputs)

    loop_count = sum(node.target == torch.ops.higher_order.while_loop for node in exported.graph_module.graph.nodes)
    max_diff = max(torch.max(torch.abs(result - reference)).item() for result, reference in zip(actual, expected))
    print(f"captured while_loop nodes: {loop_count}")
    print(f"maximum eager/export difference: {max_diff}")
    if loop_count != 1 or max_diff != 0.0:
        raise RuntimeError("MiniMax MSA prefill query loop was not captured correctly.")


if __name__ == "__main__":
    main()
