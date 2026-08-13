# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
from megatron.core.models.gpt.experimental_attention_variant_module_specs import (
    get_transformer_block_with_experimental_attention_variant_spec,
)
from megatron.core.models.gpt.gpt_model import GPTModel

from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
from megatron.bridge.models.conversion.param_mapping import (
    AutoMapping,
    GatedMLPMapping,
    GDNConv1dMapping,
    GDNLinearMapping,
    QKVMapping,
    ReplicatedMapping,
)
from megatron.bridge.models.conversion.transformers_compat import full_attention_interval_from_hf


@MegatronModelBridge.register_bridge(source="AlphaForCausalLM", target=GPTModel, model_type="alpha")
class AlphaBridge(MegatronModelBridge):
    """
    Megatron Bridge for the alpha model family (``AlphaForCausalLM``).

    Alpha is a Qwen3-Next-style hybrid architecture — gated delta net linear
    attention interleaved with gated full attention — combined with
    DeepSeek-V3-style MoE routing (sigmoid scores, group-limited top-k,
    fp32 expert score-correction bias) and a gated shared expert.

    Alpha is distributed as a trust-remote-code HuggingFace checkpoint
    (``model_type: "alpha"`` with ``auto_map``), so this bridge registers by
    architecture-name string rather than by transformers class.

    Differences from :class:`~megatron.bridge.models.qwen.qwen3_next_bridge.Qwen3NextBridge`
    that this bridge encodes (getting these wrong passes weight validation but
    silently corrupts the forward pass):

    - **Standard RMSNorm everywhere.** Alpha does NOT use zero-centered gamma.
      ``layernorm_zero_centered_gamma`` stays False and the GDN out-norm maps as
      a direct copy (no ``RMSNorm2ZeroCenteredRMSNormMapping``).
    - **DeepSeek-V3 routing.** ``scoring_func=sigmoid`` with
      ``n_group``/``topk_group`` group-limited selection,
      ``routed_scaling_factor`` weight scaling and an fp32
      ``e_score_correction_bias`` on the gate (mapped to
      ``mlp.router.expert_bias``).
    - No MTP layers.

    Example:
        >>> from megatron.bridge import AutoBridge
        >>> bridge = AutoBridge.from_hf_pretrained("/path/to/hfmodel_00NNNNN", trust_remote_code=True)
        >>> provider = bridge.to_megatron_provider()
    """

    def provider_bridge(self, hf_pretrained):
        """Convert HuggingFace Alpha config to a GPTModelProvider."""
        provider = super().provider_bridge(hf_pretrained)
        hf_config = hf_pretrained.config

        # Standard GPT settings
        provider.normalization = "RMSNorm"
        provider.gated_linear_unit = True
        provider.position_embedding_type = "rope"
        provider.add_bias_linear = False
        provider.add_qkv_bias = False
        provider.hidden_dropout = 0.0
        provider.attention_dropout = 0.0
        provider.qk_layernorm = True
        provider.autocast_dtype = torch.bfloat16
        provider.share_embeddings_and_output_weights = getattr(hf_config, "tie_word_embeddings", False)

        # Alpha uses STANDARD RMSNorm (weight multiplies directly, ones-init).
        # Qwen3-Next-style zero-centered gamma here corrupts every norm output
        # while all weight-level validation still passes.
        provider.layernorm_zero_centered_gamma = False

        # MoE with DeepSeek-V3-style routing. scoring_func / n_group /
        # topk_group / routed_scaling_factor flow in via CONFIG_MAPPING; the
        # flags below make the remaining routing semantics explicit.
        provider.moe_grouped_gemm = True
        provider.moe_router_dtype = "fp32"
        provider.moe_router_score_function = "sigmoid"
        provider.moe_router_enable_expert_bias = True
        # Conversion-time default: bias frozen. Training configs may override.
        provider.moe_router_bias_update_rate = 0.0
        provider.moe_router_pre_softmax = False
        provider.moe_router_load_balancing_type = "none"
        provider.moe_aux_loss_coeff = 0.0
        provider.moe_token_dispatcher_type = "alltoall"
        provider.moe_permute_fusion = True
        provider.moe_shared_expert_gate = True
        provider.moe_shared_expert_intermediate_size = hf_config.shared_expert_intermediate_size

        # Gated attention: q_proj carries a fused per-head output gate.
        provider.attention_output_gate = True

        # Hybrid gated-delta-net + full attention stack.
        provider.transformer_layer_spec = get_transformer_block_with_experimental_attention_variant_spec
        provider.experimental_attention_variant = "gated_delta_net"
        provider.linear_attention_freq = full_attention_interval_from_hf(hf_config)
        provider.linear_conv_kernel_dim = hf_config.linear_conv_kernel_dim
        provider.linear_key_head_dim = hf_config.linear_key_head_dim
        provider.linear_value_head_dim = hf_config.linear_value_head_dim
        provider.linear_num_key_heads = hf_config.linear_num_key_heads
        provider.linear_num_value_heads = hf_config.linear_num_value_heads

        # Heterogeneous checkpointing for mixed attention layer types
        provider.hetereogenous_dist_checkpoint = True

        return provider

    def mapping_registry(self) -> MegatronMappingRegistry:
        """Parameter mappings between Megatron and Alpha HF format."""
        param_mappings = {
            # Embedding and output (untied in alpha)
            "embedding.word_embeddings.weight": "model.embed_tokens.weight",
            "output_layer.weight": "lm_head.weight",
            "decoder.final_layernorm.weight": "model.norm.weight",
            # MoE router (DSV3: fp32 expert bias lives on the HF gate module)
            "decoder.layers.*.mlp.router.weight": "model.layers.*.mlp.gate.weight",
            "decoder.layers.*.mlp.router.expert_bias": "model.layers.*.mlp.gate.e_score_correction_bias",
            "decoder.layers.*.pre_mlp_layernorm.weight": "model.layers.*.post_attention_layernorm.weight",
            # Standard (gated) attention
            "decoder.layers.*.self_attention.linear_qkv.layer_norm_weight": "model.layers.*.input_layernorm.weight",
            "decoder.layers.*.self_attention.q_layernorm.weight": "model.layers.*.self_attn.q_norm.weight",
            "decoder.layers.*.self_attention.k_layernorm.weight": "model.layers.*.self_attn.k_norm.weight",
            "decoder.layers.*.self_attention.linear_proj.weight": "model.layers.*.self_attn.o_proj.weight",
            # Linear attention (GDN)
            "decoder.layers.*.self_attention.in_proj.layer_norm_weight": "model.layers.*.input_layernorm.weight",
            "decoder.layers.*.self_attention.out_proj.weight": "model.layers.*.linear_attn.out_proj.weight",
            "decoder.layers.*.self_attention.A_log": "model.layers.*.linear_attn.A_log",
            "decoder.layers.*.self_attention.dt_bias": "model.layers.*.linear_attn.dt_bias",
            # Alpha's GDN out-norm is a standard RMSNorm like every other alpha
            # norm — direct copy (Qwen3-Next needs a -1 shift here; alpha must NOT).
            "decoder.layers.*.self_attention.out_norm.weight": "model.layers.*.linear_attn.norm.weight",
        }

        mapping_list = []
        for megatron_param, hf_param in param_mappings.items():
            mapping_list.append(AutoMapping(megatron_param=megatron_param, hf_param=hf_param))
        AutoMapping.register_module_type("SharedExpertMLP", "column")
        AutoMapping.register_module_type("GatedDeltaNet", "column")

        mapping_list.extend(
            [
                # QKV: alpha q_proj is 2x width (per-head fused [q, gate]);
                # handled by the attention_output_gate-aware QKV merge.
                QKVMapping(
                    megatron_param="decoder.layers.*.self_attention.linear_qkv.weight",
                    q="model.layers.*.self_attn.q_proj.weight",
                    k="model.layers.*.self_attn.k_proj.weight",
                    v="model.layers.*.self_attn.v_proj.weight",
                ),
                # GDN conv over [Q, K, V] channels
                GDNConv1dMapping(
                    megatron_param="decoder.layers.*.self_attention.conv1d.weight",
                    hf_param="model.layers.*.linear_attn.conv1d.weight",
                ),
                # GDN fused in_proj from HF (qkvz, ba) pair
                GDNLinearMapping(
                    megatron_param="decoder.layers.*.self_attention.in_proj.weight",
                    qkvz="model.layers.*.linear_attn.in_proj_qkvz.weight",
                    ba="model.layers.*.linear_attn.in_proj_ba.weight",
                ),
                # Routed experts (grouped GEMM layout)
                GatedMLPMapping(
                    megatron_param="decoder.layers.*.mlp.experts.linear_fc1.weight*",
                    gate="model.layers.*.mlp.experts.*.gate_proj.weight",
                    up="model.layers.*.mlp.experts.*.up_proj.weight",
                ),
                AutoMapping(
                    megatron_param="decoder.layers.*.mlp.experts.linear_fc2.weight*",
                    hf_param="model.layers.*.mlp.experts.*.down_proj.weight",
                ),
                # Routed experts (sequential layout, e.g. ModelOpt pruning)
                GatedMLPMapping(
                    megatron_param="decoder.layers.*.mlp.experts.local_experts.*.linear_fc1.weight",
                    gate="model.layers.*.mlp.experts.*.gate_proj.weight",
                    up="model.layers.*.mlp.experts.*.up_proj.weight",
                ),
                AutoMapping(
                    megatron_param="decoder.layers.*.mlp.experts.local_experts.*.linear_fc2.weight",
                    hf_param="model.layers.*.mlp.experts.*.down_proj.weight",
                ),
                # Shared expert
                GatedMLPMapping(
                    megatron_param="decoder.layers.*.mlp.shared_experts.linear_fc1.weight",
                    gate="model.layers.*.mlp.shared_expert.gate_proj.weight",
                    up="model.layers.*.mlp.shared_expert.up_proj.weight",
                ),
                AutoMapping(
                    megatron_param="decoder.layers.*.mlp.shared_experts.linear_fc2.weight",
                    hf_param="model.layers.*.mlp.shared_expert.down_proj.weight",
                ),
                # Shared expert gate (scalar gate per token)
                ReplicatedMapping(
                    megatron_param="decoder.layers.*.mlp.shared_experts.gate_weight",
                    hf_param="model.layers.*.mlp.shared_expert_gate.weight",
                ),
            ]
        )

        return MegatronMappingRegistry(*mapping_list)
