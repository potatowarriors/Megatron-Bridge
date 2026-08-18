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

import logging
import os

from megatron.bridge.models.alpha.alpha_bridge import AlphaBridge


logger = logging.getLogger(__name__)


def _maybe_enable_flashqla() -> None:
    """Opt-in swap of the GDN chunked kernel to FlashQLA (ALPHA_GDN_BACKEND=flashqla).

    FlashQLA's chunk_gated_delta_rule is signature-compatible with fla's and
    verified numerically equivalent on alpha's geometry (fwd/bwd cos>=0.9969,
    max|d|<=1.2e-4 bf16) while 1.6-3.7x faster fwd+bwd (growing with seqlen).
    mcore binds the kernel by name at GDN.__init__, so patching the module
    namespaces here (imported before any model construction) is sufficient.
    Kept out-of-tree because Megatron-LM sources must not be modified in this
    repo; failure to patch is loud, never silent.
    """
    if os.environ.get("ALPHA_GDN_BACKEND", "").lower() != "flashqla":
        return
    try:
        from flash_qla import chunk_gated_delta_rule as _qla_impl

        import megatron.core.ssm.gated_delta_net.common as _gdn_common
        import megatron.core.ssm.gated_delta_net.gdn as _gdn_mod

        if _gdn_common.chunk_gated_delta_rule is None:
            raise RuntimeError("fla import failed upstream; refusing to patch a broken module")
        _gdn_common.chunk_gated_delta_rule = _qla_impl
        _gdn_mod.chunk_gated_delta_rule = _qla_impl
        logger.info("alpha: GDN chunked kernel swapped to FlashQLA (ALPHA_GDN_BACKEND=flashqla)")
    except Exception as e:  # noqa: BLE001 - must never silently fall back
        raise RuntimeError(
            "ALPHA_GDN_BACKEND=flashqla was requested but the FlashQLA swap failed. "
            "Unset the env var to use the default fla kernel."
        ) from e


_maybe_enable_flashqla()

__all__ = ["AlphaBridge"]
