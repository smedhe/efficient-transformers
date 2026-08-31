# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""
Weight-free continuous-batching export tests.

All tests run with weight_free=True (set on the QEff model) and use_onnx_subfunctions=True.
CPU-only. No QAIC hardware required.
"""

from __future__ import annotations

import pytest

from .._helpers import (
    WEIGHT_FREE_CAUSAL_LM_MODEL_IDS,
    assert_has_subfunctions,
    assert_retained_state_outputs,
    build_meta_qeff_model,
    exported_onnx_path,
)


@pytest.mark.weight_free
@pytest.mark.weight_free_export
@pytest.mark.parametrize(
    "model_type,model_id",
    sorted(WEIGHT_FREE_CAUSAL_LM_MODEL_IDS.items()),
    ids=sorted(WEIGHT_FREE_CAUSAL_LM_MODEL_IDS),
)
def test_weight_free_cb_export(model_type, model_id, tmp_export_dir):
    """continuous_batching=True + weight_free=True + use_onnx_subfunctions=True.

    Validates ONNX structure only (no ORT parity — CB ORT inference requires
    batch_index routing which adds significant test complexity).
    """
    qeff_model = build_meta_qeff_model(
        model_id,
        checkpoint_dir=tmp_export_dir / "checkpoint",
        continuous_batching=True,
    )
    onnx_path = exported_onnx_path(
        qeff_model.export(
            tmp_export_dir,
            use_onnx_subfunctions=True,
            offload_pt_weights=False,
        )
    )

    num_layers = qeff_model.model.config.num_hidden_layers
    assert_retained_state_outputs(onnx_path, expected_count=2 * num_layers)
    assert_has_subfunctions(onnx_path, qeff_model)
