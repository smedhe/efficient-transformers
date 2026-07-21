# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import torch
from torch import nn

from QEfficient.customop.rms_norm_opset17 import CustomRMSNormFunc  # noqa: F401 — re-exported for callers
from QEfficient.customop.utils import select_interface


class CustomRMSNormAIC(nn.Module):
    """
    RMSNorm module that works by replacing the current module with compiler known custom-op.
    """

    def __init__(self, hidden_size, eps=1e-05):
        super(CustomRMSNormAIC, self).__init__()
        self.variance_epsilon = eps
        self.eps = eps  # Added to support GemmaRMSNorm
        self.weight = torch.nn.Parameter(torch.ones(hidden_size))

    def forward(self, hidden_states):
        rms_interface = select_interface(CustomRMSNormFunc.apply, torch.ops.qefficient.rms_norm)
        return rms_interface(
            hidden_states, self.weight, self.variance_epsilon if hasattr(self, "variance_epsilon") else self.eps
        )


class GemmaCustomRMSNormAIC(CustomRMSNormAIC):
    """
    Modify the init function to add +1 to the weights
    """

    def __qeff_init__(self):
        with torch.no_grad():
            self.weight.copy_(self.weight + 1.0)
