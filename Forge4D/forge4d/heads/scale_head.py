# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from codes.Forge4D.Forge4D.forge4d.layers import Mlp
from codes.Forge4D.Forge4D.forge4d.layers.block import Block
from codes.Forge4D.Forge4D.forge4d.heads.head_act import activate_pose


class ScaleHead(nn.Module):
    """
    CameraHead predicts camera parameters from token representations using iterative refinement.

    It applies a series of transformer blocks (the "trunk") to dedicated camera tokens.
    """

    def __init__(
        self,
        dim_in: int = 2048,
        trunk_depth: int = 4,
        num_heads: int = 16,
        mlp_ratio: int = 4,
        init_values: float = 0.01,
        trans_act: str = "linear",
        quat_act: str = "linear",
        fl_act: str = "relu",  # Field of view activations: ensures FOV values are positive.
    ):
        super().__init__()

        self.target_dim = 1

        self.trans_act = trans_act
        self.quat_act = quat_act
        self.fl_act = fl_act
        self.trunk_depth = trunk_depth

        # Build the trunk using a sequence of transformer blocks.
        self.trunk = nn.Sequential(
            *[
                Block(
                    dim=dim_in//4,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    init_values=init_values,
                )
                for _ in range(trunk_depth)
            ]
        )

        self.token_norm = nn.LayerNorm(dim_in)
        self.trunk_norm = nn.LayerNorm(dim_in)

        self.empty_scale_tokens = nn.Parameter(torch.zeros(1, self.target_dim))
        self.embed_scale = nn.Linear(self.target_dim, dim_in)

        # Module for producing modulation parameters: shift, scale, and a gate.
        self.scaleLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim_in, 3 * dim_in, bias=True))

        # Adaptive layer normalization without affine parameters.
        self.adaln_norm = nn.LayerNorm(dim_in, elementwise_affine=False, eps=1e-6)
        self.scale_branch = Mlp(
            in_features=dim_in,
            hidden_features=dim_in // 2,
            out_features=self.target_dim,
            drop=0,
        )

    def forward(self, aggregated_tokens_list: list, patch_start_idx: int, num_iterations: int = 4) -> list:
        """
        Forward pass to predict camera parameters.

        Args:
            aggregated_tokens_list (list): List of token tensors from the network;
                the last tensor is used for prediction.
            num_iterations (int, optional): Number of iterative refinement steps. Defaults to 4.

        Returns:
            list: A list of predicted camera encodings (post-activation) from each iteration.
        """
        # Use tokens from the last block for camera prediction.
        tokens = aggregated_tokens_list[-1]

        # Extract the camera tokens
        scale_tokens = tokens[:, :, 0].reshape(tokens.shape[0], -1)
        scale_tokens = self.token_norm(scale_tokens)

        pred_scale_enc_list = self.trunk_fn(scale_tokens, num_iterations)
        return pred_scale_enc_list

    def trunk_fn(self, scale_tokens: torch.Tensor, num_iterations: int) -> list:
        B, D = scale_tokens.shape  # S is expected to be 1.
        pred_scale_enc = None
        pred_scale_enc_list = []

        for _ in range(num_iterations):
            if pred_scale_enc is None:
                module_input = self.embed_scale(self.empty_scale_tokens.expand(B, -1))
            else:
                # Detach the previous prediction to avoid backprop through time.
                pred_scale_enc = pred_scale_enc.detach()
                module_input = self.embed_scale(pred_scale_enc)

            # Generate modulation parameters and split them into shift, scale, and gate components.
            shift_msa, scale_msa, gate_msa = self.scaleLN_modulation(module_input).chunk(3, dim=-1)

            # Adaptive layer normalization and modulation.
            scale_tokens_modulated = gate_msa * modulate(self.adaln_norm(scale_tokens), shift_msa, scale_msa)
            scale_tokens_modulated = scale_tokens_modulated + scale_tokens

            scale_tokens_modulated = self.trunk(scale_tokens_modulated.reshape(B,4,D//4)).reshape(B,D)
            # Compute the delta update for the scale encoding.
            pred_scale_enc_delta = self.scale_branch(self.trunk_norm(scale_tokens_modulated))

            if pred_scale_enc is None:
                pred_scale_enc = pred_scale_enc_delta
            else:
                pred_scale_enc = pred_scale_enc + pred_scale_enc_delta

            # Apply final activation functions for translation, quaternion, and field-of-view.
            activated_scale = F.relu(pred_scale_enc)
            pred_scale_enc_list.append(activated_scale)

        return pred_scale_enc_list


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """
    Modulate the input tensor using scaling and shifting parameters.
    """
    # modified from https://github.com/facebookresearch/DiT/blob/796c29e532f47bba17c5b9c5eb39b9354b8b7c64/models.py#L19
    return x * (1 + scale) + shift
