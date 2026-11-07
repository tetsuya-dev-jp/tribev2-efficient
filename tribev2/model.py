# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import typing as tp

import torch
from einops import rearrange
from neuralset.dataloader import SegmentData
from neuraltrain.models.base import BaseModelConfig
from neuraltrain.models.common import Mlp, SubjectLayers, SubjectLayersModel
from neuraltrain.models.transformer import TransformerEncoder
from torch import nn

logger = logging.getLogger(__name__)


class TemporalSmoothing(BaseModelConfig):
    kernel_size: int = 9
    sigma: float | None = None

    def build(self, dim: int) -> nn.Module:

        def gaussian_kernel_1d(kernel_size: int, sigma: float):
            x = torch.arange(kernel_size) - kernel_size // 2
            kernel = torch.exp(-0.5 * (x / sigma) ** 2)
            kernel = kernel / kernel.sum()
            return kernel.view(1, 1, -1)

        conv = nn.Conv1d(
            dim,
            dim,
            kernel_size=self.kernel_size,
            padding=self.kernel_size // 2,
            bias=False,
            groups=dim,
        )
        if self.sigma is not None:
            kernel = gaussian_kernel_1d(kernel_size=self.kernel_size, sigma=self.sigma)
            kernel = kernel.repeat(dim, 1, 1)
            conv.weight.data = kernel
            conv.requires_grad = False
        return conv


class FmriEncoder(BaseModelConfig):

    # architecture
    projector: BaseModelConfig = Mlp(norm_layer="layer", activation_layer="gelu")
    combiner: Mlp | None = Mlp(norm_layer="layer", activation_layer="gelu")
    encoder: TransformerEncoder | None = TransformerEncoder()
    # other hyperparameters
    time_pos_embedding: bool = True
    subject_embedding: bool = False
    subject_layers: SubjectLayers | None = SubjectLayers()
    hidden: int = 256
    max_seq_len: int = 1024
    dropout: float = 0.0
    extractor_aggregation: tp.Literal["stack", "sum", "cat"] = "cat"
    layer_aggregation: tp.Literal["mean", "cat"] = "cat"
    linear_baseline: bool = False
    modality_dropout: float = 0.0
    temporal_dropout: float = 0.0
    low_rank_head: int | None = None
    temporal_smoothing: TemporalSmoothing | None = None

    def model_post_init(self, __context):
        if self.encoder is not None:
            for key in ["attn_dropout", "ff_dropout", "layer_dropout"]:
                setattr(self.encoder, key, self.dropout)
        if hasattr(self.projector, "dropout"):
            self.projector.dropout = self.dropout
        return super().model_post_init(__context)

    def build(
        self, feature_dims: dict[int], n_outputs: int, n_output_timesteps: int
    ) -> nn.Module:
        return FmriEncoderModel(
            feature_dims,
            n_outputs,
            n_output_timesteps,
            config=self,
        )


class FmriEncoderModel(nn.Module):

    def __init__(
        self,
        feature_dims: dict[str, tuple[int, int]],
        n_outputs: int,
        n_output_timesteps: int,
        config: FmriEncoder,
    ):
        super().__init__()
        self.config = config
        self.feature_dims = feature_dims
        self.n_outputs = n_outputs
        self.n_output_timesteps = n_output_timesteps
        self._inference_optimized = False
        self.projectors = nn.ModuleDict()
        self.pooler = nn.AdaptiveAvgPool1d(n_output_timesteps)
        hidden = config.hidden
        for modality, tup in feature_dims.items():
            if tup is None:
                logger.warning(
                    "%s has no feature dimensions. Skipping projector.", modality
                )
                continue
            else:
                num_layers, feature_dim = tup
            input_dim = (
                feature_dim * num_layers
                if config.layer_aggregation == "cat"
                else feature_dim
            )
            output_dim = (
                hidden // len(feature_dims)
                if config.extractor_aggregation == "cat"
                else hidden
            )
            self.projectors[modality] = self.config.projector.build(
                input_dim, output_dim
            )
        input_dim = (
            (hidden // len(feature_dims)) * len(feature_dims)
            if config.extractor_aggregation == "cat"
            else hidden
        )
        if self.config.combiner is not None:
            self.combiner = self.config.combiner.build(input_dim, hidden)
        else:
            assert (
                hidden % len(feature_dims) == 0
            ), "hidden must be divisible by the number of modalities if there is no combiner"
            self.combiner = nn.Identity()
        if config.low_rank_head is not None:
            self.low_rank_head = nn.Linear(hidden, config.low_rank_head, bias=False)
            bottleneck = config.low_rank_head
        else:
            bottleneck = hidden
        self.predictor = config.subject_layers.build(
            in_channels=bottleneck,
            out_channels=n_outputs,
        )
        if config.temporal_smoothing is not None:
            self.temporal_smoothing = config.temporal_smoothing.build(dim=hidden)
        if not config.linear_baseline:
            if config.time_pos_embedding:
                self.time_pos_embed = nn.Parameter(
                    torch.randn(1, config.max_seq_len, hidden)
                )
            if config.subject_embedding:
                self.subject_embed = nn.Embedding(config.n_subjects, hidden)
            self.encoder = config.encoder.build(dim=hidden)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def forward(self, batch: SegmentData, pool_outputs: bool = True) -> torch.Tensor:
        x, subject_id = self.encode(batch)  # B, T, H
        x = x.transpose(1, 2)  # B, H, T
        if self._inference_optimized:
            # The fused readout is a per-timestep affine map, so pooling the
            # hidden states before it is equivalent to pooling its outputs.
            if pool_outputs:
                x = self.pooler(x)
            return self.predictor(x, subject_id)
        if self.config.low_rank_head is not None:
            x = self.low_rank_head(x.transpose(1, 2)).transpose(1, 2)
        x = self.predictor(x, subject_id)  # B, O, T
        if pool_outputs:
            out = self.pooler(x)  # B, O, T'
        else:
            out = x
        return out

    def encode(self, batch: SegmentData) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Run the feature projectors and optional transformer up to hidden states."""
        x = self.aggregate_features(batch)  # B, T, H
        subject_id = batch.data.get("subject_id", None)
        if hasattr(self, "temporal_smoothing"):
            x = self.temporal_smoothing(x.transpose(1, 2)).transpose(1, 2)
        if not self.config.linear_baseline:
            x = self.transformer_forward(x, subject_id)
        return x, subject_id

    def optimize_for_inference(self) -> "FmriEncoderModel":
        """Fuse the average-subject readout for inference.

        The low-rank head and the shared average-subject affine head are
        composed into a single linear map, so the intermediate low-rank
        activations are no longer materialized.  Because that map is applied
        per timestep, the hidden states can be pooled before the readout.

        This is an explicit, one-way conversion: the model must be in eval
        mode, the predictor must use the average-subject head, and training is
        not supported afterwards.  The original forward path and state dict
        are left untouched unless this method is called.

        The converted state dict has different keys and shapes from the
        original checkpoint and cannot be loaded with the original config.
        To obtain a converted model later, reload the original checkpoint and
        call this method again.
        """
        if self._inference_optimized:
            return self
        if self.training:
            raise RuntimeError(
                "optimize_for_inference() requires the model to be in eval mode."
            )
        predictor = self.predictor
        if not isinstance(predictor, SubjectLayersModel):
            raise TypeError(
                "optimize_for_inference() requires a SubjectLayersModel predictor."
            )
        if not predictor.average_subjects:
            raise RuntimeError(
                "optimize_for_inference() requires an average-subject predictor."
            )
        if not predictor.subject_dropout:
            raise RuntimeError(
                "optimize_for_inference() requires subject_dropout to be set "
                "for an average-subject predictor."
            )

        shared_index = predictor.n_subjects
        device = predictor.weights.device
        dtype = predictor.weights.dtype
        with torch.no_grad():
            weights = predictor.weights[shared_index].detach()
            if self.config.low_rank_head is not None:
                low_rank = self.low_rank_head
                if not isinstance(low_rank, nn.Linear):
                    raise TypeError(
                        "optimize_for_inference() requires an nn.Linear low_rank_head."
                    )
                if low_rank.bias is not None:
                    raise RuntimeError(
                        "optimize_for_inference() does not support a low-rank bias."
                    )
                # float64 on CPU keeps the weight composition accurate without
                # allocating a float64 matrix on the accelerator.
                fused = (
                    low_rank.weight.detach().to(device="cpu", dtype=torch.float64).T
                    @ weights.to(device="cpu", dtype=torch.float64)
                )
                weights = fused.to(device=device, dtype=dtype)
            else:
                # Avoid keeping the full per-subject storage alive through a view.
                weights = weights.clone()
            predictor.weights = nn.Parameter(
                weights.unsqueeze(0), requires_grad=False
            )
            if predictor.bias is not None:
                predictor.bias = nn.Parameter(
                    predictor.bias[shared_index].detach().unsqueeze(0).clone(),
                    requires_grad=False,
                )
            predictor.n_subjects = 0
            if self.config.low_rank_head is not None:
                self.low_rank_head = nn.Identity()
        self._inference_optimized = True
        return self

    def train(self, mode: bool = True):
        if mode and self._inference_optimized:
            raise RuntimeError(
                "Training is not supported after optimize_for_inference(); "
                "reload the original checkpoint to train."
            )
        return super().train(mode)

    def aggregate_features(self, batch):
        tensors = []
        # get B, T
        for modality in batch.data.keys():
            if modality in self.feature_dims:
                break
        x = batch.data[modality]
        B, T = x.shape[0], x.shape[-1]
        for modality in self.feature_dims.keys():
            if modality not in self.projectors or modality not in batch.data:
                data = torch.zeros(
                    B, T, self.config.hidden // len(self.feature_dims)
                ).to(x.device)
            else:
                data = batch.data[modality]  # B, L, H, T
                data = data.to(torch.float32)
                if data.ndim == 3:
                    data = data.unsqueeze(1)
                # mean over layers
                if self.config.layer_aggregation == "mean":
                    data = data.mean(dim=1)
                elif self.config.layer_aggregation == "cat":
                    data = rearrange(data, "b l d t -> b (l d) t")
                data = data.transpose(1, 2)
                assert data.ndim == 3  # B, T, D
                if isinstance(self.projectors[modality], SubjectLayersModel):
                    data = self.projectors[modality](
                        data.transpose(1, 2), batch.data["subject_id"]
                    ).transpose(1, 2)
                else:
                    data = self.projectors[modality](data)  # B, T, H
                if self.config.modality_dropout > 0 and self.training:
                    mask = torch.rand(data.shape[0]) < self.config.modality_dropout
                    data[mask, :] = torch.zeros_like(data[mask, :])
            tensors.append(data)
        if self.config.extractor_aggregation == "stack":
            out = torch.cat(tensors, dim=1)
        elif self.config.extractor_aggregation == "cat":
            out = torch.cat(tensors, dim=-1)
        elif self.config.extractor_aggregation == "sum":
            out = sum(tensors)
        if self.config.temporal_dropout > 0 and self.training:
            for batch_idx in range(out.shape[0]):
                mask = torch.rand(out.shape[1]) < self.config.temporal_dropout
                out[batch_idx, mask, :] = torch.zeros_like(out[batch_idx, mask, :])
        return out

    def transformer_forward(self, x, subject_id=None):
        x = self.combiner(x)
        if hasattr(self, "time_pos_embed"):
            x = x + self.time_pos_embed[:, : x.size(1)]
        if hasattr(self, "subject_embed"):
            x = x + self.subject_embed(subject_id)
        x = self.encoder(x)
        return x
