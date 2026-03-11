"""Standard SAE implementation adapted from goodfire-core's StandardSAE.

Uses decoder-weighted L1 sparsity, proper weight initialization, and structured
loss computation. Kept self-contained with no goodfire-core dependency.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class SAELoss:
	"""Loss components from an SAE training step."""

	loss: torch.Tensor
	reconstruction_loss: torch.Tensor
	sparsity_loss: torch.Tensor
	unweighted_sparsity_loss: torch.Tensor
	sae_features: torch.Tensor | None = None
	variance_explained: torch.Tensor | None = None

	def to_dict(self) -> dict[str, Any]:
		result = {}
		for field_name in self.__dataclass_fields__:
			field_value = getattr(self, field_name)
			if isinstance(field_value, torch.Tensor):
				if field_value.numel() == 1:
					result[field_name] = field_value.item()
			else:
				result[field_name] = field_value
		return result


class StandardSAE(nn.Module):
	"""Standard Sparse Autoencoder with decoder-weighted L1 sparsity.

	Architecture:
		x -> Linear(d_model, d_sae) -> ReLU -> f (features)
		f -> Linear(d_sae, d_model) -> x_hat (reconstruction)

	Sparsity: L_sparsity = sum(|f_i| * ||W_dec[:, i]||_2)

	Based on goodfire-core's StandardSAE implementation.
	"""

	def __init__(
		self,
		d_model: int,
		d_sae: int,
		decoder_weight_init_scale: float = 0.1,
	):
		super().__init__()
		self._d_model = d_model
		self._d_sae = d_sae

		self.encoder = nn.Linear(d_model, d_sae)
		self.decoder = nn.Linear(d_sae, d_model)

		self._initialize_weights(decoder_weight_init_scale)

	def _initialize_weights(self, decoder_weight_init_scale: float):
		# Decoder: xavier uniform, normalized to unit L2 per column, scaled
		nn.init.xavier_uniform_(self.decoder.weight)
		self.decoder.weight.data = self.decoder.weight.data / torch.linalg.vector_norm(
			self.decoder.weight.data, dim=0, keepdim=True
		)
		self.decoder.weight.data *= decoder_weight_init_scale

		# Encoder: initialized as decoder transpose
		with torch.no_grad():
			self.encoder.weight.data.copy_(self.decoder.weight.T.contiguous())
		self.encoder.weight.requires_grad = True

		# Biases: zero
		nn.init.zeros_(self.encoder.bias)
		nn.init.zeros_(self.decoder.bias)

	@property
	def d_model(self) -> int:
		return self._d_model

	@property
	def d_sae(self) -> int:
		return self._d_sae

	def encode(self, x: torch.Tensor) -> torch.Tensor:
		"""Encode to sparse features. (batch, d_model) -> (batch, d_sae)"""
		return F.relu(self.encoder(x))

	def decode(self, features: torch.Tensor) -> torch.Tensor:
		"""Decode features to reconstruction. (batch, d_sae) -> (batch, d_model)"""
		return self.decoder(features)

	def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
		"""Encode then decode. Returns (features, reconstruction)."""
		features = self.encode(x)
		reconstruction = self.decode(features)
		return features, reconstruction

	def compute_loss(self, x: torch.Tensor, sparsity_weight: float = 1.0) -> SAELoss:
		"""Compute MSE + decoder-weighted L1 loss.

		Args:
			x: (batch, d_model) input activations
			sparsity_weight: multiplier for the L1 term (i.e. l1_coeff)

		Returns:
			SAELoss with all components
		"""
		features = self.encode(x)
		x_hat = self.decode(features)

		# MSE reconstruction loss (sum over features, mean over batch)
		mse_loss = F.mse_loss(x_hat, x, reduction='none').sum(1).mean()

		# Variance explained
		ve = 1.0 - (x - x_hat).var() / x.var().clamp(min=1e-8)

		# Decoder-weighted L1 sparsity
		W_dec_norms = torch.linalg.vector_norm(self.decoder.weight, dim=0, ord=2)
		unweighted_sparsity = (features * W_dec_norms).abs().sum(1).mean()
		sparsity_loss = unweighted_sparsity * sparsity_weight

		total_loss = mse_loss + sparsity_loss

		return SAELoss(
			loss=total_loss,
			reconstruction_loss=mse_loss,
			sparsity_loss=sparsity_loss,
			unweighted_sparsity_loss=unweighted_sparsity,
			sae_features=features,
			variance_explained=ve,
		)

	def reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
		"""Per-sample MSE reconstruction error. (batch, d_model) -> (batch,)"""
		_, reconstruction = self.forward(x)
		return ((x - reconstruction) ** 2).mean(dim=-1)

	def save_checkpoint(self, path: str, **extra_metadata):
		"""Save SAE checkpoint with config and state dict."""
		checkpoint = {
			'model_config': {
				'd_model': self._d_model,
				'd_sae': self._d_sae,
			},
			'state_dict': self.state_dict(),
		}
		checkpoint.update(extra_metadata)
		torch.save(checkpoint, path)

	@classmethod
	def from_checkpoint(cls, path: str) -> StandardSAE:
		"""Load SAE from checkpoint."""
		checkpoint = torch.load(path, map_location='cpu', weights_only=False)
		config = checkpoint['model_config']
		sae = cls(d_model=config['d_model'], d_sae=config['d_sae'])
		sae.load_state_dict(checkpoint['state_dict'])
		return sae
