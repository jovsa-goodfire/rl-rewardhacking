import torch
import pytest

from sae import StandardSAE, SAELoss


def test_sae_shapes():
	"""Test that encode/decode/forward produce correct shapes."""
	d_model, d_sae, batch = 64, 256, 32
	sae = StandardSAE(d_model, d_sae)
	x = torch.randn(batch, d_model)

	features = sae.encode(x)
	assert features.shape == (batch, d_sae)

	recon = sae.decode(features)
	assert recon.shape == (batch, d_model)

	features2, recon2 = sae.forward(x)
	assert features2.shape == (batch, d_sae)
	assert recon2.shape == (batch, d_model)

	errors = sae.reconstruction_error(x)
	assert errors.shape == (batch,)


def test_compute_loss_returns_sae_loss():
	"""compute_loss should return an SAELoss with all components."""
	sae = StandardSAE(64, 256)
	x = torch.randn(32, 64)

	loss = sae.compute_loss(x, sparsity_weight=1e-3)
	assert isinstance(loss, SAELoss)
	assert loss.loss.shape == ()
	assert loss.reconstruction_loss.shape == ()
	assert loss.sparsity_loss.shape == ()
	assert loss.variance_explained.shape == ()
	assert loss.sae_features is not None
	assert loss.sae_features.shape == (32, 256)

	# Total loss = recon + sparsity
	expected = loss.reconstruction_loss + loss.sparsity_loss
	assert torch.allclose(loss.loss, expected, atol=1e-5)


def test_sae_training_loss_decreases():
	"""Train SAE for a few epochs and verify loss decreases."""
	sae = StandardSAE(64, 256)
	optimizer = torch.optim.Adam(sae.parameters(), lr=1e-3)

	x = torch.randn(512, 64)

	losses = []
	for _ in range(20):
		sae_loss = sae.compute_loss(x, sparsity_weight=1e-3)
		optimizer.zero_grad()
		sae_loss.loss.backward()
		optimizer.step()
		losses.append(sae_loss.loss.item())

	assert losses[-1] < losses[0], f'Loss did not decrease: {losses[0]:.4f} -> {losses[-1]:.4f}'


def test_sae_sparsity():
	"""After training, encoded features should be sparse (>50% zeros)."""
	sae = StandardSAE(64, 256)
	optimizer = torch.optim.Adam(sae.parameters(), lr=1e-3)

	x = torch.randn(512, 64)
	for _ in range(50):
		sae_loss = sae.compute_loss(x, sparsity_weight=1e-2)
		optimizer.zero_grad()
		sae_loss.loss.backward()
		optimizer.step()

	sae.eval()
	with torch.no_grad():
		features = sae.encode(x)
		sparsity = (features == 0).float().mean().item()

	assert sparsity > 0.5, f'Sparsity too low: {sparsity:.1%}'


def test_sae_biases_initialized_to_zero():
	"""Encoder and decoder biases should be initialized to zero."""
	sae = StandardSAE(64, 256)
	assert torch.all(sae.encoder.bias == 0)
	assert torch.all(sae.decoder.bias == 0)


def test_encoder_init_is_decoder_transpose():
	"""Encoder weights should be initialized as decoder transpose."""
	sae = StandardSAE(64, 256)
	assert torch.allclose(sae.encoder.weight.data, sae.decoder.weight.T.contiguous())


def test_decoder_columns_normalized():
	"""Decoder weight columns should be initialized with unit L2 norm (before scaling)."""
	sae = StandardSAE(64, 256, decoder_weight_init_scale=1.0)
	col_norms = torch.linalg.vector_norm(sae.decoder.weight.data, dim=0, ord=2)
	assert torch.allclose(col_norms, torch.ones_like(col_norms), atol=1e-5)


def test_save_and_load_checkpoint(tmp_path):
	"""Save and reload checkpoint, verify weights match."""
	sae = StandardSAE(64, 256)
	path = str(tmp_path / 'sae.pt')
	sae.save_checkpoint(path, my_extra='hello')

	loaded = StandardSAE.from_checkpoint(path)
	assert loaded.d_model == 64
	assert loaded.d_sae == 256

	for (k1, v1), (k2, v2) in zip(sae.state_dict().items(), loaded.state_dict().items()):
		assert k1 == k2
		assert torch.allclose(v1, v2)


def test_variance_explained_improves():
	"""Variance explained should increase during training."""
	sae = StandardSAE(64, 256)
	optimizer = torch.optim.Adam(sae.parameters(), lr=1e-3)

	x = torch.randn(512, 64)
	ve_values = []
	for _ in range(30):
		sae_loss = sae.compute_loss(x, sparsity_weight=1e-3)
		optimizer.zero_grad()
		sae_loss.loss.backward()
		optimizer.step()
		ve_values.append(sae_loss.variance_explained.item())

	assert ve_values[-1] > ve_values[0], f'VE did not improve: {ve_values[0]:.3f} -> {ve_values[-1]:.3f}'


if __name__ == '__main__':
	pytest.main([__file__, '-v'])
