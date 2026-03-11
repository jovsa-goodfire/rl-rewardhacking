import argparse
import os

import torch
from torch.utils.data import DataLoader, TensorDataset

from sae import StandardSAE


def parse_args():
	parser = argparse.ArgumentParser(description='Train a StandardSAE on collected activations')
	parser.add_argument('--activations_dir', type=str, required=True, help='Path to results/rlookout/<model>/<run_name>/')
	parser.add_argument('--checkpoints', type=str, default='0,50,100,200', help='Comma-separated checkpoint steps')
	parser.add_argument('--weights', type=str, default='0.4,0.3,0.2,0.1', help='Sampling weights per checkpoint')
	parser.add_argument('--dict_size', type=int, default=8192)
	parser.add_argument('--l1_coeff', type=float, default=1e-3)
	parser.add_argument('--lr', type=float, default=3e-4)
	parser.add_argument('--epochs', type=int, default=50)
	parser.add_argument('--batch_size', type=int, default=256)
	parser.add_argument('--layer_idx', type=int, default=0, help='Layer index within the activation tensor (dim 0)')
	parser.add_argument('--clip_grad_max_norm', type=float, default=1.0, help='Max gradient norm for clipping (0 to disable)')
	return parser.parse_args()


def load_and_mix_activations(
	activations_dir: str,
	checkpoints: list[int],
	weights: list[float],
	layer_idx: int = 0,
) -> torch.Tensor:
	"""Load checkpoint activation files and mix them according to weights.

	Each checkpoint file is expected to contain:
		'activations': Tensor(n_layers, n_samples, hidden_dim)

	Returns:
		Tensor(total_samples, hidden_dim)
	"""
	all_acts = []
	for step, weight in zip(checkpoints, weights):
		path = os.path.join(activations_dir, f'checkpoint_{step}.pt')
		data = torch.load(path, map_location='cpu', weights_only=False)
		acts = data['activations'][layer_idx]  # (n_samples, hidden_dim)
		n_keep = max(1, int(len(acts) * weight))
		perm = torch.randperm(len(acts))[:n_keep]
		all_acts.append(acts[perm].float())
		print(f'  Checkpoint {step}: {len(acts)} samples, keeping {n_keep} (weight={weight})')

	combined = torch.cat(all_acts, dim=0)
	print(f'  Total training samples: {len(combined)}')
	return combined


def train_sae(
	activations: torch.Tensor,
	dict_size: int,
	l1_coeff: float,
	lr: float,
	epochs: int,
	batch_size: int,
	clip_grad_max_norm: float = 1.0,
) -> tuple[StandardSAE, list[dict]]:
	"""Train a StandardSAE on the given activations.

	Returns:
		(trained_sae, loss_history) where loss_history is a list of dicts per epoch
	"""
	device = 'cuda' if torch.cuda.is_available() else 'cpu'
	d_model = activations.shape[-1]
	sae = StandardSAE(d_model, dict_size).to(device)
	optimizer = torch.optim.Adam(sae.parameters(), lr=lr)

	dataset = TensorDataset(activations)
	loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

	loss_history = []
	for epoch in range(epochs):
		epoch_totals = {'loss': 0.0, 'recon': 0.0, 'sparsity': 0.0, 've': 0.0}
		n_batches = 0

		for (batch,) in loader:
			batch = batch.to(device)
			sae_loss = sae.compute_loss(batch, sparsity_weight=l1_coeff)

			optimizer.zero_grad()
			sae_loss.loss.backward()
			if clip_grad_max_norm > 0:
				torch.nn.utils.clip_grad_norm_(sae.parameters(), clip_grad_max_norm)
			optimizer.step()

			epoch_totals['loss'] += sae_loss.loss.item()
			epoch_totals['recon'] += sae_loss.reconstruction_loss.item()
			epoch_totals['sparsity'] += sae_loss.sparsity_loss.item()
			epoch_totals['ve'] += sae_loss.variance_explained.item()
			n_batches += 1

		epoch_avg = {k: v / n_batches for k, v in epoch_totals.items()}
		loss_history.append(epoch_avg)

		if (epoch + 1) % 10 == 0 or epoch == 0:
			print(
				f'  Epoch {epoch + 1}/{epochs} — '
				f'loss: {epoch_avg["loss"]:.6f} '
				f'(recon: {epoch_avg["recon"]:.6f}, '
				f'sparsity: {epoch_avg["sparsity"]:.6f}, '
				f've: {epoch_avg["ve"]:.3f})'
			)

	return sae.cpu(), loss_history


def main():
	args = parse_args()

	checkpoints = [int(x) for x in args.checkpoints.split(',')]
	weights = [float(x) for x in args.weights.split(',')]
	assert len(checkpoints) == len(weights), f'Mismatch: {len(checkpoints)} checkpoints vs {len(weights)} weights'

	print(f'Loading activations from {args.activations_dir}')
	activations = load_and_mix_activations(
		args.activations_dir, checkpoints, weights, layer_idx=args.layer_idx
	)

	d_model = activations.shape[-1]
	print(f'Training SAE (d_model={d_model}, d_sae={args.dict_size})')
	sae, loss_history = train_sae(
		activations,
		dict_size=args.dict_size,
		l1_coeff=args.l1_coeff,
		lr=args.lr,
		epochs=args.epochs,
		batch_size=args.batch_size,
		clip_grad_max_norm=args.clip_grad_max_norm,
	)

	# Save using StandardSAE's checkpoint format + training metadata
	save_path = os.path.join(args.activations_dir, 'sae.pt')
	os.makedirs(os.path.dirname(save_path), exist_ok=True)
	sae.save_checkpoint(
		save_path,
		training_config={
			'l1_coeff': args.l1_coeff,
			'lr': args.lr,
			'epochs': args.epochs,
			'batch_size': args.batch_size,
			'checkpoints': checkpoints,
			'weights': weights,
			'clip_grad_max_norm': args.clip_grad_max_norm,
		},
		final_loss=loss_history[-1]['loss'],
		loss_history=loss_history,
	)
	print(f'SAE saved to {save_path}')

	# Quick validation
	sae.eval()
	with torch.no_grad():
		sample = activations[:min(256, len(activations))]
		features = sae.encode(sample)
		recon_error = sae.reconstruction_error(sample)

	print(f'\nValidation:')
	print(f'  Reconstruction MSE: {recon_error.mean():.4f}')
	print(f'  Sparsity (% zeros): {(features == 0).float().mean():.1%}')
	print(f'  Dead features: {(features.sum(0) == 0).sum().item()} / {args.dict_size}')
	print(f'  Active features per sample: {(features > 0).float().sum(1).mean():.0f}')


if __name__ == '__main__':
	main()
