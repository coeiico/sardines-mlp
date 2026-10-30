"""
Scoring a sweep on fresh samples, and the baselines every score is compared with.

Evaluation samples come from their own generator, seeded once for the whole project and
never used for training. There is no dataset to leak, so these fresh samples are the
held-out set.
"""

from typing import Any

import torch
from torch import Tensor

from sardines_mlp import features
from sardines_mlp.experiments import Sweep
from sardines_mlp.model import ToyModel, weighted_loss


def loss(model: ToyModel, sweep: Sweep, n_samples: int, chunk: int, seed: int, device: torch.device) -> Tensor:
  """
  Per-instance loss on n_samples fresh samples, the same samples on every call and run.

  The samples are regenerated in chunks rather than stored: a 400-feature sweep of 40
  instances would need gigabytes to hold 65,536 of them. CUDA and CPU generators give
  different streams, so samples are identical across runs on the same device type.
  """
  generator = torch.Generator(device).manual_seed(seed)
  probability = sweep.feature_probability.to(device)
  importance = sweep.importance.to(device)
  total = torch.zeros(sweep.n_instances, device=device)
  was_training = model.training
  model.eval()
  try:
    with torch.no_grad():
      for start in range(0, n_samples, chunk):
        size = min(chunk, n_samples - start)
        x = features.sample(size, probability, generator)
        total += weighted_loss(x, model(x), importance) * size
  finally:
    model.train(was_training)
  return (total / n_samples).cpu()


def baselines(sweep: Sweep) -> dict[str, Any]:
  """The closed-form losses: output nothing, or keep the m best features and drop the rest."""
  return {
    'zero_output': features.zero_output_loss(sweep.feature_probability, sweep.importance).tolist(),
    'keep_top_m': features.keep_top_m_loss(sweep.feature_probability, sweep.importance, sweep.n_hidden).tolist(),
  }
