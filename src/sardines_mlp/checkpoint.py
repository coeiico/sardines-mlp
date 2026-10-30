"""
Saving and loading a trained sweep: the weights and the per-instance feature settings.

A run folder holds model.safetensors (W and b), sweep.safetensors (each instance's
feature probability and importance) and metrics.json (the config, coordinates and results).
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from safetensors.torch import load_file, save_file

from sardines_mlp.experiments import Sweep
from sardines_mlp.model import ToyModel


@dataclass
class Run:
  model: ToyModel
  sweep: Sweep
  record: dict[str, Any]
  """metrics.json as written by train.py."""


def save(run_dir: Path, model: ToyModel, sweep: Sweep) -> None:
  save_file({k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}, run_dir / 'model.safetensors')
  tensors = {'feature_probability': sweep.feature_probability, 'importance': sweep.importance}
  save_file({k: v.contiguous() for k, v in tensors.items()}, run_dir / 'sweep.safetensors')


def load(run_dir: Path) -> Run:
  record = json.loads((run_dir / 'metrics.json').read_text())
  settings = load_file(run_dir / 'sweep.safetensors')
  weights = load_file(run_dir / 'model.safetensors')
  n_instances, n_hidden, n_features = weights['W'].shape
  sweep = Sweep(n_features, n_hidden, settings['feature_probability'], settings['importance'], record['coordinates'])
  model = ToyModel(n_instances, n_features, n_hidden, record['config']['model']['activation'])
  model.load_state_dict(weights, strict=True)
  model.eval()
  return Run(model, sweep, record)

