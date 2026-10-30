"""A saved and reloaded run reproduces the trained model's outputs and its evaluation."""

import json
from pathlib import Path

import torch

from sardines_mlp import checkpoint, evaluation, experiments
from sardines_mlp.model import ToyModel


def test_save_and_load_give_identical_outputs(tmp_path: Path) -> None:
  torch.manual_seed(0)
  sweep = experiments.polygons()
  model = ToyModel(sweep.n_instances, sweep.n_features, sweep.n_hidden, 'relu')
  with torch.no_grad():
    model.b.uniform_(-0.2, 0.2)
  checkpoint.save(tmp_path, model, sweep)
  record = {'config': {'model': {'activation': 'relu'}}, 'coordinates': sweep.coordinates}
  (tmp_path / 'metrics.json').write_text(json.dumps(record))

  run = checkpoint.load(tmp_path)
  x = torch.rand(8, sweep.n_instances, sweep.n_features)
  with torch.no_grad():
    assert torch.equal(run.model(x), model(x))
  assert torch.equal(run.sweep.importance, sweep.importance)
  assert run.sweep.coordinates == sweep.coordinates
  cpu = torch.device('cpu')
  assert torch.equal(
    evaluation.loss(run.model, run.sweep, 1000, 300, 7, cpu),
    evaluation.loss(model, sweep, 1000, 300, 7, cpu),
  )


def test_evaluation_is_the_same_samples_every_call() -> None:
  sweep = experiments.intro()
  model = ToyModel(sweep.n_instances, sweep.n_features, sweep.n_hidden)
  cpu = torch.device('cpu')
  a = evaluation.loss(model, sweep, 2000, 512, 11, cpu)
  b = evaluation.loss(model, sweep, 2000, 512, 11, cpu)
  c = evaluation.loss(model, sweep, 2000, 512, 12, cpu)
  assert torch.equal(a, b) and not torch.equal(a, c)
  assert model.training  # the evaluation helper restores the mode it found
