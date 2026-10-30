"""
Train one sweep of toy models: every instance of an experiment's grid at once.

`--experiment` picks the grid (intro, polygons, phase or dimensionality) and
`--model.activation none` trains the linear baseline on it. `just quick` runs this with
--quick: a tiny CPU smoke test that must finish in under 5 minutes.

The instances are independent models. Their losses are summed, so each one's gradient
comes only from its own loss, and Adam scales every weight on its own, so summing does not
couple them either (tests/test_model.py checks the gradients).
"""

import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Literal

import matplotlib

matplotlib.use('Agg')  # write PNGs without needing a display
import matplotlib.pyplot as plt
import torch
import tyro

from sardines_mlp import checkpoint, evaluation, experiments, features
from sardines_mlp.model import Activation, ToyModel, weighted_loss
from sardines_mlp.utils import (
  code_state,
  environment,
  get_device,
  make_run_dir,
  save_json,
  seed_everything,
)


@dataclass(frozen=True)
class ModelArgs:
  activation: Activation = 'relu'
  """'relu' is the toy model; 'none' is the linear baseline, which cannot use superposition."""


@dataclass(frozen=True)
class OptimArgs:
  steps: int = 30_000
  """Chosen on the development seed: at 10,000 the sparse instances were still improving."""
  batch_size: int = 1024
  lr: float = 1e-3
  """Adam, at a constant rate."""
  weight_decay: float = 0.0
  """None by default: decay would shrink ‖Wᵢ‖ and blur whether a feature is represented."""


@dataclass(frozen=True)
class EvalArgs:
  samples: int = 65_536
  """Fresh samples each instance is scored on."""
  chunk: int = 4096
  """Samples generated at a time, to bound memory."""
  seed: int = 1_000_003
  """The evaluation stream's seed, the same for every run and never used for training."""
  every: int = 1000
  """Steps between evaluations."""


@dataclass
class Config:
  run_name: str = 'default'
  """Outputs go to runs/<run-name>/, which must not exist yet."""
  experiment: experiments.Experiment = 'polygons'
  seed: int = 0
  """Seeds the initial weights and the stream of training samples. 100 is the development seed."""
  quick: bool = False
  """Run the tiny CPU smoke test instead of the full experiment."""
  device: Literal['auto', 'cpu', 'cuda', 'mps'] = 'auto'
  model: ModelArgs = field(default_factory=ModelArgs)
  optim: OptimArgs = field(default_factory=OptimArgs)
  eval: EvalArgs = field(default_factory=EvalArgs)


def quick_config(cfg: Config) -> Config:
  """Shrink everything so the whole pipeline runs on a CPU in well under a minute."""
  return replace(
    cfg,
    experiment='polygons',
    device='cpu',
    optim=replace(cfg.optim, steps=2000),
    eval=replace(cfg.eval, samples=8192, every=500),
  )


def plot_curves(history: list[dict[str, Any]], record: dict[str, Any], path: Path) -> None:
  steps = [h['step'] for h in history]
  losses = torch.tensor([h['loss'] for h in history])  # (evaluations, instances)
  zero = torch.tensor(record['baselines']['zero_output'])
  top_m = torch.tensor(record['baselines']['keep_top_m'])
  fig, ax = plt.subplots(figsize=(8, 4.5))
  # Relative to outputting nothing, so instances of very different sparsity share one scale.
  for k in range(losses.shape[1]):
    ax.plot(steps, losses[:, k] / zero[k], color='C0', alpha=0.35, lw=0.8)
  ax.set(xlabel='step', ylabel='eval loss / zero-output loss', yscale='log')
  beats = int((losses[-1] < top_m * 0.99).sum())
  m = record['sweep']['n_hidden']
  ax.set_title(f'{record["config"]["experiment"]}, {record["config"]["model"]["activation"]}: '
               f'{beats} of {losses.shape[1]} instances end more than 1% below the keep-top-{m} loss')
  fig.tight_layout()
  fig.savefig(path, dpi=120)
  plt.close(fig)


def main(cfg: Config) -> None:
  if cfg.quick:
    cfg = quick_config(cfg)
  seed_everything(cfg.seed)
  device = get_device() if cfg.device == 'auto' else torch.device(cfg.device)
  run_dir = make_run_dir(cfg.run_name)
  optim = cfg.optim

  sweep = experiments.build(cfg.experiment)
  model = ToyModel(sweep.n_instances, sweep.n_features, sweep.n_hidden, cfg.model.activation).to(device)
  optimizer = torch.optim.AdamW(model.parameters(), lr=optim.lr, weight_decay=optim.weight_decay)
  probability = sweep.feature_probability.to(device)
  importance = sweep.importance.to(device)
  train_stream = torch.Generator(device).manual_seed(cfg.seed)

  print(f'sardines-mlp: {cfg.experiment}, {sweep.n_instances} instances of {sweep.n_features} → {sweep.n_hidden}, '
        f'{cfg.model.activation}, on {device}, writing to {run_dir}')
  example = features.sample(4, sweep.feature_probability[:1], torch.Generator().manual_seed(0))[:, 0]
  print(f'  four samples for instance 0 (density {sweep.feature_probability[0, 0]:.3f}):\n{example.numpy().round(2)}')

  record: dict[str, Any] = {
    'config': asdict(cfg),
    'sweep': {
      'n_instances': sweep.n_instances,
      'n_features': sweep.n_features,
      'n_hidden': sweep.n_hidden,
      'distribution': 'x_i = 0 with probability S, else U[0, 1], independent; density = 1 − S',
    },
    'coordinates': sweep.coordinates,
    'importance': sweep.importance.tolist(),
    'evaluation': f'{cfg.eval.samples} fresh samples per instance from a generator seeded {cfg.eval.seed} on {device.type}',
    'baselines': evaluation.baselines(sweep),
    'code': code_state(run_dir),
    'environment': environment(device),
    'resumable': False,
    'status': 'running',
  }
  save_json(run_dir / 'metrics.json', record)

  def evaluate(step: int) -> dict[str, Any]:
    loss = evaluation.loss(model, sweep, cfg.eval.samples, cfg.eval.chunk, cfg.eval.seed, device)
    return {'step': step, 'loss': loss.tolist()}

  # Step 0 scores the untrained model, the third baseline.
  history = [evaluate(0)]
  start = time.time()
  for step in range(1, optim.steps + 1):
    x = features.sample(optim.batch_size, probability, train_stream)

    x_hat = model(x)
    loss = weighted_loss(x, x_hat, importance).sum()
    if not torch.isfinite(loss):
      record.update(status=f'failed: non-finite loss {loss.item()} at step {step}', history=history)
      save_json(run_dir / 'metrics.json', record)
      raise SystemExit(f'Stopping: loss is {loss.item()} at step {step}.')
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % cfg.eval.every == 0 or step == optim.steps:
      history.append(evaluate(step))
      print(f'step {step:6d}  mean eval loss {sum(history[-1]["loss"]) / sweep.n_instances:.5f}')

  checkpoint.save(run_dir, model, sweep)
  final = torch.tensor(history[-1]['loss'])
  # Convergence: how much each instance's loss still moved over the last 10% of training.
  late = [h for h in history if h['step'] <= 0.9 * optim.steps][-1]
  late_change = (torch.tensor(late['loss']) - final) / final.clamp_min(1e-12)
  record.update(
    status='finished',
    seconds=round(time.time() - start, 1),
    history=history,
    final={
      'loss': final.tolist(),
      'untrained_loss': history[0]['loss'],
      'late_relative_change': late_change.tolist(),
      'max_late_relative_change': late_change.max().item(),
    },
  )
  save_json(run_dir / 'metrics.json', record)
  plot_curves(history, record, run_dir / 'training.png')
  print(f'done in {record["seconds"]}s; loss fell by at most {late_change.max().item():.2%} '
        f'over the last 10% of steps: {run_dir}')


if __name__ == '__main__':
  main(tyro.cli(Config))
