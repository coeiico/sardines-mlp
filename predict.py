"""Load a trained sardines-mlp run and reconstruct a few fresh samples, as a sanity view."""

from dataclasses import dataclass
from pathlib import Path

import torch
import tyro

from sardines_mlp import checkpoint, features


@dataclass
class Config:
  run: Path
  """Run directory, e.g. runs/polygons-relu-seed0."""
  instance: int = 0
  """Which instance of the sweep to show."""
  samples: int = 4
  seed: int = 0
  """Seeds the samples shown, a stream separate from training and evaluation."""


def main(cfg: Config) -> None:
  run = checkpoint.load(cfg.run)
  k = cfg.instance
  probability = run.sweep.feature_probability
  # Keep drawing until the instance shows some active features, since sparse ones are mostly 0.
  generator = torch.Generator().manual_seed(cfg.seed)
  x = features.sample(cfg.samples * 50, probability, generator)
  x = x[(x[:, k] > 0).any(-1)][: cfg.samples]
  with torch.no_grad():
    x_hat = run.model(x)
  coords = {name: round(values[k], 4) for name, values in run.sweep.coordinates.items()}
  print(f'{cfg.run}, instance {k} {coords}: input, then reconstruction')
  for a, b in zip(x[:, k], x_hat[:, k], strict=True):
    print('  in ', a.numpy().round(2))
    print('  out', b.numpy().round(2))


if __name__ == '__main__':
  main(tyro.cli(Config))
