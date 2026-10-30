"""
The four sweeps, each a grid of independent instances trained together.

Settings follow Elhage et al. 2022 where the paper states them (checked against its text,
see the plan log); the grids' spacing is ours.
"""

from dataclasses import dataclass
from typing import Literal

import torch
from jaxtyping import Float
from torch import Tensor

Experiment = Literal['intro', 'polygons', 'phase', 'dimensionality']


@dataclass(frozen=True)
class Sweep:
  n_features: int
  n_hidden: int
  feature_probability: Float[Tensor, 'instances features']
  """1 − S per instance and feature."""
  importance: Float[Tensor, 'instances features']
  coordinates: dict[str, list[float]]
  """What varies across instances, one value per instance, for analysis and plots."""

  @property
  def n_instances(self) -> int:
    return self.feature_probability.shape[0]


def _uniform(n_features: int, density: Tensor, importance: Tensor) -> tuple[Tensor, Tensor]:
  """Every feature of an instance shares its density; importance is the same in every instance."""
  probability = density[:, None].expand(-1, n_features).clone()
  return probability, importance[None, :].expand(len(density), -1).clone()


def intro() -> Sweep:
  """Demonstrating Superposition: n = 20 into m = 5, Iᵢ = 0.7ⁱ, one instance per sparsity."""
  n = 20
  sparsity = torch.tensor([0.0, 0.7, 0.9, 0.97, 0.99, 0.997, 0.999])
  density = 1 - sparsity
  probability, importance = _uniform(n, density, 0.7 ** torch.arange(n, dtype=torch.float32))
  return Sweep(n, 5, probability, importance, {'density': density.tolist()})


def polygons() -> Sweep:
  """
  Perturbing a single feature: n = 5 into m = 2 with uniform importance.

  The paper shows a regular pentagon at density 0.05; the sweep runs from dense to 0.01
  in 21 log-spaced steps, which puts 10^-1.3 ≈ 0.05 on the grid.
  """
  n = 5
  density = 10 ** -torch.linspace(0, 2, 21)
  probability, importance = _uniform(n, density, torch.ones(n))
  return Sweep(n, 2, probability, importance, {'density': density.tolist()})


def phase() -> Sweep:
  """
  Superposition as a Phase Change: n = 2 into m = 1.

  Feature 1 has importance 1, the extra feature 2 has importance r. The paper varies r from 0.1 to 10 and the density from 1 to 0.01; here each is a
  24-point log grid, density-major (instance = density index × 24 + r index).
  """
  steps = 24
  density = 10 ** -torch.linspace(0, 2, steps)
  relative = 10 ** torch.linspace(-1, 1, steps)
  grid_density, grid_relative = torch.meshgrid(density, relative, indexing='ij')
  grid_density, grid_relative = grid_density.flatten(), grid_relative.flatten()
  probability = grid_density[:, None].expand(-1, 2).clone()
  importance = torch.stack([torch.ones_like(grid_relative), grid_relative], dim=-1)
  coordinates = {'density': grid_density.tolist(), 'relative_importance': grid_relative.tolist()}
  return Sweep(2, 1, probability, importance, coordinates)


def dimensionality() -> Sweep:
  """Uniform superposition: n = 400 into m = 30, uniform importance, 1/(1 − S) log-spaced from 1 to 1,000."""
  n = 400
  density = 10 ** -torch.linspace(0, 3, 40)
  probability, importance = _uniform(n, density, torch.ones(n))
  return Sweep(n, 30, probability, importance, {'density': density.tolist()})


PRESETS = {'intro': intro, 'polygons': polygons, 'phase': phase, 'dimensionality': dimensionality}


def build(experiment: Experiment) -> Sweep:
  return PRESETS[experiment]()
