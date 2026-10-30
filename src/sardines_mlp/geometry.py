"""
What a trained W says about how features are stored. Every measurement is defined here once.

Wᵢ, column i of W, is the direction that represents feature i. From the columns:
- ‖Wᵢ‖ says whether feature i is represented at all (about 1) or dropped (about 0).
- Superposition, Σ_{j≠i} (Ŵᵢ · Wⱼ)², is how much of the other features lands on feature i's
  direction.
- Feature dimensionality, Dᵢ = ‖Wᵢ‖² / Σⱼ (Ŵᵢ · Wⱼ)², is the fraction of a dimension feature
  i gets: 1 alone, ½ in an antipodal pair, ⅖ in a pentagon, 0 when dropped.
- Dimensions per feature, m / ‖W‖²_F, averages that over the whole model.
"""

import math
from typing import Literal

import torch
from jaxtyping import Float
from torch import Tensor

REPRESENTED = 0.5
"""A feature counts as represented when ‖Wᵢ‖ exceeds this; trained norms sit near 0 or near 1."""

Phase = Literal['not learned', 'dedicated', 'superposition', 'other']
PHASES: tuple[Phase, ...] = ('not learned', 'dedicated', 'superposition', 'other')


def norms(W: Float[Tensor, 'instances hidden features']) -> Float[Tensor, 'instances features']:
  return W.norm(dim=1)


def overlaps(W: Float[Tensor, 'instances hidden features']) -> Float[Tensor, 'instances features features']:
  """(Ŵᵢ · Wⱼ)² for every pair, row i the feature whose direction is projected onto."""
  # A dropped feature's column is near zero, so its unit vector is noise; the clamp only
  # keeps that noise finite. Its numbers are not read as geometry (see represented()).
  unit = W / W.norm(dim=1, keepdim=True).clamp_min(1e-8)
  return torch.einsum('imf,img->ifg', unit, W) ** 2


def superposition(W: Float[Tensor, 'instances hidden features']) -> Float[Tensor, 'instances features']:
  """Σ_{j≠i} (Ŵᵢ · Wⱼ)²: the other features' weight along feature i's direction."""
  o = overlaps(W)
  return o.sum(-1) - o.diagonal(dim1=-2, dim2=-1)


def feature_dimensionality(W: Float[Tensor, 'instances hidden features']) -> Float[Tensor, 'instances features']:
  """Dᵢ = ‖Wᵢ‖² / Σⱼ (Ŵᵢ · Wⱼ)², the j = i term included."""
  return norms(W) ** 2 / overlaps(W).sum(-1).clamp_min(1e-12)


def dimensions_per_feature(W: Float[Tensor, 'instances hidden features']) -> Float[Tensor, ' instances']:
  """M / ‖W‖²_F. ‖W‖²_F is roughly the number of features represented, in any basis."""
  return W.shape[1] / W.pow(2).sum((1, 2))


def represented(W: Float[Tensor, 'instances hidden features']) -> Tensor:
  return norms(W) > REPRESENTED


def phase(W: Float[Tensor, 'instances 1 2']) -> list[Phase]:
  """
  What an n = 2, m = 1 model did with the extra feature 2.

  - not learned: feature 2 is dropped (W ≈ [1, 0]).
  - dedicated: feature 2 gets the dimension and feature 1 is dropped (W ≈ [0, 1]).
  - superposition: both share it with opposite signs (W ≈ [1, −1]), the antipodal solution.
  - other: anything else, such as both with the same sign, which the paper's three natural
    solutions do not include.
  """
  w1, w2 = W[:, 0, 0], W[:, 0, 1]
  has1, has2 = w1.abs() > REPRESENTED, w2.abs() > REPRESENTED
  result: list[Phase] = []
  for a, b, one, two in zip(w1.tolist(), w2.tolist(), has1.tolist(), has2.tolist(), strict=True):
    if not two:
      result.append('not learned')
    elif not one:
      result.append('dedicated')
    elif a * b < 0:
      result.append('superposition')
    else:
      result.append('other')
  return result


def polygon(W: Float[Tensor, '2 features'], tolerance: float = 0.15) -> tuple[int, bool]:
  """
  The shape the represented features of one m = 2 instance make in the plane.

  Returns the number of represented features k and whether they are evenly spread around
  the origin, every angular gap within `tolerance` radians of 2π/k: a regular k-gon (k = 2
  is an antipodal pair). Two orthogonal features, the dense solution, are not regular.
  """
  columns = W[:, W.norm(dim=0) > REPRESENTED]
  k = columns.shape[1]
  if k < 2:
    return k, False
  angles = torch.atan2(columns[1], columns[0]).sort().values
  gaps = torch.diff(angles, append=angles[:1] + 2 * math.pi)
  return k, bool((gaps - 2 * math.pi / k).abs().max() < tolerance)
