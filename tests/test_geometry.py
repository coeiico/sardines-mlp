"""Geometry measurements on hand-built weights whose answers are known."""

import math

import pytest
import torch

from sardines_mlp import geometry


def _regular(k: int, n: int | None = None) -> torch.Tensor:
  """Unit columns evenly spaced round the circle, k of them, padded with zero columns to n."""
  angles = torch.arange(k) * 2 * math.pi / k
  W = torch.stack([angles.cos(), angles.sin()])
  if n is not None:
    W = torch.cat([W, torch.zeros(2, n - k)], dim=1)
  return W


def test_antipodal_pair_has_half_a_dimension_each() -> None:
  W = torch.tensor([[[1.0, -1.0]]])
  assert torch.allclose(geometry.feature_dimensionality(W), torch.tensor([[0.5, 0.5]]))
  assert torch.allclose(geometry.superposition(W), torch.tensor([[1.0, 1.0]]))
  assert geometry.dimensions_per_feature(W).item() == pytest.approx(0.5)


def test_pentagon_has_two_fifths() -> None:
  W = _regular(5)[None]
  assert torch.allclose(geometry.feature_dimensionality(W), torch.full((1, 5), 0.4), atol=1e-6)
  assert geometry.dimensions_per_feature(W).item() == pytest.approx(0.4)
  assert geometry.polygon(W[0]) == (5, True)


def test_orthogonal_features_have_one_dimension_and_dropped_ones_zero() -> None:
  W = torch.tensor([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]])
  assert torch.allclose(geometry.feature_dimensionality(W), torch.tensor([[1.0, 1.0, 0.0]]))
  assert torch.allclose(geometry.superposition(W)[0, :2], torch.zeros(2))
  assert geometry.represented(W).tolist() == [[True, True, False]]
  assert geometry.polygon(W[0]) == (2, False)


def test_tetrahedron_has_three_quarters() -> None:
  # A regular tetrahedron's vertices in 3D, scaled to unit length.
  W = torch.tensor([[1.0, 1.0, -1.0, -1.0], [1.0, -1.0, 1.0, -1.0], [1.0, -1.0, -1.0, 1.0]]) / math.sqrt(3)
  assert torch.allclose(geometry.feature_dimensionality(W[None]), torch.full((1, 4), 0.75), atol=1e-6)


def test_polygon_ignores_dropped_features() -> None:
  assert geometry.polygon(_regular(3, n=5)) == (3, True)
  assert geometry.polygon(_regular(2, n=5)) == (2, True)


def test_phases_of_two_features_in_one_dimension() -> None:
  W = torch.tensor([[[1.0, 0.0]], [[0.05, 1.1]], [[0.9, -0.9]], [[0.8, 0.8]]])
  assert geometry.phase(W) == ['not learned', 'dedicated', 'superposition', 'other']
