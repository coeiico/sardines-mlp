"""The sparse feature generator, the closed-form baselines and the four sweeps."""

import pytest
import torch

from sardines_mlp import experiments, features


def test_sparsity_and_range_match_the_definition() -> None:
  probability = torch.tensor([[1.0, 1.0], [0.3, 0.3], [0.01, 0.01]])
  x = features.sample(200_000, probability, torch.Generator().manual_seed(0))
  assert x.shape == (200_000, 3, 2)
  assert x.min() >= 0 and x.max() <= 1
  active = (x > 0).float().mean(0)
  assert torch.allclose(active, probability, atol=0.005)
  # Active values are U[0, 1]: mean ½.
  dense = x[:, 0]
  assert abs(dense.mean().item() - 0.5) < 0.005


def test_moments_match_the_closed_forms() -> None:
  probability = torch.tensor([[0.2, 0.7]])
  x = features.sample(400_000, probability, torch.Generator().manual_seed(1))[:, 0]
  assert torch.allclose(x.pow(2).mean(0), features.expected_square(probability)[0], atol=0.002)
  assert torch.allclose(x.var(0), features.variance(probability)[0], atol=0.002)


def test_same_seed_same_batch_and_seeds_differ() -> None:
  probability = torch.full((2, 4), 0.5)
  a = features.sample(8, probability, torch.Generator().manual_seed(3))
  b = features.sample(8, probability, torch.Generator().manual_seed(3))
  c = features.sample(8, probability, torch.Generator().manual_seed(4))
  assert torch.equal(a, b)
  assert not torch.equal(a, c)


def test_keep_top_m_drops_the_cheapest_features() -> None:
  # Uniform density ½: Var = ½/3 − ¼² = 5/48 per feature; importances 4, 3, 2, 1.
  probability = torch.full((1, 4), 0.5)
  importance = torch.tensor([[4.0, 3.0, 2.0, 1.0]])
  var = 1 / 6 - 1 / 16
  assert features.keep_top_m_loss(probability, importance, 2).item() == pytest.approx(3 * var)
  assert features.keep_top_m_loss(probability, importance, 4).item() == pytest.approx(0.0)
  assert features.zero_output_loss(probability, importance).item() == pytest.approx(10 / 6)


@pytest.mark.parametrize('name', ['intro', 'polygons', 'phase', 'dimensionality'])
def test_sweeps_are_consistent(name: experiments.Experiment) -> None:
  sweep = experiments.build(name)
  shape = (sweep.n_instances, sweep.n_features)
  assert sweep.feature_probability.shape == shape and sweep.importance.shape == shape
  assert sweep.n_hidden < sweep.n_features
  assert ((sweep.feature_probability > 0) & (sweep.feature_probability <= 1)).all()
  for values in sweep.coordinates.values():
    assert len(values) == sweep.n_instances


def test_sweeps_match_the_paper() -> None:
  intro = experiments.intro()
  assert (intro.n_features, intro.n_hidden) == (20, 5)
  assert torch.allclose(intro.importance[0, :3], torch.tensor([1.0, 0.7, 0.49]))
  polygons = experiments.polygons()
  assert (polygons.n_features, polygons.n_hidden) == (5, 2)
  assert min(abs(d - 0.05) for d in polygons.coordinates['density']) < 0.001
  phase = experiments.phase()
  assert min(phase.coordinates['relative_importance']) == pytest.approx(0.1)
  assert max(phase.coordinates['relative_importance']) == pytest.approx(10)
  assert min(phase.coordinates['density']) == pytest.approx(0.01)
  dim = experiments.dimensionality()
  assert (dim.n_features, dim.n_hidden) == (400, 30)
