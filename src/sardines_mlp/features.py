"""
Synthetic sparse features, the toy models' only data.

Each feature xᵢ is 0 with probability S (the sparsity) and otherwise drawn from U[0, 1],
independently of every other feature (Elhage et al. 2022, "Toy Models of Superposition").
Every instance in a sweep has its own feature probability 1 − S and its own importances, so
one batch feeds many independent models at once.
"""

import torch
from jaxtyping import Float
from torch import Tensor


def sample(
  batch_size: int,
  feature_probability: Float[Tensor, 'instances features'],
  generator: torch.Generator,
) -> Float[Tensor, 'batch instances features']:
  """Draw a batch of sparse feature vectors for every instance."""
  shape = (batch_size, *feature_probability.shape)
  device = feature_probability.device
  values = torch.rand(shape, generator=generator, device=device)
  active = torch.rand(shape, generator=generator, device=device) < feature_probability
  return torch.where(active, values, 0.0)


def expected_square(feature_probability: Tensor) -> Tensor:
  """E[xᵢ²] = (1 − S) · E[u²] with u ~ U[0, 1], which is (1 − S)/3."""
  return feature_probability / 3


def variance(feature_probability: Tensor) -> Tensor:
  """Var[xᵢ] = E[xᵢ²] − E[xᵢ]², with E[xᵢ] = (1 − S)/2."""
  return feature_probability / 3 - (feature_probability / 2) ** 2


def zero_output_loss(
  feature_probability: Float[Tensor, 'instances features'],
  importance: Float[Tensor, 'instances features'],
) -> Float[Tensor, ' instances']:
  """The expected loss of a model that outputs 0 for every feature."""
  return (importance * expected_square(feature_probability)).sum(-1)


def keep_top_m_loss(
  feature_probability: Float[Tensor, 'instances features'],
  importance: Float[Tensor, 'instances features'],
  m: int,
) -> Float[Tensor, ' instances']:
  """
  The expected loss of reconstructing m features perfectly and dropping the rest.

  A dropped feature is best replaced by its mean, (1 − S)/2, which the bias can supply,
  so it costs Iᵢ · Var[xᵢ]. Keeping the m features with the largest Iᵢ · Var[xᵢ] is the
  best the linear model can do (it is PCA on importance-weighted features), so a ReLU model
  that beats this is representing more than m features.
  """
  cost = importance * variance(feature_probability)
  n = cost.shape[-1]
  dropped = cost.sort(dim=-1).values[..., : n - m]  # the n − m cheapest to drop
  return dropped.sum(-1)
