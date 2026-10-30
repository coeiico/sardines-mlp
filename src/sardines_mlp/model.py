"""
The toy model: n features squeezed into m dimensions and read back out with the same matrix.

  h = Wx,  x' = ReLU(Wᵀh + b),  W of shape m × n

`activation='none'` drops the ReLU and gives the linear baseline, which cannot use
superposition. A sweep trains many instances side by side: W holds one m × n matrix per
instance, and instance k only ever sees its own slice of each batch.
"""

import math
from typing import Literal

import torch
from jaxtyping import Float
from torch import Tensor, nn

Activation = Literal['relu', 'none']


class ToyModel(nn.Module):
  def __init__(self, n_instances: int, n_features: int, n_hidden: int, activation: Activation = 'relu') -> None:
    super().__init__()
    self.activation = activation
    # Xavier normal for each m × n matrix, std √(2 / (m + n)). nn.init.xavier_normal_ on the
    # stacked 3D tensor would count the instances into its fan-in and fan-out.
    std = math.sqrt(2 / (n_hidden + n_features))
    self.W = nn.Parameter(torch.randn(n_instances, n_hidden, n_features) * std)
    self.b = nn.Parameter(torch.zeros(n_instances, n_features))

  def hidden(self, x: Float[Tensor, 'batch instances features']) -> Float[Tensor, 'batch instances hidden']:
    return torch.einsum('bin,imn->bim', x, self.W)

  def forward(self, x: Float[Tensor, 'batch instances features']) -> Float[Tensor, 'batch instances features']:
    h = self.hidden(x)
    # The same W reads the features back out (tied weights): Wᵢ, column i of W, is then
    # unambiguously the direction that represents feature i.
    out = torch.einsum('bim,imn->bin', h, self.W) + self.b
    return torch.relu(out) if self.activation == 'relu' else out


def weighted_loss(
  x: Float[Tensor, 'batch instances features'],
  x_hat: Float[Tensor, 'batch instances features'],
  importance: Float[Tensor, 'instances features'],
) -> Float[Tensor, ' instances']:
  """Importance-weighted squared error, Σᵢ Iᵢ (xᵢ − x'ᵢ)², averaged over the batch, per instance."""
  return (importance * (x - x_hat) ** 2).sum(-1).mean(0)
