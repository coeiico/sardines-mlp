"""The toy model's loss, its tied-weight gradients, instance independence and a learning check."""

import pytest
import torch

from sardines_mlp import features
from sardines_mlp.model import ToyModel, weighted_loss


def test_shapes_and_linear_variant() -> None:
  model = ToyModel(3, 5, 2, 'none')
  x = torch.rand(4, 3, 5)
  assert model.hidden(x).shape == (4, 3, 2)
  out = model(x)
  assert out.shape == (4, 3, 5)
  # The linear model can output negatives, which the ReLU model cannot.
  with torch.no_grad():
    model.b.fill_(-10.0)
  assert (model(x) < 0).any()
  model.activation = 'relu'
  assert (model(x) >= 0).all()


def test_weighted_loss_by_hand() -> None:
  # One instance, three features, a batch of two.
  x = torch.tensor([[[1.0, 0.0, 0.5]], [[0.0, 1.0, 0.0]]])
  x_hat = torch.tensor([[[0.5, 0.0, 0.5]], [[0.0, 0.0, 1.0]]])
  importance = torch.tensor([[1.0, 2.0, 4.0]])
  # Example 1: 1·0.25 = 0.25. Example 2: 2·1 + 4·1 = 6. Mean 3.125.
  assert weighted_loss(x, x_hat, importance).tolist() == [pytest.approx(3.125)]


def test_forward_by_hand() -> None:
  model = ToyModel(1, 2, 1)
  with torch.no_grad():
    model.W.copy_(torch.tensor([[[1.0, -1.0]]]))
    model.b.copy_(torch.tensor([[-0.25, 0.0]]))
  x = torch.tensor([[[0.75, 0.0]], [[0.25, 0.5]]])
  # h = 0.75 → Wᵀh + b = [0.5, −0.75] → [0.5, 0]. h = −0.25 → [−0.5, 0.25] → [0, 0.25].
  expected = torch.tensor([[[0.5, 0.0]], [[0.0, 0.25]]])
  assert torch.allclose(model(x), expected)


def _loss(model: ToyModel, x: torch.Tensor, importance: torch.Tensor) -> torch.Tensor:
  return weighted_loss(x, model(x), importance).sum()


def test_tied_gradient_is_encoder_plus_decoder_gradient() -> None:
  torch.manual_seed(0)
  model = ToyModel(2, 4, 3, 'relu').double()
  with torch.no_grad():
    model.b.fill_(0.1)  # keep most outputs away from the ReLU's kink
  x = torch.rand(16, 2, 4, dtype=torch.float64)
  importance = torch.rand(2, 4, dtype=torch.float64)
  _loss(model, x, importance).backward()
  assert model.W.grad is not None
  tied = model.W.grad.clone()

  # The same computation with separate encoder and decoder copies of W.
  enc = model.W.detach().clone().requires_grad_()
  dec = model.W.detach().clone().requires_grad_()
  h = torch.einsum('bin,imn->bim', x, enc)
  out = torch.relu(torch.einsum('bim,imn->bin', h, dec) + model.b.detach())
  weighted_loss(x, out, importance).sum().backward()
  assert enc.grad is not None and dec.grad is not None
  assert torch.allclose(tied, enc.grad + dec.grad, atol=1e-12)
  # Both uses contribute: neither half alone is the whole gradient.
  assert enc.grad.abs().sum() > 0 and dec.grad.abs().sum() > 0


def test_tied_gradient_matches_finite_differences() -> None:
  torch.manual_seed(1)
  model = ToyModel(1, 3, 2, 'relu').double()
  with torch.no_grad():
    model.b.fill_(0.2)
  x = torch.rand(8, 1, 3, dtype=torch.float64)
  importance = torch.tensor([[1.0, 0.7, 0.49]], dtype=torch.float64)
  assert torch.autograd.gradcheck(
    lambda W: weighted_loss(x, torch.relu(torch.einsum('bim,imn->bin', torch.einsum('bin,imn->bim', x, W), W) + model.b), importance),
    (model.W.detach().clone().requires_grad_(),),
  )


def test_instances_are_independent() -> None:
  torch.manual_seed(2)
  model = ToyModel(2, 4, 2)
  importance = torch.ones(2, 4)
  x = torch.rand(16, 2, 4)
  _loss(model, x, importance).backward()
  assert model.W.grad is not None
  grad0 = model.W.grad[0].clone()
  model.zero_grad()
  y = x.clone()
  y[:, 1] = torch.rand(16, 4)  # change only instance 1's inputs
  _loss(model, y, importance).backward()
  assert model.W.grad is not None
  assert torch.equal(model.W.grad[0], grad0)


def test_dense_identity_is_learned() -> None:
  # n = m = 2 has room for both features, so the loss should go to about zero.
  torch.manual_seed(0)
  model = ToyModel(1, 2, 2)
  probability = torch.ones(1, 2)
  importance = torch.ones(1, 2)
  optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
  generator = torch.Generator().manual_seed(0)
  for _ in range(1500):
    x = features.sample(256, probability, generator)
    loss = weighted_loss(x, model(x), importance).sum()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
  x = features.sample(4096, probability, torch.Generator().manual_seed(9))
  with torch.no_grad():
    assert weighted_loss(x, model(x), importance).item() < 1e-3
