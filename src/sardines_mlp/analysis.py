"""
Turn a trained sweep into tables: per instance, per feature, and per experiment.

Experiments:
- intro: WᵀW and b for every sparsity, with ‖Wᵢ‖ and superposition per feature.
- polygons: W's columns in the plane and the shape they make.
- phase: what each n = 2, m = 1 instance did with the extra feature.
- dimensionality: dimensions per feature and every feature's Dᵢ.
"""

from typing import Any

import torch

from sardines_mlp import evaluation, geometry
from sardines_mlp.checkpoint import Run


def instance_rows(run: Run, loss: torch.Tensor) -> list[dict[str, Any]]:
  W = run.model.W.detach()
  sweep = run.sweep
  base = evaluation.baselines(sweep)
  rows: list[dict[str, Any]] = []
  dims = geometry.dimensions_per_feature(W)
  count = geometry.represented(W).sum(-1)
  phases = geometry.phase(W) if (sweep.n_features, sweep.n_hidden) == (2, 1) else None
  for k in range(sweep.n_instances):
    row: dict[str, Any] = {'instance': k, **{name: values[k] for name, values in sweep.coordinates.items()}}
    row.update(
      loss=loss[k].item(),
      keep_top_m=base['keep_top_m'][k],
      zero_output=base['zero_output'][k],
      untrained=run.record['final']['untrained_loss'][k],
      represented=int(count[k]),
      dimensions_per_feature=dims[k].item(),
    )
    if phases is not None:
      row['phase'] = phases[k]
      row['w1'], row['w2'], row['b1'], row['b2'] = W[k, 0, 0].item(), W[k, 0, 1].item(), *run.model.b[k].tolist()
    if sweep.n_hidden == 2:
      row['polygon_k'], row['polygon_regular'] = geometry.polygon(W[k])
    rows.append(row)
  return rows


def feature_rows(run: Run) -> list[dict[str, Any]]:
  W = run.model.W.detach()
  b = run.model.b.detach()
  sweep = run.sweep
  norm, sup, dim = geometry.norms(W), geometry.superposition(W), geometry.feature_dimensionality(W)
  rows: list[dict[str, Any]] = []
  for k in range(sweep.n_instances):
    for i in range(sweep.n_features):
      rows.append({
        'instance': k,
        'feature': i,
        'density': sweep.feature_probability[k, i].item(),
        'importance': sweep.importance[k, i].item(),
        'norm': norm[k, i].item(),
        'superposition': sup[k, i].item(),
        'dimensionality': dim[k, i].item(),
        'bias': b[k, i].item(),
        'represented': bool(norm[k, i] > geometry.REPRESENTED),
      })
  return rows


def column_rows(run: Run) -> list[dict[str, Any]]:
  """W's columns as points in the plane, for an m = 2 sweep."""
  W = run.model.W.detach()
  return [
    {'instance': k, 'feature': i, 'x': W[k, 0, i].item(), 'y': W[k, 1, i].item(), 'bias': run.model.b[k, i].item()}
    for k in range(W.shape[0])
    for i in range(W.shape[2])
  ]


def gram(run: Run) -> dict[str, torch.Tensor]:
  """WᵀW and b per instance, the intro figure's matrices."""
  W = run.model.W.detach()
  return {'wtw': torch.einsum('imf,img->ifg', W, W), 'b': run.model.b.detach()}
