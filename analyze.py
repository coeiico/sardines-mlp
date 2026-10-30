"""
Geometry, baselines and diagnostic plots for a saved sardines-mlp run.

Analysis reads saved weights, so it never needs to retrain. Everything goes to
runs/<run>/<name>/ (default `analysis`), which must not exist yet: per-instance and
per-feature CSVs, the experiment's figure data, a summary and diagnostic PNGs.
"""

import csv
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import tyro

from sardines_mlp import analysis, checkpoint, evaluation
from sardines_mlp.geometry import PHASES
from sardines_mlp.utils import code_state, get_device, make_output_dir, save_json


@dataclass
class Config:
  run: Path
  """Run directory, e.g. runs/polygons-relu-seed0."""
  name: str = 'analysis'
  """Output folder inside the run, which must not exist yet."""


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
  with path.open('w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)


def plot_intro(run: checkpoint.Run, rows: list[dict[str, Any]], path: Path) -> None:
  mats = analysis.gram(run)
  n = run.sweep.n_instances
  fig, axes = plt.subplots(2, n, figsize=(2.2 * n, 4.6), gridspec_kw={'height_ratios': [5, 1]})
  for k in range(n):
    axes[0, k].imshow(mats['wtw'][k], cmap='RdBu_r', vmin=-1, vmax=1)
    axes[0, k].set_title(f'1−S = {rows[k]["density"]:.3g}\n{rows[k]["represented"]} represented', fontsize=8)
    axes[1, k].imshow(mats['b'][k][None], cmap='RdBu_r', vmin=-1, vmax=1)
    for ax in axes[:, k]:
      ax.set_xticks([])
      ax.set_yticks([])
  counts = [r['represented'] for r in rows]
  fig.suptitle(f'WᵀW (top) and b: {counts[0]} features represented when dense, {counts[-1]} at the sparsest')
  fig.tight_layout()
  fig.savefig(path, dpi=120)
  plt.close(fig)


def plot_polygons(run: checkpoint.Run, rows: list[dict[str, Any]], path: Path) -> None:
  W = run.model.W.detach()
  n = run.sweep.n_instances
  cols = 7
  fig, axes = plt.subplots(math.ceil(n / cols), cols, figsize=(2 * cols, 2 * math.ceil(n / cols)))
  for k, ax in enumerate(axes.flat):
    ax.set_axis_off()
    if k >= n:
      continue
    for i in range(W.shape[2]):
      ax.plot([0, W[k, 0, i]], [0, W[k, 1, i]], color='k', lw=1)
      ax.plot(W[k, 0, i], W[k, 1, i], 'o', ms=3, color='C3')
    ax.set(xlim=(-1.5, 1.5), ylim=(-1.5, 1.5), aspect='equal')
    shape = f'{rows[k]["polygon_k"]}{"-gon" if rows[k]["polygon_regular"] else " irregular"}'
    ax.set_title(f'1−S = {rows[k]["density"]:.3g}: {shape}', fontsize=8)
  regular = [r for r in rows if r['polygon_regular'] and r['polygon_k'] == 5]
  first = f'a regular pentagon from 1−S = {max(r["density"] for r in regular):.3g}' if regular else 'no regular pentagon'
  fig.suptitle(f'W columns in the plane: {first}')
  fig.tight_layout()
  fig.savefig(path, dpi=120)
  plt.close(fig)


def plot_phase(rows: list[dict[str, Any]], path: Path) -> None:
  density = sorted({r['density'] for r in rows}, reverse=True)
  relative = sorted({r['relative_importance'] for r in rows})
  grid = np.zeros((len(density), len(relative)))
  for r in rows:
    grid[density.index(r['density']), relative.index(r['relative_importance'])] = PHASES.index(r['phase'])
  fig, ax = plt.subplots(figsize=(6, 5))
  image = ax.imshow(grid, cmap='viridis', vmin=0, vmax=len(PHASES) - 1, origin='upper', aspect='auto',
                    extent=(math.log10(relative[0]), math.log10(relative[-1]), math.log10(density[-1]), math.log10(density[0])))
  bar = fig.colorbar(image, ticks=range(len(PHASES)))
  bar.ax.set_yticklabels(PHASES)
  ax.set(xlabel='log10 relative importance of feature 2', ylabel='log10 feature density 1−S')
  share = sum(r['phase'] == 'superposition' for r in rows) / len(rows)
  ax.set_title(f'Feature 2 is in superposition in {share:.0%} of the grid')
  fig.tight_layout()
  fig.savefig(path, dpi=120)
  plt.close(fig)


def plot_dimensionality(rows: list[dict[str, Any]], feature_rows: list[dict[str, Any]], path: Path) -> None:
  fig, ax = plt.subplots(figsize=(8, 5))
  x = [1 / r['density'] for r in rows]
  ax.plot(x, [r['dimensions_per_feature'] for r in rows], color='k', lw=1.2, label='m / ‖W‖²_F')
  fx = [1 / f['density'] for f in feature_rows if f['represented']]
  fy = [f['dimensionality'] for f in feature_rows if f['represented']]
  ax.scatter(fx, fy, s=2, alpha=0.3, color='C3', label='Dᵢ of represented features')
  for level in (1, 3 / 4, 2 / 3, 1 / 2, 2 / 5, 3 / 8):
    ax.axhline(level, color='grey', lw=0.5, ls=':')
  ax.set(xscale='log', xlabel='1 / (1 − S)', ylabel='dimensions per feature', ylim=(0, 1.1))
  ax.legend()
  half = sum(abs(r['dimensions_per_feature'] - 0.5) < 0.02 for r in rows)
  ax.set_title(f'Dimensions per feature sits within 0.02 of ½ for {half} of {len(rows)} sparsities')
  fig.tight_layout()
  fig.savefig(path, dpi=120)
  plt.close(fig)


def main(cfg: Config) -> None:
  out = make_output_dir(cfg.run / cfg.name)
  run = checkpoint.load(cfg.run)
  train_cfg = run.record['config']
  device = get_device()
  run.model.to(device)
  # Rescored here on the training run's evaluation samples, so this analysis checks the
  # saved weights reproduce the loss train.py recorded.
  ev = train_cfg['eval']
  loss = evaluation.loss(run.model, run.sweep, ev['samples'], ev['chunk'], ev['seed'], device)
  recorded = torch.tensor(run.record['final']['loss'])
  run.model.cpu()

  rows = analysis.instance_rows(run, loss)
  features = analysis.feature_rows(run)
  write_csv(out / 'instances.csv', rows)
  write_csv(out / 'features.csv', features)
  experiment = train_cfg['experiment']
  linear = train_cfg['model']['activation'] == 'none'
  if experiment == 'intro':
    mats = analysis.gram(run)
    np.savez(out / 'wtw.npz', wtw=mats['wtw'].numpy(), b=mats['b'].numpy())
    plot_intro(run, rows, out / 'intro.png')
  elif experiment == 'polygons':
    write_csv(out / 'columns.csv', analysis.column_rows(run))
    plot_polygons(run, rows, out / 'polygons.png')
  elif experiment == 'phase':
    plot_phase(rows, out / 'phase.png')
  elif experiment == 'dimensionality':
    plot_dimensionality(rows, features, out / 'dimensionality.png')

  top_m = torch.tensor([r['keep_top_m'] for r in rows])
  summary = {
    'config': {**asdict(cfg), 'run': str(cfg.run)},
    'training_config': train_cfg,
    'code': code_state(out),
    'rescored_matches_recorded': bool(torch.allclose(loss, recorded, rtol=1e-4, atol=1e-7)) if device.type == run.record['environment']['device'].split(':')[0] else None,
    'instances': len(rows),
    'below_keep_top_m': int((loss < top_m * 0.99).sum()),
    'above_keep_top_m': int((loss > top_m * 1.01).sum()),
    'represented_max': max(r['represented'] for r in rows),
    'linear': linear,
  }
  if experiment == 'phase':
    summary['phases'] = {p: sum(r['phase'] == p for r in rows) for p in PHASES}
  if run.sweep.n_hidden == 2:
    summary['polygons'] = [(r['density'], r['polygon_k'], r['polygon_regular']) for r in rows]
  save_json(out / 'summary.json', summary)
  print(f'{cfg.run}: {summary["below_keep_top_m"]} of {len(rows)} instances below keep-top-m, '
        f'{summary["above_keep_top_m"]} above; rescored loss matches: {summary["rescored_matches_recorded"]}; wrote {out}')


if __name__ == '__main__':
  main(tyro.cli(Config))
