"""
Compare one experiment's geometry across seeds, from the runs' saved analyses.

Reads runs/<experiment>-<relu|linear>-seed<s>/analysis/ for every seed and writes
runs/compare-<experiment>/, which must not exist yet:
- intro: losses.csv, every seed's loss and represented count per sparsity, ReLU and linear,
  beside the baselines.
- polygons: shapes.csv, the shape every seed's ReLU model makes at every density.
- phase: cells.csv, how many seeds land in each phase per grid cell, and their agreement.
- dimensionality: curves.csv, dimensions per feature per seed, and features.csv, every
  represented feature's Dᵢ.
No seed is singled out: the comparison is the result.
"""

import csv
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import tyro

from sardines_mlp.experiments import Experiment
from sardines_mlp.geometry import PHASES
from sardines_mlp.utils import make_output_dir, save_json


@dataclass
class Config:
  experiment: Experiment
  seeds: list[int] = field(default_factory=lambda: [0, 1, 2, 3, 4])
  runs: Path = Path('runs')


def read_csv(path: Path) -> list[dict[str, str]]:
  with path.open() as f:
    return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
  with path.open('w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)


def instances(cfg: Config, model: str, seed: int) -> list[dict[str, str]]:
  return read_csv(cfg.runs / f'{cfg.experiment}-{model}-seed{seed}' / 'analysis' / 'instances.csv')


def intro(cfg: Config, out: Path) -> dict[str, Any]:
  relu = {s: instances(cfg, 'relu', s) for s in cfg.seeds}
  linear = {s: instances(cfg, 'linear', s) for s in cfg.seeds}
  rows = []
  for k, base in enumerate(relu[cfg.seeds[0]]):
    row: dict[str, Any] = {'density': base['density'], 'keep_top_m': base['keep_top_m'], 'zero_output': base['zero_output']}
    row['untrained_min'] = min(float(relu[s][k]['untrained']) for s in cfg.seeds)
    row['untrained_max'] = max(float(relu[s][k]['untrained']) for s in cfg.seeds)
    for s in cfg.seeds:
      row[f'relu_seed{s}'] = relu[s][k]['loss']
      row[f'relu_represented_seed{s}'] = relu[s][k]['represented']
    for s in cfg.seeds:
      row[f'linear_seed{s}'] = linear[s][k]['loss']
      row[f'linear_represented_seed{s}'] = linear[s][k]['represented']
    rows.append(row)
  write_csv(out / 'losses.csv', rows)
  return {
    'represented_relu': {s: [int(r['represented']) for r in relu[s]] for s in cfg.seeds},
    'represented_linear': {s: [int(r['represented']) for r in linear[s]] for s in cfg.seeds},
  }


def polygons(cfg: Config, out: Path) -> dict[str, Any]:
  rows = []
  for s in cfg.seeds:
    for r in instances(cfg, 'relu', s):
      rows.append({
        'seed': s, 'instance': r['instance'], 'density': r['density'], 'k': r['polygon_k'],
        'regular': r['polygon_regular'] == 'True', 'loss': r['loss'],
      })
  write_csv(out / 'shapes.csv', rows)
  densities = sorted({float(r['density']) for r in rows}, reverse=True)
  pentagons = {d: sum(r['regular'] and r['k'] == '5' for r in rows if float(r['density']) == d) for d in densities}
  return {'regular_pentagons_by_density': pentagons}


def phase(cfg: Config, out: Path) -> dict[str, Any]:
  runs = {s: instances(cfg, 'relu', s) for s in cfg.seeds}
  rows = []
  for k, base in enumerate(runs[cfg.seeds[0]]):
    counts = Counter(runs[s][k]['phase'] for s in cfg.seeds)
    majority, top = counts.most_common(1)[0]
    # The fit with the lowest loss, as the paper reports its phase diagram (best of several
    # fits). A selection over seeds, so it is shown beside the majority, never instead of it.
    best = min(cfg.seeds, key=lambda s: float(runs[s][k]['loss']))
    rows.append({
      'instance': k,
      'density': base['density'],
      'relative_importance': base['relative_importance'],
      **{p.replace(' ', '_'): counts.get(p, 0) for p in PHASES},
      'majority': majority,
      'agreement': top / len(cfg.seeds),
      'lowest_loss_seed': best,
      'lowest_loss_phase': runs[best][k]['phase'],
    })
  write_csv(out / 'cells.csv', rows)
  return {
    'cells': len(rows),
    'unanimous': sum(r['agreement'] == 1 for r in rows),
    'majority': dict(Counter(r['majority'] for r in rows)),
    'lowest_loss': dict(Counter(r['lowest_loss_phase'] for r in rows)),
  }


def dimensionality(cfg: Config, out: Path) -> dict[str, Any]:
  curves, feats = [], []
  for model in ('relu', 'linear'):
    for s in cfg.seeds:
      for r in instances(cfg, model, s):
        curves.append({
          'model': model, 'seed': s, 'density': r['density'],
          'dimensions_per_feature': r['dimensions_per_feature'], 'represented': r['represented'],
        })
      if model == 'relu':
        path = cfg.runs / f'{cfg.experiment}-relu-seed{s}' / 'analysis' / 'features.csv'
        feats += [
          {'seed': s, 'density': f['density'], 'dimensionality': f['dimensionality']}
          for f in read_csv(path)
          if f['represented'] == 'True'
        ]
  write_csv(out / 'curves.csv', curves)
  write_csv(out / 'features.csv', feats)
  return {'represented_features': len(feats)}


def main(cfg: Config) -> None:
  out = make_output_dir(cfg.runs / f'compare-{cfg.experiment}')
  compare = {'intro': intro, 'polygons': polygons, 'phase': phase, 'dimensionality': dimensionality}
  summary = compare[cfg.experiment](cfg, out)
  save_json(out / 'summary.json', {'experiment': cfg.experiment, 'seeds': cfg.seeds, **summary})
  print(f'{out}: {summary}')


if __name__ == '__main__':
  main(tyro.cli(Config))
