set dotenv-load

hf_user := env('HF_USER', 'your-hf-username')
repo := file_name(justfile_directory())

default:
  @just --list

# Install dependencies and the pre-commit hooks
setup:
  uv sync
  uv run pre-commit install

# CPU smoke test, under 5 minutes
quick:
  uv run train.py --quick --run-name quick

# Full training run; pass flags through, e.g. just train --run-name wd1.0-seed0
train *args:
  uv run train.py {{args}}

# Run any recipe in a detached tmux session, logged to runs/logs/, e.g. just bg train --run-name wd1.0-seed0
bg recipe *args:
  #!/usr/bin/env bash
  set -euo pipefail
  mkdir -p runs/logs
  session="{{repo}}-{{recipe}}-$(date +%Y%m%d-%H%M%S)"
  tmux new-session -d -s "$session" "PYTHONUNBUFFERED=1 just {{recipe}} {{args}} 2>&1 | tee runs/logs/$session.log; echo; echo 'Finished. Press enter to close.'; read"
  echo "tmux session $session: tmux attach -t $session (detach with ctrl-b d), log runs/logs/$session.log"

# Train, analyse and compare every reported run (sweep.sh); start it with just bg sweep
sweep:
  bash sweep.sh

# Cross-seed comparison of one experiment's analysed runs, e.g. just compare polygons
compare experiment:
  uv run compare.py --experiment {{experiment}}

# Inference on a saved run
predict run *args:
  uv run predict.py --run runs/{{run}} {{args}}

# Baselines, probes, interventions and plots for a saved run
analyze run:
  uv run analyze.py --run runs/{{run}}

# Copy the README figures' numbers out of the runs into figures/data/
figure-data:
  cp runs/compare-intro/losses.csv figures/data/intro-losses.csv
  cp runs/intro-relu-seed0/analysis/wtw.npz figures/data/intro-wtw-relu-seed0.npz
  cp runs/intro-linear-seed0/analysis/wtw.npz figures/data/intro-wtw-linear-seed0.npz
  cp runs/compare-polygons/shapes.csv figures/data/polygons-shapes.csv
  for s in 0 1 2 3 4; do cp runs/polygons-relu-seed$s/analysis/columns.csv figures/data/polygons-columns-seed$s.csv; done
  cp runs/compare-phase/cells.csv figures/data/phase-cells.csv
  cp runs/compare-dimensionality/curves.csv figures/data/dimensionality-curves.csv
  cp runs/compare-dimensionality/features.csv figures/data/dimensionality-features.csv

# Build the README figures: figures/src/*.tex -> figures/*.png (.agents/instructions/figure-design.md)
figures:
  #!/usr/bin/env bash
  set -euo pipefail
  cd figures
  mkdir -p build
  shopt -s nullglob
  for script in src/*.py; do uv run python "$script"; done
  for source in src/*.tex; do
    name=$(basename "$source" .tex)
    if ! TEXINPUTS='tex//:' latexmk -lualatex -interaction=nonstopmode -halt-on-error -outdir=build "$source" > "build/$name.latexmk.log" 2>&1; then
      tail -n 30 "build/$name.log"
      exit 1
    fi
    pdftoppm -r 300 -png -singlefile "build/$name.pdf" "$name"
    echo "figures/$name.png"
  done

test:
  uv run pytest

# Lint and type check, as in CI (no ruff format; see .agents/instructions/python-formatting.md)
check:
  uv run ruff check .
  uv run basedpyright

# Apply Ruff's safe fixes (import order, quotes, docstring layout)
fix:
  uv run ruff check --fix .

# Exploration notebooks (marimo, saved as plain .py files)
notebook name='explore':
  uv run marimo edit notebooks/{{name}}.py

# Upload a run's weights, metrics and plots to the Hugging Face Hub, into a folder named after the run
push run:
  uv run hf upload {{hf_user}}/{{repo}} runs/{{run}} {{run}}

# Run a sweep from cloud.py on Modal, one GPU per run, e.g. just cloud seeds (.agents/instructions/cloud.md)
cloud sweep *args:
  uv run modal run --detach cloud.py::sweep --name {{sweep}} {{args}}

# One run on Modal, e.g. just cloud-train lr3e-3-seed0 '--optim.lr 3e-3'
cloud-train run flags='' *args:
  uv run modal run --detach cloud.py::single --name {{run}} --flags '{{flags}}' {{args}}

# Copy a run from the Modal Volume into runs/<run>/
cloud-get run:
  mkdir -p runs
  uv run modal volume get {{repo}}-runs {{run}} runs
