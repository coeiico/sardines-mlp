# sardines-mlp

**Toy models of superposition.** Reproducing toy models of superposition: how a network stores more features than it has dimensions.

Part of the **Latent Space Project**: 18 small PyTorch repos that test one idea, that neural networks encode concepts and strategy in their latent space, and that we can read and edit those representations. Everything trains on a single consumer GPU or a laptop.

| Roadmap | Phase | Depends on | Status |
| --- | --- | --- | --- |
| 5 of 18 | 2. Interpretability practice | Nothing | Not started (scaffolded) |

## Question

How does a network store more features than it has dimensions?

## Plan

- **Architecture:** x → ReLU(WᵀWx + b), with n features (5–20) squeezed into m dimensions (2–5), varying feature sparsity and importance.

## Experiments

- Reproduce the phase diagram of superposition versus sparsity.
- Plot W's columns in 2D to see the polygons.
- Measure each feature's dimensionality.

## Done when

The definition of done every Latent Space Project repo shares:

Apply this checklist with the project-specific requirements and exceptions in
[CLAUDE.md](CLAUDE.md#project-specific-scope).

- `just quick` runs end to end on CPU in under 5 minutes, as a smoke test.
- The full run fits on one consumer GPU (8–12 GB VRAM) and auto-detects cuda, mps or cpu.
- Each run saves weights, `metrics.json` and its plots to `runs/<run-name>/`.
- This README shows the headline result with plots, compares against baselines, and states negative results plainly.
- Tests cover data generators, tensor shapes and one tiny overfit check.

## Notes

- Trains in seconds on CPU. Good for fast iteration on plotting.

## Quickstart

Needs [uv](https://docs.astral.sh/uv/) and [just](https://just.systems/). Everything else, Python 3.12 included, comes from uv.

```sh
just setup              # uv sync, then install the pre-commit hooks
just quick              # CPU smoke test, under 5 minutes
just train              # full run; flags pass through, e.g. --run-name wd1.0-seed0
just analyze <run>      # baselines, probes, interventions and plots
just test               # pytest
just check              # ruff and basedpyright, as in CI
```

Copy `.env.example` to `.env` (gitignored) for `HF_TOKEN` and `HF_USER`.

## Layout

```
sardines-mlp/
├── README.md                goal, plan, results with plots
├── CLAUDE.md                instructions for coding agents; AGENTS.md points here
├── .agents/                 instructions/ (Python formatting, commits) and plans/
├── pyproject.toml           metadata, dependencies, tool settings
├── uv.lock                  committed lockfile
├── .python-version          pinned by uv
├── justfile                 every routine command
├── .pre-commit-config.yaml
├── .github/workflows/ci.yml
├── src/sardines_mlp/
│   └── model.py, data.py, analysis.py, utils.py
├── train.py                 CLI; --quick runs the CPU smoke test
├── predict.py               load weights and run inference
├── analyze.py               baselines, probes, interventions, plots
├── notebooks/               marimo notebooks for exploration
├── tests/
└── runs/                    outputs, gitignored
```

## Results

_No results yet._ The headline result with plots, the comparison against baselines, and any negative results go here.

## Sources

- Elhage et al. 2022, Toy models of superposition

## License

MIT
