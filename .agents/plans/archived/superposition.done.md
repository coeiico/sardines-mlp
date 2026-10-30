# Plan — Superposition

Build sardines-mlp from its scaffold: a reproduction of Anthropic's *Toy Models of
Superposition* (Elhage et al. 2022). We train the toy ReLU model x' = ReLU(WᵀWx + b) on
synthetic sparse features, then run four experiments: the WᵀW demo, the 2D polygons, the
phase diagram and feature dimensionality. A linear model and analytic baselines go
alongside. This is the series' quick win before blindfold-tfm. It trains on a laptop CPU in
minutes, and it is the first repo where the geometry of a representation is the result.

Written 2026-10-30 against `f93c62d`.
Log: @.agents/plans/archived/superposition.done.log.md

## What we're reproducing

- **The model.** n features are compressed into m < n hidden dimensions and read back out
  with the same matrix: h = Wx, x' = ReLU(Wᵀh + b), with W of shape m × n.
- **The data.** Each feature xᵢ is 0 with probability S (the sparsity), otherwise drawn
  from U[0, 1], independently.
- **The loss.** Importance-weighted reconstruction: L = E[Σᵢ Iᵢ (xᵢ − x'ᵢ)²].
- **The claim.** When features are dense, the model keeps the m most important ones in
  orthogonal directions and drops the rest, which is what PCA would do. When features are
  sparse, it packs more than m features into m dimensions. They sit in interference
  patterns (antipodal pairs, triangles, pentagons), and the ReLU and a negative bias filter
  out the interference. The linear model can never do this.

## Design decisions

- **Many models trained as one.** Each sweep trains a batch of independent "instances"
  together. W has shape (instances, m, n), b has shape (instances, n), and every instance
  has its own feature probability and importance vector. The loss is summed over
  instances. Adam updates each element independently, so the instances stay independent
  models. A test checks this. This is how the paper does its sweeps, and it makes each
  experiment one short run instead of hundreds.
- **Variants are config, not code paths.** `--model.activation relu|none` picks between the
  toy model and the linear baseline. `--experiment intro|polygons|phase|dimensionality`
  picks a preset instance grid, and n, m, the sparsities and the importance come from that
  preset. Each preset is a small function in `experiments.py` that returns per-instance
  tensors. Each one can still be overridden from the CLI.
- **Evaluation on fresh samples.** Training draws from a generator seeded by `seed`.
  Evaluation uses 65,536 fresh samples from a separate generator with a fixed eval seed,
  identical for every run. There is no dataset to leak, so this is the held-out set.
- **Development seed vs reported seeds.** Seed 100 is the development seed. It is used to
  choose steps, learning rate and batch size, and to check convergence. Reported results
  are seeds 0–4, all of them, compared to each other. No seed is picked for having a nicer
  polygon (CLAUDE.md, project-specific scope).
- **Optimiser.** Adam, lr 1e-3, batch 1,024, 10,000 steps with a constant rate, and Xavier
  normal init for W, as in Anthropic's companion Colab. No weight decay: it would shrink
  ‖Wᵢ‖ and blur the "represented or not" line. The development seed checks that 10,000
  steps is enough. The phase diagram may need more steps or a bigger batch, because at a
  density of 0.01 a batch holds only about 10 active examples per feature.
- **The model stays simple.** It has no cache and no patch interface. Nothing here
  intervenes on hidden activations, so the line in the `model.py` docstring that promises
  them goes (CLAUDE.md: do not impose an unused interface).
- **Device.** Default `auto`. The local RTX 3060 is used when present, and `--quick` and the
  tests stay on the CPU. No Modal: CLAUDE.md already rules the cloud out for this repo.

## The four experiments

| Preset | n → m | Importance | Sweep (one instance each) | What it shows |
| --- | --- | --- | --- | --- |
| `intro` | 20 → 5 | 0.7^i | S ∈ {0, 0.7, 0.9, 0.97, 0.99, 0.997, 0.999} | WᵀW and b, ‖Wᵢ‖ and superposition per feature, for ReLU and linear |
| `polygons` | 5 → 2 | 0.9^i | 20 densities, log-spaced from 1 to 0.05 | W's columns in 2D, from 2 orthogonal features to a pentagon |
| `phase` | 2 → 1 | 1 for feature 1, r for feature 2 | r × density grid, 24 × 24 log-spaced (r 0.1–10, density 0.01–1) | where feature 2 is dropped, gets the dimension, or shares it antipodally |
| `dimensionality` | 200 → 20 | uniform | 40 densities, 1/(1−S) log-spaced from 1 to 1,000 | dimensions per feature, and per-feature Dᵢ on the plateaus |

- The `intro` importance (0.7^i), the `polygons` importance (0.9^i) and the grid ranges are
  my reading of the paper and its Colab. Phase 1 checks them against the paper before any
  run counts.
- **`dimensionality` goes beyond the README's sizes (n 5–20, m 2–5).** The sticky plateaus
  (1, 3/4, 2/3, 1/2, 2/5, 3/8) only show up with many features. At 200 × 20 that is 4,000
  weights per instance, which still trains on the GPU in about a minute. The README plan
  is updated to say so.

## Measurements

All defined once, in `geometry.py`:

- **Weighted reconstruction loss** on the eval samples, per instance.
- **‖Wᵢ‖**, the norm of feature i's column. A feature counts as **represented** when
  ‖Wᵢ‖ > 0.5.
- **Superposition of feature i**: Σ_{j≠i} (Ŵᵢ · Wⱼ)², where Ŵᵢ = Wᵢ/‖Wᵢ‖. This is how much
  of the other features lands in feature i's direction.
- **Feature dimensionality**: Dᵢ = ‖Wᵢ‖² / Σⱼ (Ŵᵢ · Wⱼ)². An antipodal pair gives ½ each, a
  pentagon ⅖ each and an orthogonal feature 1.
- **Dimensions per feature**: m / ‖W‖²_F.
- **Phase of an n = 2, m = 1 instance**, read from W and with thresholds stated in code:
  - feature 2 not represented,
  - feature 2 represented and feature 1 not,
  - both represented with opposite signs (antipodal superposition).

## Baselines

- **The linear model** (`--model.activation none`), trained on the same grids and seeds. We
  expect it to learn the top m features orthogonally at every sparsity, with no
  superposition.
- **The analytic "keep the top m" loss.** Reconstruct the m most important features
  perfectly and output the best constant (E[xᵢ] = (1 − S)/2) for the rest. This is computed
  in closed form, so it is not trained and needs no seeds. The ReLU model beating it is the
  evidence of superposition.
- **The untrained model** (Xavier init, b = 0) and **the zero output**, both scored on the
  same eval samples.

## Phases

### 1. Foundations

- Replace `utils.py` with the shared version from deja-vu-tfm, which refuses an existing
  run folder and records the code state, the environment and a source snapshot. Keep its
  "copied from" note.
- Write `features.py`, which replaces the empty `data.py`. It samples sparse features for a
  batch of instances, each with its own feature probability, from a given
  `torch.Generator`. It also builds the importance vectors.
- Write `model.py`: `ToyModel(n_instances, n_features, n_hidden, activation)` with tied W,
  plus the weighted loss as its own function. Annotate shapes with jaxtyping.
- Write `experiments.py`, the four presets. Check their settings against the paper and
  record anything that differs in the log.
- Write tests, replacing `test_scaffold.py`'s placeholders:
  - **Features:** the share of zeros is within tolerance of S, values lie in [0, 1], the
    same seed gives the same batch, and the eval generator is independent of the training
    seed.
  - **Loss:** checked against a value worked out by hand on a 1-instance, 3-feature input.
  - **Tied gradients:** the autograd gradient of W matches finite differences. It also
    equals the sum of the encoder and decoder gradients from an untied copy where
    W_enc = W_dec = W.
  - **Independence:** changing instance 1's inputs leaves instance 0's gradient unchanged.
  - **Overfit:** 1 instance, n = m = 2, dense, reaches a loss near 0.
  - **Shapes:** checked through jaxtyping and beartype.

### 2. Training

- Write `train.py`:
  - **Config:** `experiment`, `model`, `optim`, `seed`, `device`, `quick` and `eval_every`.
  - **Loop:** a visible forward, zero_grad, backward and step. It stops on a NaN or
    infinite loss.
  - **Outputs:** `model.safetensors` and the per-instance settings (feature probability
    and importance, as a safetensors file beside the weights). `metrics.json` holds the
    config, seeds, eval seed, code state and environment, the per-instance eval loss over
    training, and the final losses next to the baselines. It also saves a loss-curve PNG
    whose title states the finding.
- `--quick`: the `polygons` preset with fewer instances and 2,000 steps, on the CPU, in well
  under a minute.
- Run the development seed (100) for each preset to check that loss has flattened by the
  last 10% of steps. Adjust steps or batch for `phase` if it hasn't, and log the choice.

### 3. Analysis

- Write `geometry.py`, the measurements above. Its tests use hand-built W:
  - an antipodal pair gives Dᵢ = ½,
  - a regular pentagon gives ⅖,
  - an orthogonal basis gives 1,
  - each of the three n = 2, m = 1 phases is classified correctly.
- Write `analyze.py`, which loads a run and writes `runs/<run>/analysis/`, refusing to
  overwrite it. It contains the per-instance geometry CSV, the per-feature CSV, the
  per-experiment figure data (CSV, plus `.npz` for the WᵀW grids) and diagnostic PNGs.
- Write `predict.py`: load a run, reconstruct a few fresh samples, and print inputs next
  to outputs per instance. This is a sanity view, not an experiment.
- Test that a saved and reloaded run gives identical outputs.

### 4. Runs

- Train every preset, ReLU and linear, for seeds 0–4: 4 presets × 2 models × 5 seeds = 40
  runs. Each run is one process holding all its instances. On the RTX 3060 the whole set
  should take well under an hour, mostly the `dimensionality` and `phase` runs. Then
  analyse each run.
- Compare geometry across seeds. For `polygons`, tabulate per seed and density how many
  features are represented and which shape they form. For `phase`, report per cell how many
  seeds agree. Disagreements are reported, not resolved by picking a seed.

### 5. Figures and write-up

- Build four README figures in the engraving style (`figure-design.md`), from
  `figures/data/` files copied out of the analysis folders:
  - `intro.png`: WᵀW grids (SIGNED) and b for linear vs ReLU at a few sparsities, with
    ‖Wᵢ‖ bars marked by superposition.
  - `polygons.png`: W's columns as arrows in 2D, one small panel per density. Seed 0 is
    drawn in ink and seeds 1–4 as faint thin lines on the same axes, so a difference
    between seeds shows.
  - `phase.png`: the 24 × 24 grid. The three phases are told apart by hatching direction
    and blank, with superposition in spot. Cells where seeds disagree are marked.
  - `dimensionality.png`: dimensions per feature against 1/(1−S), each seed a thin line,
    with per-feature Dᵢ as small open marks and level lines at 1, ¾, ⅔, ½, ⅖ and ⅜.
- README:
  - the results first, with a loss table (ReLU vs linear vs keep-top-m vs untrained, at
    each `intro` sparsity),
  - a "How it works" section on superposition, interference and why the ReLU and the
    negative bias matter,
  - negative results stated plainly,
  - Status set to done.
- Update `PROGRESS.md` in the project folder.

## Done when

- The general checklist in README.md and CLAUDE.md holds.
- `intro` shows the paper's picture: at S = 0 both models keep about 5 features. As S
  rises, the ReLU model represents more than 5 with growing superposition, while the
  linear model stays at 5.
- `polygons` reaches a pentagon (or a clear polygon with all 5 features represented) at the
  sparsest densities in most of seeds 0–4. The README states how many.
- `phase` shows the three regions in the arrangement the paper reports: superposition at
  low density and comparable importance.
- `dimensionality` shows at least the ½ and ⅖ plateaus, or the README says plainly that it
  doesn't.

## Stretch, only if asked

- The n = 3, m = 2 phase diagram.
- A theoretical n = 2, m = 1 phase diagram from the closed-form loss of each candidate
  solution, drawn against the empirical one.
- Correlated or anti-correlated features (paper §7).
