# Log — Superposition

Plan: @.agents/plans/archived/superposition.done.md

## Status

Done (2026-10-31). Archived with its plan.

## Open questions

## Entries

### 2026-10-30 — Start (`f93c62d`)

The owner approved the plan and asked to run everything locally on the RTX 3060 and the 24 CPU
cores, as many runs in parallel as the machine allows, instead of on Modal. Phase 4 therefore
launches the runs as concurrent local processes; the plan already keeps them local, so this is
how it is executed, not a change to it.

### 2026-10-30 — Phase 1: presets checked against the paper (`f93c62d`)

Read the arXiv PDF (2209.10652) as text. What it states, against the plan's table:
- `intro`: n = 20, m = 5, Iᵢ = 0.7ⁱ. Matches. The sparsity values are only in the figure,
  so the plan's list stands.
- `polygons`: the regular pentagon is n = 5 in m = 2 with **uniform importance** (I = 1) at
  density 0.05 ("Perturbing a single feature"). The plan said 0.9ⁱ, which would deform the
  pentagon. **Changed to uniform importance.** The density grid now runs from 1 to 0.01 in
  21 log steps, so 0.05 is on it.
- `phase`: feature 1 importance 1, the extra feature 0.1 to 10, density 1.0 to 0.01.
  Matches. The paper trains ten models per point, discards the worst and averages. We report
  agreement across our five seeds instead.
- `dimensionality`: the paper uses **n = 400, m = 30**, uniform importance. The plan said
  200 → 20 (from my memory of the Colab). **Changed to 400 → 30**, to match the paper; it
  still trains in minutes.
- Dᵢ and the sticky values (¾, ⅔, ½, ⅖, ⅜) match the paper's definitions.
- The paper gives no optimiser settings. The plan's Adam, lr 1e-3, batch 1,024 stand.
- The paper warns that 2D models are hard to optimise and it took the best of several fits.
  We don't select: every seed is reported.

Both changes follow the plan's own phase 1 step (check presets against the paper), but they
change the plan's table, so they are listed under Open questions for the owner to confirm.

Smaller departures from the plan:
- Presets are not overridable from the CLI (only steps, batch, lr and so on are). Nothing
  needs it yet.
- Evaluation samples are regenerated per chunk from a generator on the run's device,
  instead of a fixed CPU set: 65,536 samples of 40 × 400 features would be 4 GB. Samples are
  therefore identical across runs on the same device type, which `metrics.json` records.
- `evaluation.py` (scoring and closed-form baselines) and `checkpoint.py` (save and load) are
  modules of their own; `data.py` became `features.py`.
- Xavier init is computed per m × n matrix: `nn.init.xavier_normal_` on the stacked 3D
  tensor would count the instances into its fan-in.

### 2026-10-30 — Phase 2: development seed (`f93c62d`, uncommitted changes)

Seed 100, every preset, ReLU and linear, 10,000 steps on the GPU. The ReLU model beats the
keep-top-m loss at every sparse setting, the linear model sits on it. The convergence check
failed for two presets: the sparsest `intro` instance still fell 4.7% in the last 10% of
steps, and several `phase` instances found superposition late (one fell 17×).

At 30,000 steps (`dev30k-*`): every instance of `intro`, `polygons` and `dimensionality`
moved less than 1% in the last 10%. In `phase` 5 of 576 still did, up to 24%. These are
instances still switching between solutions, the 1D optimisation difficulty the paper
describes. **Decision: 30,000 steps for every preset**, set as the default. The 5
late-movers are left as they are and will show up as seed disagreement.

Dev-seed observations worth keeping:
- `polygons`: 4 features in a regular square from 1−S ≈ 0.5, a pentagon at 0.25–0.16, back
  to 4 at 0.13–0.08, then a pentagon from 0.063 down. The dip is a local minimum, which
  the paper also warns about.
- `dimensionality`: a clear ½ plateau for 1/(1−S) between about 2 and 4. Dᵢ clusters near ¾
  and ⅔ at the dense end, and a band near ⅖ just past the plateau. From 1/(1−S) ≈ 100 all
  400 features are represented, so the curve floors at about 30/400.
- `phase` (one seed): superposition in 57% of the grid, not learned for small r at high
  density, dedicated for large r. There is a noisy diagonal of "not learned" at r ≈ 0.3–1
  down to low density.
- The linear model's loss sits within about 1% of the closed-form keep-top-m loss. The
  "beats" and "worse" counts at the 1% margin are evaluation-sample noise.

### 2026-10-30 — Phase 3: analysis (`f93c62d`, uncommitted changes)

`geometry.py` holds every measurement, tested on hand-built antipodal pairs, a pentagon, a
tetrahedron and orthogonal features. `analyze.py` writes `runs/<run>/analysis/`, refusing to
overwrite: instance and feature CSVs, `columns.csv` for m = 2, `wtw.npz` for `intro`, a
summary and a diagnostic PNG. It rescores the saved weights and checks the loss matches the
one `train.py` recorded; it did for all four dev runs. `predict.py` prints inputs next to
reconstructions. 26 tests pass.

### 2026-10-30 — Phase 4: launch

40 runs launched at once, named `<preset>-<relu|linear>-seed<0–4>`: `intro`, `polygons` and
`phase` on the CPU, one thread each (these models are too small to gain from the GPU), and
`dimensionality` on the RTX 3060.

### 2026-10-30 — Phase 4: first launch failed partly; relaunched in tmux

31 of the 40 runs finished; 9 died of CUDA out-of-memory (3 small CPU sweeps and 6
dimensionality sweeps). Cause: PyTorch's Adam step calls `torch.accelerator.current_stream()`,
which opens a CUDA context (about 100 MB) even when every tensor is on the CPU, so the 30
"CPU" processes took about 3 GB of the GPU, and the 10 dimensionality runs (about 1.2 GB each)
did not fit beside them. Claude Code also froze during the launch, likely under the load of
40 processes; the runs themselves were unaffected by that.

Fix, at the owner's request so runs survive Claude Code being killed: `sweep.sh`, started by
`just sweep` inside a detached tmux session (`sardines-sweep`). It hides the GPU from CPU runs
(`CUDA_VISIBLE_DEVICES=''`, checked: no context opened), runs at most 12 CPU and 3 GPU jobs
at once, skips runs whose weights exist, moves unfinished run folders to `runs/failed/`
rather than deleting them, analyses every run, then runs `compare.py` for each experiment.
`compare.py` is new: it gathers each experiment's per-seed analyses into one cross-seed
table (`runs/compare-<experiment>/`). The plan called for that comparison but named no script.

### 2026-10-30 — Phase 4: results for intro, polygons and phase

- **intro** (seeds 0–4): the linear model keeps exactly 5 features at every sparsity and its
  loss matches the closed-form keep-top-5 loss within evaluation noise. The ReLU model keeps
  5 when dense, 7–8 at 1 − S = 0.3, 9–10 at 0.1, 10 at 0.03, 11–12 at 0.01, 11–13 at 0.003
  and 13–14 at 0.001, with loss 0.13–0.15 × keep-top-5 at the sparsest. Matches the paper.
- **polygons** (seeds 0–4): a regular pentagon in all five seeds at every density from 0.05
  down. From 0.25 to 0.06 the seeds split between pentagon and square (two antipodal pairs,
  fifth feature dropped). Dense: 2–3 features, not evenly spread. Pentagon norms exceed 1
  with a small negative bias, as the paper describes.
- **phase** (seeds 0–4): each seed lands in a different phase from the lowest-loss seed in
  96–104 of 576 cells; where seeds split, the best fit in the other phase has a median 10×
  the lowest loss, so these are local minima, not ties. Majority: superposition 347,
  dedicated 118, not learned 111, 211 cells split. Lowest-loss seed per cell: superposition
  433, dedicated 71, not learned 72, which is the paper's picture.
  `compare.py` now records each cell's lowest-loss seed and phase. **The owner decided to
  show both panels** (majority and lowest-loss), with the selection stated in the caption.
  The paper itself averages 9 of 10 fits per cell for its phase diagram and takes the
  lowest-loss fit only for its 2D figures; the README says so.

Phase 5 started in parallel: `intro`, `polygons` and `phase` figures are built from the
five-seed data, and `just figure-data` copies every figure's numbers out of `runs/`.

### 2026-10-31 — Phase 4 done, phase 5: dimensionality, figures, README

All 40 runs trained and analysed, no failures; the four `runs/compare-*` folders are
written. `sweep.sh` was edited (comments, `PYTHONUNBUFFERED`) while bash was running it, so
the script skipped its last line and never printed "sweep finished"; the outputs are all
there. Lesson: never edit a running bash script.

- **dimensionality** (seeds 0–4): m/‖W‖²_F holds at 0.50 ± 0.01 for 1/(1 − S) ≈ 2.4–3.5
  in every seed. Dᵢ clusters: 1,441 features within 0.01 of ½, 214 of ⅖, 171 of ⅜, 21 of ⅔,
  none of ¾. Near the dense end Dᵢ spreads over 0.61–0.72. From 1/(1 − S) ≈ 84 all 400
  features are represented and the curve floors at 0.049 (‖Wᵢ‖ 1.18–1.30). The linear model
  stays at 0.99–1.00.
- The 10 dimensionality analyses say `rescored_matches_recorded: null`: `sweep.sh` analyses
  with the GPU hidden, and these runs were trained and scored on CUDA, whose evaluation
  stream differs from the CPU's. The 30 CPU runs all rescored to their recorded loss.

Phase 5: four README figures (`intro`, `polygons`, `phase`, `dimensionality`) built in the
engraving style from `figures/data/` (`just figure-data` copies them out of `runs/`), with
no overfull boxes and Libertinus fonts only. README rewritten: results with tables and
captions, how it works, experiment, tests, notes. Every quoted number was checked against
the compare files. `just check`, 26 tests and `just quick` pass. PROGRESS.md updated.

Done-when check:
- intro: matches (dense 5 for both; ReLU up to 13–14, linear 5).
- polygons: regular pentagon in 5 of 5 seeds from 1 − S = 0.05 down.
- phase: the lowest-loss diagram matches the paper's arrangement; the majority diagram
  shows local minima in about 17% of cells per seed. Both shown, by the owner's decision.
- dimensionality: the ½ plateau, and ⅖ and ⅜ clusters per feature; no ¾. Stated in the README.

### 2026-10-31 — Owner's answers; archived

The owner accepted the finished work and asked to commit it and upload the runs to Hugging
Face. That answers the open question: the two preset changes taken from the paper (uniform
importance for `polygons`, 400 → 30 for `dimensionality`) stand. All 40 runs and the four
cross-seed comparisons go to coeiico/sardines-mlp, one folder each; `just push` now uploads
into a folder named after the run (it uploaded to the repo root before).
