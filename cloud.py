"""
Run train.py and analyze.py on Modal: one GPU per run, every run of a sweep at once.

Scaffold: SWEEPS is empty until the experiments exist. Before the first cloud run, read
.agents/instructions/cloud.md and the Cloud Runs notes in CLAUDE.md, then set GPU, HOURS and
SWEEPS, plus a prepare() step if runs share downloaded or generated data.

The scripts run unchanged, in a container built from pyproject.toml and uv.lock. runs/ and
data/ live on Modal Volumes, so results outlast the container, and `just cloud-get <run>`
copies a run back into runs/. Figures and Hugging Face uploads stay local.

  just cloud <sweep>                  every run in SWEEPS[<sweep>], in parallel
  just cloud-train <run> '<flags>'    one run
"""

import shlex
import subprocess
from pathlib import Path, PurePosixPath

import modal

PROJECT = 'sardines-mlp'
GPU = 'L4'
"""The default GPU, from the Cloud Runs notes in CLAUDE.md. Replace it once a run has been timed."""
HOURS = 4
"""How long a run may take before Modal stops it."""

SWEEPS: dict[str, list[tuple[str, list[str]]]] = {
  # Each entry is (run name, flags), with the flags `just train` takes. For example:
  # 'seeds': [(f'default-seed{s}', ['--seed', str(s)]) for s in range(3)],
}

REPO = Path('/repo')
runs = modal.Volume.from_name(f'{PROJECT}-runs', create_if_missing=True)
data = modal.Volume.from_name(f'{PROJECT}-data', create_if_missing=True)
VOLUMES: dict[str | PurePosixPath, modal.Volume | modal.CloudBucketMount] = {'/volumes/runs': runs, '/volumes/data': data}

image = (
  modal.Image.debian_slim(python_version='3.12')
  # code_state() records the commit with git, as do git dependencies when installed. The
  # copied files belong to another user and lose their modes, which git would otherwise
  # refuse or report as edits.
  .apt_install('git')
  .run_commands("git config --global --add safe.directory '*'", 'git config --global core.fileMode false')
  .uv_sync(extra_options='--no-dev')  # exactly uv.lock's packages, without the dev tools
  .env({'PYTHONPATH': str(REPO / 'src'), 'PYTHONUNBUFFERED': '1'})
  # The working tree as it is, uncommitted edits and .git included, so metrics.json records
  # the code state just as a local run does. Only gitignored paths stay behind; .env holds
  # secrets and must never be uploaded.
  .add_local_dir('.', str(REPO), copy=True, ignore=[
    '.env', '.venv', 'runs', 'data', 'figures/build', '.trackio',
    '**/__pycache__', '.pytest_cache', '.ruff_cache', '**/__marimo__',
  ])
)
app = modal.App(PROJECT, image=image)


def link_volumes() -> None:
  # The scripts read and write runs/ and data/ relative to the repo, as they do locally.
  # .gitignore's runs/ and data/ match folders, not symlinks, so git is told to skip the
  # links too; otherwise every cloud run would record its code as edited.
  with open(REPO / '.git/info/exclude', 'a') as exclude:
    exclude.write('/runs\n/data\n')
  for name in ('runs', 'data'):
    link = REPO / name
    if not link.exists():
      link.symlink_to(f'/volumes/{name}')


def python(*args: str) -> None:
  subprocess.run(['python', *args], cwd=REPO, check=True)


@app.function(gpu=GPU, volumes=VOLUMES, timeout=HOURS * 3600)
def train(name: str, flags: list[str]) -> None:
  """Train one run and analyse it, as `just train` then `just analyze <run>`."""
  link_volumes()
  try:
    python('train.py', '--run-name', name, *flags)
    python('analyze.py', '--run', f'runs/{name}')
  finally:
    runs.commit()  # keep whatever the run wrote, even if it failed partway


def launch(specs: list[tuple[str, list[str]]], gpu: str) -> None:
  # Data that several runs read is built once, before this, by a prepare() function (cloud.md).
  # One container and one GPU per run, all at once; a failed run doesn't stop the others.
  results = train.with_options(gpu=gpu).starmap(specs, return_exceptions=True)
  for (name, _), result in zip(specs, results, strict=True):
    print(f'{name}: failed: {result}' if isinstance(result, Exception) else f'{name}: done')


@app.local_entrypoint()
def sweep(name: str, gpu: str = GPU) -> None:
  if name not in SWEEPS:
    raise SystemExit(f'no sweep named {name!r}; SWEEPS has {sorted(SWEEPS) or "none yet"}')
  launch(SWEEPS[name], gpu)


@app.local_entrypoint()
def single(name: str, flags: str = '', gpu: str = GPU) -> None:
  launch([(name, shlex.split(flags))], gpu)
