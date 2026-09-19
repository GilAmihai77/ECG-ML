"""Generate the Colab runner notebook.

**This script is a scaffold, not the source of truth.** The notebook it writes
has since been edited directly -- the research summary in section 12 exists
only there -- so regenerating would delete that work. The guard below refuses
to overwrite a notebook that has grown more cells than this script produces.
Edit the notebook, not this file, unless you are rebuilding it from scratch.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import nbformat as nbf

REPO = "https://github.com/GilAmihai77/ECG-ML.git"
DEST = Path("D:/ECG/notebooks/02_colab_run.ipynb")

nb = nbf.v4.new_notebook()
cells = []


def md(text: str) -> None:
    cells.append(nbf.v4.new_markdown_cell(text.strip()))


def code(text: str) -> None:
    cells.append(nbf.v4.new_code_cell(text.strip()))


md(
    """
# Running the study on Colab

Everything below assumes an **A100** runtime and a Drive folder holding the
preprocessed data. Run the cells in order; each one prints enough to tell you
whether to continue.

## Before you start

**1. Set the runtime.** Runtime -> Change runtime type -> **A100 GPU**.
Cell 1 refuses to continue on a T4, because the timings below assume A100 and a
T4 would turn a 3-hour study into an overnight one.

**2. Put two things on Drive**, at `MyDrive/ecg/`:

| what | from | size |
|---|---|---|
| `store_100hz/` | your local `data/store_100hz/` | 500 MB |
| `ptbxl/ptbxl_database.csv` | your local `data/ptbxl/` | 6.6 MB |
| `ptbxl/scp_statements.csv` | your local `data/ptbxl/` | 10 KB |

**Do not upload the 3 GB `records100/` and `records500/` trees.** The waveforms
are already inside the store; only the two CSVs are read at training time.

Upload the store as a **folder**, not as a zip you unpack on Drive — unzipping
onto a FUSE mount is far slower than uploading the two `.npy` files directly.

**3. Cell 4 copies the data to `/content`.** Drive holds the master copy;
training reads the local one. This is not premature optimisation: the store is
*memory-mapped*, so reading it from Drive turns every page fault into a network
round-trip, and a mount that goes stale mid-run surfaces as a SIGBUS inside
numpy rather than a catchable error. You pay the same 500 MB read either way —
copying makes it one bulk sequential read, which is what Drive is fastest at,
and every later `ecg-run` in the session reads from local disk instead.

## The order to run in

1. **Cells 1-4** — setup and checks. Fast.
2. **Cell 5, the dry run** — prints the 70-run grid without booking anything.
   Read it. This is the cheapest place to catch a wrong fraction, and the only
   place the seed count is free to change.
3. **Cell 6, the timing probe** — one epoch, to turn "about 3 hours" into a
   real number before you commit to the full study.
4. **Cell 7, the mask-ratio ablation** — 3 pretrains + 3 short fine-tunes,
   selected on validation. Pick the winner and pass it to cell 8.
5. **Cell 8, the study** — the 70 runs.
6. **Cells 9-10** — results.

## Five seeds, and what that costs

Every experiment runs at seeds 0-4 and is reported as mean ± sd. A single run's
macro AUROC moves by more than the differences this study is trying to measure,
so one number per arm could not support a claim about either the embedder or
SSL. The seed drives the weights, the batch order, the SSL mask *and* which
labelled records the fraction draws — the last one because "if I had a
different 20% of the labels" is the question a label-scarcity study is actually
asking.

**This is five times the GPU.** 70 runs, not 14: size it with cell 6 before you
start, and expect more than one Colab session. That is survivable only because
resume is per run — see below — so plan on re-running cell 8 until it stops
finding work. If you are short of time, `--seeds 0 1 2` is a defensible three
replicates; `--share-pretraining` is the other lever, and cell 8 says what it
costs you.

## If Colab disconnects

Re-run cells 1-4, then re-run the same training cell. **Finished runs are
skipped**: each writes `result.json` when it completes, and the runner resumes
from the one that was interrupted. You lose at most the run that was in flight.

## Where things live

- **Checkpoints and results** go to Drive, under `MyDrive/ecg/runs/<run name>/`.
- **MLflow's database stays on local disk** at `/content/mlruns/mlflow.db`, and
  a consistent copy is written to Drive every epoch. This is deliberate:
  SQLite's locking assumes POSIX semantics a Drive mount does not honour, so a
  database living on Drive can corrupt silently. Cell 8 warns if you override
  `--tracking` with a Drive path.
"""
)

md("## 1. Check the GPU")

code(
    """
import subprocess

print(subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout)

import torch

assert torch.cuda.is_available(), "No GPU. Runtime -> Change runtime type -> A100."
name = torch.cuda.get_device_name(0)
memory = torch.cuda.get_device_properties(0).total_memory / 1e9
print(f"{name}, {memory:.0f} GB, torch {torch.__version__}")

if "A100" not in name:
    print(
        f"\\nWARNING: this is a {name}, not an A100. The timings in this "
        "notebook assume an A100; expect roughly 3-4x longer on a T4, and "
        "bf16 autocast is only useful on Ampere or newer."
    )
"""
)

md("## 2. Mount Drive")

code(
    """
from google.colab import drive

drive.mount("/content/drive")

from pathlib import Path

# Drive holds the master data and receives every checkpoint.
ECG = Path("/content/drive/MyDrive/ecg")
DRIVE_STORE = ECG / "store_100hz"
DRIVE_METADATA = ECG / "ptbxl"
RUNS = ECG / "runs"
RUNS.mkdir(parents=True, exist_ok=True)

# Training reads from local disk. See cell 4.
LOCAL = Path("/content/data")
STORE = LOCAL / "store_100hz"
METADATA = LOCAL / "ptbxl"

# Replicate seeds, read by both the dry run and the study so the two cannot
# describe different studies. Every experiment runs once per seed and is
# reported as mean +/- sd over them.
#
# "0 1 2" is a defensible three replicates if you are short of A100 hours, and
# it extends cleanly: those three keep their run names under "0 1 2 3 4", so
# adding the last two later re-runs nothing. Do NOT start from "0" alone --
# a one-seed plan leaves run names unsuffixed, so growing it afterwards
# re-runs the whole study under new names.
SEEDS = "0 1 2 3 4"

print(f"drive:  {ECG}")
print(f"local:  {LOCAL}")
print(f"seeds:  {SEEDS}")
"""
)

md("## 3. Install the package")

code(
    f"""
!git clone --depth 1 {REPO} /content/ECG-ML 2>/dev/null || (cd /content/ECG-ML && git pull -q)
%pip install -q -e /content/ECG-ML

# The editable install writes a .pth file into site-packages, but `site` only
# reads .pth files when the interpreter starts, so it does not affect the kernel
# that just ran the install. Point at src/ by hand instead of restarting.
# Subprocesses (the ecg-run console script) start fresh and need none of this.
import importlib
import sys

if "/content/ECG-ML/src" not in sys.path:
    sys.path.insert(0, "/content/ECG-ML/src")
importlib.invalidate_caches()

import ecg

print("ecg", ecg.__version__, "from", ecg.__file__)
"""
)

md(
    """
## 4. Copy the data to local disk, then check it

The store is memory-mapped, so leaving it on Drive would turn page faults into
network round-trips during training and make a stale mount a SIGBUS inside
numpy. Copy once, ~30-60 s, then everything reads locally at SSD speed.

The check afterwards prints the cohort sizes. If they do not match what you saw
locally, stop -- a truncated upload is much cheaper to find now than an hour
into pretraining.
"""
)

code(
    """
import shutil
import time

for source, destination in ((DRIVE_STORE, STORE), (DRIVE_METADATA, METADATA)):
    assert source.exists(), f"missing on Drive: {source}"
    if destination.exists():
        print(f"{destination} already present, skipping copy")
        continue
    start = time.perf_counter()
    shutil.copytree(source, destination)
    size = sum(f.stat().st_size for f in destination.rglob("*") if f.is_file())
    print(
        f"copied {source.name}: {size / 1e6:.0f} MB in "
        f"{time.perf_counter() - start:.0f} s"
    )

!df -h /content | tail -1
"""
)

code(
    """
from ecg.data.datasets import assert_patient_disjoint, build_cohorts, describe_cohorts
from ecg.data.preprocess import WaveformStore
from ecg.data.ptbxl import load_metadata

for required in (STORE / "waveforms.npy", METADATA / "ptbxl_database.csv"):
    assert required.exists(), f"missing locally: {required}"

store = WaveformStore.load(STORE)
metadata = load_metadata(METADATA)
cohorts = build_cohorts(metadata)

print(f"store: {len(store)} records, {store.n_samples} samples, {store.sampling_rate} Hz")
assert len(store) == 21799, f"expected 21799 records, got {len(store)}"

assert_patient_disjoint(cohorts, metadata)
print("patient-disjoint: ok\\n")
print(describe_cohorts(cohorts).to_string())
"""
)

md(
    """
## 5. Dry run: print the plan

Nothing is trained here. Read the grid before committing GPU time.
"""
)

code(
    """
!ecg-run --dry-run \\
    --store "$STORE" --metadata "$METADATA" --output "$RUNS/study" \\
    --fractions 0.2 0.5 1.0 --seeds $SEEDS \\
    --epochs 50 --d-model 256 --n-heads 8
"""
)

md(
    """
## 6. Timing probe

One epoch of each kind, so the full study's cost is a measurement rather than a
guess. It writes to a throwaway directory, so it does not pollute the real
results or the resume state.

Multiply what you see by `--epochs` **and by the number of seeds** to size
the study. The probe runs one seed; the study runs five.
"""
)

code(
    """
import time

start = time.perf_counter()
!ecg-run \\
    --store "$STORE" --metadata "$METADATA" \\
    --output /content/timing --tracking /content/timing/mlflow.db \\
    --epochs 1 --fractions 1.0 --no-track
print(f"\\nprobe wall clock: {(time.perf_counter() - start) / 60:.1f} min")
"""
)

md(
    """
### Reading the probe

The probe ran 2 pretraining epochs and 4 supervised epochs at 100% labels.
For a study with `E` epochs:

- pretraining is `2 x E` epochs
- the supervised grid is `E x (0.2 + 0.2 + 0.5 + 0.5 + 1.0 + 1.0) x 2 embedders`
  = `6.8 x E` supervised-epoch-equivalents at full size

so the study costs roughly `2 x E` pretraining epochs plus `6.8 x E` fine-tuning
epochs. At 50 epochs that is 100 pretraining epochs and ~340 fine-tuning
epoch-equivalents. If the probe says a pretraining epoch takes 40 s, budget
about 70 minutes for pretraining alone.
"""
)

md(
    """
## 7. Mask-ratio ablation

Three pretraining runs at ratios 0.3 / 0.5 / 0.7, each fine-tuned at the
smallest label fraction -- where SSL's effect is largest and the runs are
cheapest. **Selection is on validation.**

Run this before the study, and pass the winning ratio to cell 8. Use the same
ratio for every arm: tuning the mask per arm would fold the SSL setup into what
is being compared.
"""
)

code(
    """
!ecg-run --plan ablation \\
    --store "$STORE" --metadata "$METADATA" \\
    --output "$RUNS/ablation" --tracking /content/mlruns/mlflow.db \\
    --experiment ecg-ablation \\
    --mask-ratios 0.3 0.5 0.7 --fractions 0.2 --epochs 30 --seeds 0

# One seed: this selects a mask ratio on validation, it is not a reported
# result. If the three ratios land within noise of each other, re-run it
# with --seeds 0 1 2 -- picking between them on one seed is picking at
# random, and the choice is then frozen into all four arms of the study.
# That re-runs all three: one seed leaves the names unsuffixed, so the runs
# below are not reused. It is a selection step and cheap; the study is not.
"""
)

code(
    """
import pandas as pd

ablation = pd.read_csv(RUNS / "ablation" / "results.csv")
table = (
    ablation[ablation["kind"] == "supervised"]
    .loc[:, ["mask_ratio", "val_macro_auroc", "test_macro_auroc"]]
    .sort_values("mask_ratio")
)
print(table.to_string(index=False))
print(
    "\\nselect on val_macro_auroc:",
    table.loc[table["val_macro_auroc"].idxmax(), "mask_ratio"],
)
"""
)

md(
    """
## 8. The study

Fourteen runs: two pretraining runs, then four arms at each of three label
fractions. **Set `MASK_RATIO` to whatever cell 7 selected.**

Safe to re-run after a disconnect -- finished runs are skipped.

### `VARIANT`: set this whenever you change the model

A run name says which embedder and which label fraction — nothing about the
architecture. A run name is also its output directory, and finished directories
are skipped. So if you change the transformer and re-run with the same
`VARIANT`, **all fourteen runs are skipped and you get the old architecture's
numbers back**, labelled as the new one's.

`--variant` prefixes every run name, which gives the new architecture its own
directories and its own rows in MLflow. Name what changed, in a slug with no
spaces:

| you changed | `VARIANT` |
|---|---|
| nothing yet, the first study | `base` |
| 4 layers -> 6 | `deep6` |
| one more conv layer in the stem | `extra-conv-layer` |
| d_model 256 -> 384 | `d384` |

Keep `--experiment ecg-ssl` the same across all of them. Old and new belong in
one table — that comparison is the whole point, and separate experiments would
turn it into a manual join.

The run also logs the git commit and whether the tree was dirty, because the
variant is a label you type and the commit is not. **Commit before you run**: a
dirty tree means the SHA names something that is not what ran, and the study
will say so in `git_dirty`.

If you re-run across a code change without moving the variant, the runner
prints the commit each skipped run came from and warns at the end. That warning
means the results on screen are the old model's.
"""
)

code(
    """
MASK_RATIO = 0.5       # <- from cell 7
VARIANT = "base"       # <- change whenever you change the model

# 14 runs per seed, so 70 at the default SEEDS (set in cell 2). Re-run this
# cell after a disconnect; finished runs are skipped. Widening SEEDS later
# re-runs only the seeds you added, as long as you did not start from one.
# --share-pretraining would cut it to 62 by pretraining once per embedder
# instead of once per seed, but then the SSL arm's error bar omits
# pretraining variance and is narrower than the scratch arm's for a reason
# that has nothing to do with SSL. Only worth it if you are out of hours,
# and it has to be said in the write-up.

!ecg-run \\
    --store "$STORE" --metadata "$METADATA" \\
    --output "$RUNS/study" --tracking /content/mlruns/mlflow.db \\
    --experiment ecg-ssl --variant $VARIANT \\
    --mask-ratio $MASK_RATIO --mask-span 2 \\
    --fractions 0.2 0.5 1.0 --seeds $SEEDS \\
    --epochs 50 --batch-size 256 --d-model 256 --n-layers 4 --n-heads 8
"""
)

md(
    """
## 9. Results

`load_results` rather than `results.csv`: the CSV holds only the plan that last
ran, so a second variant's study overwrites the first's summary. The per-run
`result.json` files are never overwritten — each variant has its own
directories — so this rebuilds the full table from them, **every variant
included**.

When more than one variant is present, the tables below split by it instead of
averaging across it. Two architectures averaged into one row would be a number
that describes neither.
"""
)

code(
    """
from ecg.experiments.runner import (
    aggregate_runs,
    embedder_benefit,
    label_efficiency_report,
    label_efficiency_table,
    load_results,
    ssl_benefit,
)

frame = load_results(RUNS / "study")
print(frame.groupby("variant")["run"].count().to_string(), "\\n")

# Which seeds each arm actually has, first. A study that is four-fifths
# finished produces exactly the same tables below as a finished one; this
# column is the only thing that says so.
print(aggregate_runs(frame, metrics=["test_macro_auroc"])[["n_seeds", "seeds"]].to_string())

print("\\nlabel efficiency (test macro AUROC, mean +/- sd over seeds)")
print(label_efficiency_report(frame).to_string())

# Both contrasts subtract within a seed and then average, rather than
# differencing two averages. The means come out the same; the interval does
# not. A seed that was a bad draw for one arm was the same bad draw for the
# other -- same records, same batch order -- so pairing takes that shared
# movement out of the interval instead of leaving it in both terms.
# gain_beats_noise is the 95% interval on the paired mean excluding zero.
print("\\nwhat SSL bought (arms C/D against A/B)")
print(ssl_benefit(frame).to_string())

print("\\nwhat the conv stem bought (A against B, C against D)")
print(embedder_benefit(frame).to_string())
"""
)

md(
    """
## 10. The label-efficiency curve

The study's headline figure: does the SSL gap widen as labels get scarcer?

**One architecture per figure.** The four arms are only comparable within a
variant, so the cell plots `VARIANT` from cell 8; change it and re-run to see
another. Overlaying two architectures here would put eight lines on two panels
and make the SSL gap — the thing the figure exists to show — the hardest thing
on it to see.
"""
)

code(
    """
import matplotlib.pyplot as plt

curve = label_efficiency_table(frame)
spread = label_efficiency_table(frame, stat="sd").reindex(
    index=curve.index, columns=curve.columns
)
if "variant" in (curve.index.names or []):
    curve = curve.loc[VARIANT]
    spread = spread.loc[VARIANT]
fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)

for axis, embedder in zip(axes, ("linear", "conv")):
    for arm, style in ((embedder, "o--"), (f"{embedder}+ssl", "o-")):
        if arm in curve:
            # The bar is the seed-to-seed sd of that arm. Overlapping bars do
            # NOT mean the arms are indistinguishable: most of that spread is
            # shared between them, and ssl_benefit removes it by pairing. Read
            # gain_beats_noise there for the comparison; read this for scale.
            axis.errorbar(
                curve.index * 100,
                curve[arm],
                yerr=spread[arm],
                fmt=style,
                capsize=3,
                label=arm,
            )
    axis.set_title(f"{embedder} embedder")
    axis.set_xlabel("labelled training data (%)")
    axis.grid(alpha=0.3)
    axis.legend()

axes[0].set_ylabel("test macro AUROC")
fig.suptitle(
    f"Does SSL help more when labels are scarce?  ({VARIANT}, mean +/- sd over seeds)"
)
fig.tight_layout()
plt.show()
"""
)

md(
    """
## 11. Browsing the MLflow runs

The database is at `/content/mlruns/mlflow.db` with a per-epoch copy in each
run's output directory on Drive. To browse it locally, download the copy from
the last completed run and point the UI at it:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

To keep the tracking database across sessions, copy it to Drive once at the end
of a session and restore it at the start of the next:

```python
from ecg.training.checkpoints import restore_tracking, sync_tracking

sync_tracking("/content/mlruns/mlflow.db", ECG / "mlflow.db")      # end of session
restore_tracking(ECG / "mlflow.db", "/content/mlruns/mlflow.db")   # start of next
```
"""
)

nb["cells"] = cells
nb["metadata"] = {
    "accelerator": "GPU",
    "colab": {"provenance": [], "gpuType": "A100"},
    "kernelspec": {"display_name": "Python 3", "name": "python3"},
    "language_info": {"name": "python"},
}


def refuse_to_clobber(destination: Path, writing: int) -> None:
    """Stop if the notebook on disk holds work this script would delete.

    A generator that silently overwrites is a loaded gun once anyone edits the
    notebook directly. Cell count is a crude test but it catches the case that
    actually happens: cells added in Jupyter and never back-ported here.

    Args:
        destination: The notebook that would be overwritten.
        writing: How many cells this script is about to write.

    Raises:
        SystemExit: If the existing notebook has more cells.
    """
    if not destination.exists() or "--force" in sys.argv:
        return
    existing = len(json.loads(destination.read_text(encoding="utf-8"))["cells"])
    if existing > writing:
        raise SystemExit(
            f"refusing to overwrite {destination.name}: it has {existing} cells, "
            f"this script writes {writing}. Those extra cells were added to the "
            "notebook directly and are not in this script -- regenerating would "
            "delete them. Edit the notebook instead, or pass --force if you "
            "really mean to rebuild it from scratch."
        )


refuse_to_clobber(DEST, len(cells))
DEST.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, DEST)
print(f"wrote {DEST} with {len(cells)} cells")
