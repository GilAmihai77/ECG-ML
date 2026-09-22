"""Generate the MLflow exploration notebook.

This script is currently the source of truth for the notebook -- edit here and
regenerate. The guard at the bottom refuses to overwrite once that stops being
true, i.e. once someone adds cells in Jupyter without back-porting them.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import nbformat as nbf

REPO = "https://github.com/GilAmihai77/ECG-ML.git"
DEST = Path("D:/ECG/notebooks/03_mlflow_explore.ipynb")

nb = nbf.v4.new_notebook()
cells = []


def md(text: str) -> None:
    cells.append(nbf.v4.new_markdown_cell(text.strip()))


def code(text: str) -> None:
    cells.append(nbf.v4.new_code_cell(text.strip()))


md(
    """
# Exploring the MLflow tracking database

Read-only companion to `02_colab_run.ipynb`. That notebook trains; this one
opens what the training wrote and lets you look at it.

**No GPU.** Runtime -> Change runtime type -> **CPU**. Nothing here trains
anything, and a GPU runtime spent reading SQLite is a GPU runtime wasted.

## Where the database comes from

During a study MLflow writes to `/content/mlruns/mlflow.db` on Colab's **local**
disk, and a consistent snapshot is copied to Drive beside each checkpoint —
`MyDrive/ecg/runs/<study>/<run>/mlflow.db`. Each snapshot is a copy of the
*whole* database as it stood at that moment, not just that run's rows.

So within one session the snapshots are nested: every later one contains
everything the earlier ones did. **Across sessions they are not.** A fresh
Colab session starts from an empty database unless `restore_tracking` is run
first, so the snapshots left by different sessions are disjoint, and Drive ends
up holding several complete-but-different databases side by side.

Cell 3 therefore picks the **largest** snapshot, not the most recently modified
one. Size is a fair proxy for completeness when every file is a full copy, and
modification time is not trustworthy here: Drive's FUSE layer does not promise
that `mtime` tracks write order, and a re-upload or a re-sync can leave an old
snapshot looking like the newest thing on the mount. Confirm the choice in
cells 6-7 rather than assuming it — they show which runs are in the database and
how many epochs each logged, which is what a wrong pick looks like.

If you ran the end-of-session `sync_tracking` from notebook 02,
`MyDrive/ecg/mlflow.db` is in the listing too and is found the same way.

## The one rule: copy it off Drive before opening it

SQLite's locking assumes POSIX semantics that Drive's FUSE mount does not
honour. Opening the database *on the mount* — even to read — is how that file
corrupts, and it is the only copy of the study's history. Cell 4 copies it to
local disk and everything afterwards reads the local copy. `training_history`
opens read-only on top of that as a second line of defence.

Nothing in this notebook writes to Drive.
"""
)

md("## 1. Install the package")

code(
    f"""
!git clone --depth 1 {REPO} /content/ECG-ML 2>/dev/null || (cd /content/ECG-ML && git pull -q)
%pip install -q -e /content/ECG-ML

# The editable install writes a .pth file into site-packages, but `site` only
# reads .pth files at interpreter start, so it does not affect the kernel that
# just ran the install. Point at src/ by hand instead of restarting.
import importlib
import sys

if "/content/ECG-ML/src" not in sys.path:
    sys.path.insert(0, "/content/ECG-ML/src")
importlib.invalidate_caches()

import ecg

print("ecg", ecg.__version__, "from", ecg.__file__)
"""
)

md("## 2. Mount Drive")

code(
    """
from google.colab import drive

drive.mount("/content/drive")

from pathlib import Path

ECG = Path("/content/drive/MyDrive/ecg")
LOCAL_DB = Path("/content/mlflow.db")

assert ECG.exists(), f"no {ECG} -- is this the right Drive account?"
print(f"drive: {ECG}")
"""
)

md(
    """
## 3. Find the snapshots

Every copy of `mlflow.db` anywhere under `MyDrive/ecg`, **largest first**. The
top row is the one to take; see the note above for why size rather than time.

Read the two columns together. Rows of a similar size with timestamps minutes
apart are one session's successive snapshots, and the largest of them is the
one you want. A row that is a different size *and* hours or days away is a
different session, holding different runs.

A study that is still running keeps producing larger snapshots, so re-run cells
3-4 to pick up the epochs written since.
"""
)

code(
    """
import datetime as dt

import pandas as pd

from ecg.training.checkpoints import fullest_tracking

# Listed largest first, and the pick is fullest_tracking's -- the same rule
# notebook 02 restores with, so the two cannot disagree about which snapshot is
# the current one. Size, not mtime: see the note at the top.
found = sorted(ECG.rglob("mlflow.db"), key=lambda p: p.stat().st_size, reverse=True)
assert found, f"no mlflow.db anywhere under {ECG}; has a study run yet?"

snapshots = pd.DataFrame(
    {
        "path": [p.relative_to(ECG).as_posix() for p in found],
        "MB": [round(p.stat().st_size / 1e6, 1) for p in found],
        "modified": [
            dt.datetime.fromtimestamp(p.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
            for p in found
        ],
    }
)
print(snapshots.to_string(index=False))

SOURCE_DB = fullest_tracking(ECG)
print()
print(f"largest: {SOURCE_DB}")
"""
)

md(
    """
## 4. Copy it to local disk

`shutil.copy2`, not `sqlite3.backup`: the source is a finished snapshot nobody
is writing to, and the backup API would have to *open* the file on the mount,
which is the thing we are avoiding.

Set `SOURCE_DB` by hand above this line if you want a snapshot other than the
largest — a `path` from the table is enough:

```python
SOURCE_DB = ECG / "runs/study/sup-conv-ssl-f020-s2/mlflow.db"
```
"""
)

code(
    """
import shutil

shutil.copy2(SOURCE_DB, LOCAL_DB)
print(f"{SOURCE_DB.name}  ->  {LOCAL_DB}  ({LOCAL_DB.stat().st_size / 1e6:.1f} MB)")
"""
)

md(
    """
## 5. What is in it

A `q()` helper for free-form SQL, then the four tables worth knowing:
`experiments`, `runs`, `metrics` (one row per key per epoch) and `params` (one
row per hyperparameter, written once at run start).

**This is also where you check cell 3 picked the right snapshot.** The run
listing carries `started` and `epochs`: the runs of one session share a
contiguous block of start times, and `epochs` is how many the run actually
logged. A pretraining run you know ran 600 epochs showing 50 here means this
database is from an earlier session — go back to cell 3 and take another row.
"""
)

code(
    """
import sqlite3


def q(sql: str, params: tuple = ()) -> pd.DataFrame:
    \"\"\"Run read-only SQL against the local copy.\"\"\"
    connection = sqlite3.connect(f"file:{LOCAL_DB.as_posix()}?mode=ro", uri=True)
    try:
        return pd.read_sql(sql, connection, params=params)
    finally:
        connection.close()


print(q("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name").to_string(index=False))
"""
)

code(
    """
print("experiments")
print(q("SELECT experiment_id, name, lifecycle_stage FROM experiments").to_string(index=False))

print()
print("runs")
print(
    q(
        \"\"\"
        SELECT e.name AS experiment,
               r.name AS run,
               (SELECT value FROM tags t
                 WHERE t.run_uuid = r.run_uuid AND t.key = 'variant') AS variant,
               (SELECT value FROM tags t
                 WHERE t.run_uuid = r.run_uuid AND t.key = 'git_sha') AS git_sha,
               (SELECT value FROM tags t
                 WHERE t.run_uuid = r.run_uuid AND t.key = 'git_dirty') AS dirty,
               r.status,
               datetime(r.start_time / 1000, 'unixepoch') AS started,
               ROUND((r.end_time - r.start_time) / 60000.0, 1) AS minutes,
               (SELECT MAX(m.step) + 1 FROM metrics m
                 WHERE m.run_uuid = r.run_uuid) AS epochs,
               (SELECT COUNT(*) FROM metrics m WHERE m.run_uuid = r.run_uuid) AS n_metrics
        FROM runs r
        JOIN experiments e ON e.experiment_id = r.experiment_id
        ORDER BY r.start_time
        \"\"\"
    ).to_string(index=False)
)
"""
)

md(
    """
### Reading the variant and commit columns

`variant` is the label passed to `--variant`; `git_sha` and `dirty` are the
code that actually ran. Both exist because neither is sufficient alone.

A **variant** with no commit behind it is a label someone typed — nothing stops
two different architectures carrying the same one. A **commit** with no variant
is precise but unreadable, and does not group the fourteen runs of one study.

Two things to look for:

- **`dirty = true`** — the tree had uncommitted changes, so `git_sha` names
  something that is not what ran. Treat that run as unreproducible.
- **One variant spanning several commits** — you changed code mid-study. The
  arms are then not strictly comparable, which is the thing `--variant` exists
  to prevent. Runs before and after the change belong to different studies.

`None` in these columns means the run predates this tagging.
"""
)

md(
    """
## 6. The per-epoch history

`training_history` is the same reader the research summary uses, so what you see
here is what the write-up will report. Long form — `run`, `key`, `step`,
`value` — because that is the shape MLflow stores and the shape `curve()` and
the plotting helpers expect.

The matrix below is the orientation step: which run logged which key, and for
how many epochs. A run that stopped early has fewer steps, which is the early
stopper working, not data missing.

`test_*` keys are the exception: they are logged once, at the selected epoch,
so they show a count of 1. Test is read once per run, after selection.

That final block re-logs the `val_macro_*` keys alongside them, so those keys
show **one count more than the run has epochs** — the selected epoch carries two
rows. The two are the same model scored on the same cohort and should agree;
cell 9 takes the later one so the table matches `result.json` exactly.
"""
)

code(
    """
from ecg.experiments.report import curve, plot_training_curves, style_table, training_history

history = training_history(LOCAL_DB)  # add experiment="ecg-ssl" to narrow
print(f"{len(history):,} metric rows, {history['run'].nunique()} runs")

matrix = history.pivot_table(
    index="key", columns="run", values="step", aggfunc="count"
).fillna(0).astype(int)
matrix
"""
)

md(
    """
## 7. Hyperparameters and seeds

Integrity rule 4 says every hyperparameter and seed is recorded; this is where
to check that it actually was. Rows that differ between runs are the interesting
ones, so the constant rows are dropped by default — flip `only_varying` to see
the full config.

**A/B and C/D are only valid if this table shows one difference beyond the
seeds.** Architecture, optimiser and budget must be identical between the arms
being compared; the embedder is the single thing allowed to vary.

`train.seed` and `subset_seed` vary too, and are supposed to: each experiment
is replicated across five of them and reported as mean ± sd. They move
together, always to the same value, so a run where they disagree is a bug. What
must NOT vary within one `(arm, label_fraction)` is anything else — compare
runs sharing a seed suffix to read the table with the replicates held still.

`variant` is logged as a parameter as well as a tag, so it appears here as its
own row. If the database holds more than one architecture, **restrict to a
single variant before reading this table** — otherwise the varying rows are the
differences between architectures, which are supposed to differ, and the check
tells you nothing. Set `only_runs` below to do that.
"""
)

code(
    """
only_varying = True
only_runs = None  # e.g. "deep6" to keep just that variant's runs

params = q(
    \"\"\"
    SELECT r.name AS run, p.key, p.value
    FROM params p
    JOIN runs r ON r.run_uuid = p.run_uuid
    \"\"\"
).pivot(index="key", columns="run", values="value")

if only_runs:
    params = params.loc[:, params.columns.str.contains(only_runs)]
if only_varying:
    params = params[params.nunique(axis=1, dropna=False) > 1]
params
"""
)

md(
    """
## 8. Training curves

Two panels, never two y-axes on one: a dual-axis plot invites the eye to read a
crossing point that is an artefact of two arbitrary scales.

`val_loss` and `train_loss` are the honest overfitting diagnostic. Neither
selects anything — selection is `val_macro_auroc` and only that.
"""
)

code(
    """
import matplotlib.pyplot as plt

runs = sorted(history["run"].unique())
chosen = runs[:5]  # edit: at most ~5 lines stay readable on one panel
print("plotting:", chosen)

plot_training_curves(
    history,
    chosen,
    keys=("train_loss", "val_loss"),
    ylabels=("BCE loss (train)", "BCE loss (val)"),
    title="Train vs validation loss",
)
plot_training_curves(
    history,
    chosen,
    keys=("val_macro_auroc", "val_macro_f1"),
    ylabels=("macro AUROC (val) -- the selection metric", "macro F1 (val)"),
    title="Validation metrics",
)
plt.show()
"""
)

md(
    """
## 9. The selected epoch

For each run: the epoch that maximises `val_macro_auroc`, and every metric
logged at that step. Because the final test metrics are logged at the selected
epoch too, the `val_` and `test_` columns of a row line up — which is the
cleanest way to see the selection gap.

Ranking by AUROC here reproduces what the runner actually did. Ranking this
table by a `test_` column instead would be selecting on test, which is what
integrity rule 2 forbids.

`aggfunc="last"` is not decoration. The selected epoch holds two rows for each
`val_macro_*` key (see cell 6), and the default `mean` would quietly average
them — reporting a number that appears in neither MLflow nor `result.json`.
"""
)

code(
    """
wide = history.pivot_table(
    index=["run", "step"], columns="key", values="value", aggfunc="last"
)

best = wide.loc[wide.groupby("run")["val_macro_auroc"].idxmax()]
best.index = best.index.set_names(["run", "epoch"])

columns = [c for c in wide.columns if c.startswith(("val_macro", "test_macro"))]
style_table(best[columns])
"""
)

md(
    """
## 10. A run in full

Everything one run logged, epoch by epoch. Change `pick` and re-run.
"""
)

code(
    """
pick = chosen[0]

print(pick)
print(sorted(history.loc[history["run"] == pick, "key"].unique()))
wide.loc[pick].round(4)
"""
)

md(
    """
## 11. The MLflow UI, optionally

The tables above answer most questions faster than clicking. The UI is worth
starting when you want to diff two runs side by side, or browse artifacts.

It serves the **local copy**, so anything it writes — a schema migration, a
deleted run — cannot reach Drive. If it refuses to start on a version mismatch,
run `!mlflow db upgrade sqlite:////content/mlflow.db` and try again; that
rewrites the local copy only.
"""
)

code(
    """
import subprocess
import time

from google.colab import output

server = subprocess.Popen(
    [
        "mlflow",
        "ui",
        "--backend-store-uri",
        f"sqlite:///{LOCAL_DB}",
        "--port",
        "5000",
    ],
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
)
time.sleep(10)

if server.poll() is not None:
    print("mlflow ui exited:")
    print(server.stdout.read())
else:
    output.serve_kernel_port_as_window(5000)
    print("serving on port 5000 -- stop it with server.terminate()")
"""
)

md(
    """
## 12. Taking the history home

To browse on your own machine, download `/content/mlflow.db` from the Colab file
browser and point the UI at it:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

To let the **next** Colab session append to these same runs rather than start
empty ones beside them, restore the snapshot before training:

```python
from ecg.training.checkpoints import restore_tracking

restore_tracking(ECG / "mlflow.db", "/content/mlruns/mlflow.db")
```

Both read the copy. Neither opens the database on the mount.
"""
)

nb["cells"] = cells
nb["metadata"] = {
    "colab": {"provenance": []},
    "kernelspec": {"display_name": "Python 3", "name": "python3"},
    "language_info": {"name": "python"},
}

def refuse_to_clobber(destination: Path, writing: int) -> None:
    """Stop if the notebook on disk holds cells this script would delete.

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
            f"this script writes {writing}. Those extra cells are not in this "
            "script -- regenerating would delete them. Edit the notebook "
            "instead, or pass --force to rebuild from scratch."
        )


refuse_to_clobber(DEST, len(cells))
DEST.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, DEST)
print(f"wrote {DEST} with {len(cells)} cells")
