# ECG — patch embedding and self-supervised pretraining on PTB-XL

> **Status: work in progress.** This is my introduction project to ECG data
> science: a hands-on way to learn the data, the preprocessing and the
> tokenization choices that 12-lead ECG forces on you. The pipeline runs
> end to end and is covered by tests, but the study itself is still being
> experimented with — budgets, mask ratio and hyperparameters are not settled,
> and no result here should be read as a finding yet.

## The question

Does self-supervised pretraining help when labelled ECG data is scarce, and
does the patch embedding choice interact with it?

PTB-XL gives 10,254 labelled training records against a 17,418-record
unlabelled pretraining pool, which is the asymmetry the study is built around.
Four arms, identical in everything but the two variables:

| Arm | Patch embedder | Training |
| --- | --- | --- |
| A | Linear | supervised from scratch |
| B | Conv1d stem | supervised from scratch |
| C | Linear | SSL pretraining → fine-tuning |
| D | Conv1d stem | SSL pretraining → fine-tuning |

Task: multi-label classification over PTB-XL's 5 diagnostic superclasses
(NORM, MI, STTC, CD, HYP). Metrics: macro AUROC, macro PR-AUC, F1.

## How it is set up

- **Tokenization** is fixed temporal patches in every arm — the compared
  variable is the *embedder*, not the tokenizer. The conv stem is a real
  multi-layer overlapping stem, since `Conv1d(kernel=patch, stride=patch)`
  would be arithmetically identical to the linear projection.
- **Parameter counts** of the two embedders are matched to within ~10% and
  both are logged, so a conv win is inductive bias rather than capacity.
- **Splits are patient-wise**, using PTB-XL's published `strat_fold`.
  Test data is never used for preprocessing statistics, model selection or
  thresholds.
- **Every experiment runs at 5 seeds** and is reported as mean ± sd; the two
  headline comparisons subtract within a seed and then average. That makes the
  default grid 70 runs, not 14.
- **Preprocessing happens once** into a single compact array. Training never
  imports `wfdb` or `neurokit2`, so a cloud environment stays small.
- **MLflow** tracks everything, on a SQLite backend copied to durable storage
  once per epoch alongside the checkpoint.

## Layout

```
src/ecg/data/          download, PTB-XL metadata, preprocessing, datasets
src/ecg/models/        patch embedders, encoder, masking, SSL head
src/ecg/training/      loops, metrics, checkpoints, MLflow tracking
src/ecg/experiments/   the run grid, runner, CLI, reporting
notebooks/             EDA, Colab runner, MLflow exploration
docs/                  tokenizer comparison write-up, architecture notes
tests/                 488 tests
```

## Usage

```bash
pip install -e ".[preprocess,dev]"

ecg-download --dest data/ptbxl              # PTB-XL from PhysioNet
ecg-preprocess                              # build the waveform store, once

ecg-run --dry-run                           # print the grid, book no GPU
ecg-run --plan ablation --seeds 0           # pick the mask ratio, on val
ecg-run --output /content/drive/MyDrive/ecg/runs
```

Resuming is the default: a finished run writes `result.json` and is skipped,
so a Colab disconnect costs one run rather than the study.

Local runs are CPU smoke tests only; real runs are on a cloud GPU
(Colab / RunPod) via `notebooks/02_colab_run.ipynb`.

## Notes and dropped directions

RR/beat-oriented tokenization was designed and analysed, then dropped for
time. The analysis is kept in `docs/tokenizer_comparison.html` because it
justifies the decision: below 50 bpm only 62% of a record falls inside any
beat token, and sequence length runs 84–336 against a constant 144.

No data augmentation — considered and declined; SSL is the answer to small
labelled data here instead.
