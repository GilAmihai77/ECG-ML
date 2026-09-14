This is a demo project- preparing for ECG ML/DL startup interview.
Write modular code, with typehints and docstrings. 
Keep codebase organized and easy to understand. Whenever possible import code instead of rewriting.
There are some stages- I will give you a stage at a time. When its working ok + tests- we proceed. 

I will start working localy, but for bigger transformer we will need cloud GPU-
colab or runpod. So we first run localy on a small transformer, then in cloud
with bigger model.

Local python in D:/conda/envs/torch
Note: the local torch is a CPU-only build (2.12.0+cpu) and the local GPU is a
GTX 1050 4GB. Local runs are for smoke tests only; every real run is cloud.

Because the cloud target is Colab/RunPod, whose filesystems are ephemeral:
- Preprocess ONCE into a single compact array committed to Drive or a RunPod
  volume. The training path must not depend on wfdb or neurokit2 - those are
  preprocessing-only, so cloud setup stays small and fast.
- Checkpoint often and to persistent storage. Colab disconnects.
- MLflow must log somewhere that survives the session. A local mlruns/ on Colab
  is lost on disconnect.

If possible I want all experiments tracked by mlflow.

Goal of reseach:
Does self-supervised pretraining help when labelled ECG data is scarce,
and does the patch embedding choice interact with it?
The emphasis is on SSL. The motivating asymmetry is 10,254 labelled
training records against a 17,418-record unlabelled pretraining pool.

Dataset:
PTB-XL.

Input:
12-lead, 10-second ECG.

Tokenization:
Fixed temporal patches in every arm. The compared variable is the
patch embedder, not the tokenizer.
- Linear: flatten the patch, one nn.Linear to d_model.
- Conv1d stem: multi-layer, overlapping kernels, nonlinearity.
  Note Conv1d(kernel_size=patch, stride=patch) is mathematically
  identical to the linear projection, so the stem must be a real one.

Models:
Same Transformer encoder for both embedders. Only the patch embedder
differs; encoder, optimizer, budget and SSL decoder are identical.

Experiment:
A. Linear embedding + supervised from scratch
B. Conv1d embedding + supervised from scratch
C. Linear embedding + SSL pretraining + fine-tuning
D. Conv1d embedding + SSL pretraining + fine-tuning

Primary metrics:
macro AUROC, macro PR-AUC, F1.

All splits must be patient-wise.

No data augmentation - considered and declined. SSL is the answer to
small labelled data instead.

Do not change architecture, optimizer, training budget, etc.
between A/B or C/D. Sequence length is now identical across all arms,
so there is no longer any excuse for the architecture to differ.

Match the two embedders' parameter counts to within ~10% and log both,
otherwise a conv win is capacity rather than inductive bias.

Dropped direction:
RR/beat-oriented tokenization was designed and analysed, then dropped
for time. The analysis is kept in docs/tokenizer_comparison.html
because it justifies the decision: below 50 bpm only 62% of a record
falls inside any beat token, and sequence length runs 84-336 against a
constant 144.

Research integrity requirements:

1. Never split individual ECG recordings randomly if this can cause
   patient leakage. Splits must be patient-wise.

2. Never use test data for preprocessing statistics, model selection,
   threshold selection, or hyperparameter tuning.

3. Keep the Transformer architecture identical between the two patch
   embedders. Only the embedder may differ.

4. Record all hyperparameters and random seeds.

5. Save experiment configurations with checkpoints.

6. Every experiment must be reproducible from a config file.

7. Do not silently modify preprocessing, tokenization or the patch
   embedder to improve results.

8. Report failures and invalid ECGs rather than silently dropping them.

9. Keep research code separate from exploratory notebooks.