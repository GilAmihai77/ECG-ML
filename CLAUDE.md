This is a demo project- preparing for ECG ML/DL startup interview.
Write modular code, with typehints and docstrings. 
Keep codebase organized and easy to understand. Whenever possible import code instead of rewriting.
There are some stages- I will give you a stage at a time. When its working ok + tests- we proceed. 

I will start working localy, but for bigger transformer we will need cloud GPU- runpod or AWS. So we first run localy on a small transformer, then in cloud with bigger model.

Local python in D:/conda/envs/torch

If possible I want all experiments tracked by mlflow.

Goal of reseach:
Compare fixed temporal patch tokenization with RR/beat-oriented
tokenization (eg tokens are 4 RR s) for self-supervised ECG representation learning.

Dataset:
PTB-XL.

Input:
12-lead, 10-second ECG.

Models:
Same Transformer encoder for both tokenizers.

Experiment:
A. Fixed tokenizer + supervised from scratch
B. RR tokenizer + supervised from scratch
C. Fixed tokenizer + SSL pretraining + fine-tuning
D. RR tokenizer + SSL pretraining + fine-tuning

Primary metrics:
macro AUROC, macro PR-AUC, F1.

All splits must be patient-wise.

Do not change architecture, optimizer, training budget, etc.
between A/B or C/D unless required by sequence length.

Research integrity requirements:

1. Never split individual ECG recordings randomly if this can cause
   patient leakage. Splits must be patient-wise.

2. Never use test data for preprocessing statistics, model selection,
   threshold selection, or hyperparameter tuning.

3. Keep the Transformer architecture identical between tokenizers
   whenever possible.

4. Record all hyperparameters and random seeds.

5. Save experiment configurations with checkpoints.

6. Every experiment must be reproducible from a config file.

7. Do not silently modify preprocessing or tokenization to improve
   results.

8. Report failures and invalid ECGs rather than silently dropping them.

9. Keep research code separate from exploratory notebooks.