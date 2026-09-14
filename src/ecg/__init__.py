"""Self-supervised ECG representation learning on PTB-XL.

Research code comparing linear and convolutional patch embeddings for a
Transformer encoder, with and without self-supervised pretraining. Exploratory
notebooks live in ``notebooks/`` and import from this package rather than
redefining logic, so that every number in a notebook is produced by the same
code path as the experiments.
"""

from __future__ import annotations

import sys

# Import-order workaround, Windows only. pyarrow -- pulled in by both pandas and
# mlflow -- loads a runtime DLL that makes a later "import torch" fail with
# "WinError 1114: A dynamic link library (DLL) initialization routine failed"
# while importing c10.dll. Loading torch first avoids it. Linux, where every
# real (cloud) run happens, is unaffected, so this costs nothing there.
if sys.platform == "win32":  # pragma: no cover - platform specific
    try:
        import torch  # noqa: F401
    except ImportError:
        pass  # torch is optional for the preprocessing-only install

__version__ = "0.1.0"
