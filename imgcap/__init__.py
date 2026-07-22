"""Image captioning: ResNet50 encoder + attention + LSTM decoder.

Two training regimes are supported from a single training loop:
  1. Cross-entropy (teacher forcing) — the baseline stage.
  2. SCST (Self-Critical Sequence Training) — a CIDEr-reward REINFORCE
     fine-tuning stage, switched on after `cfg.scst_start_epoch`.
"""

__version__ = "1.0.0"
