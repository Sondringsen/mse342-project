#!/usr/bin/env python
"""Run the one-step FinDiffusion/CSDI comparison pipeline."""

from pathlib import Path
import sys

PIPELINE_ROOT = Path(__file__).resolve().parent
if str(PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(PIPELINE_ROOT))

from pipeline.pipeline import main


if __name__ == "__main__":
    main()

