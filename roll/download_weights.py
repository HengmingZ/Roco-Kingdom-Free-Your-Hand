# [NEW - 2026-09-19]
# Reason: Convenient top-level launcher for model weights auto-download tool.
# Content: CLI entry for weights downloading with progress feedback.

import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from model.weights_manager import main

if __name__ == "__main__":
    main()
