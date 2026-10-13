#!/usr/bin/env python3
"""Standalone entry point for the Gradio demo.

Usage:
    python demo/app.py [--ckpt-dir DIR] [--share]
"""

import sys
import os

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "archive_old_pipeline", "src"))

from gradio_demo import main  # noqa: E402

if __name__ == "__main__":
    main()