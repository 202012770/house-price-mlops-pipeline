"""Cron-friendly drift check: exits 1 when the model should be retrained."""

from __future__ import annotations

import sys

from house_price.cli import main

if __name__ == "__main__":
    sys.exit(main(["drift", *sys.argv[1:]]))
