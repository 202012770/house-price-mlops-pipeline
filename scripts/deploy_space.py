"""Push the trained service to a Hugging Face Docker Space.

Usage:
    huggingface-cli login
    python scripts/deploy_space.py --repo-id <username>/house-price-predictor
"""

from __future__ import annotations

import argparse
import io
from pathlib import Path

from huggingface_hub import HfApi

PROJECT_ROOT = Path(__file__).resolve().parents[1]
UPLOAD_PATHS = ("src", "configs", "models", "data/reference")
UPLOAD_FILES = ("Dockerfile", "pyproject.toml")

SPACE_README = """---
title: House Price Predictor
emoji: 🏠
colorFrom: indigo
colorTo: green
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# House Price Predictor

End-to-end Ames housing price regression served with FastAPI.

- Interactive docs: `/docs`
- Health check: `/health`
- Model metadata: `/model/info`
- Batch inference: `POST /predict/batch`
"""


def _ignore(directory: str, names: list[str]) -> set[str]:
    ignored = set()
    for name in names:
        if name in {"__pycache__", ".pytest_cache"} or name.endswith((".pyc", ".db")):
            ignored.add(name)
    return ignored


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", required=True, help="e.g. alice/house-price-predictor")
    parser.add_argument("--private", action="store_true")
    args = parser.parse_args()

    api = HfApi()
    api.create_repo(
        repo_id=args.repo_id,
        repo_type="space",
        space_sdk="docker",
        exist_ok=True,
        private=args.private,
    )

    for relative in UPLOAD_PATHS:
        path = PROJECT_ROOT / relative
        if path.exists():
            api.upload_folder(
                folder_path=str(path),
                path_in_repo=relative,
                repo_id=args.repo_id,
                repo_type="space",
                ignore_patterns=["__pycache__/*", "*.pyc"],
            )
            print(f"uploaded {relative}")

    for relative in UPLOAD_FILES:
        api.upload_file(
            path_or_fileobj=str(PROJECT_ROOT / relative),
            path_in_repo=relative,
            repo_id=args.repo_id,
            repo_type="space",
        )
        print(f"uploaded {relative}")

    api.upload_file(
        path_or_fileobj=io.BytesIO(SPACE_README.encode("utf-8")),
        path_in_repo="README.md",
        repo_id=args.repo_id,
        repo_type="space",
    )
    print(f"Space ready: https://huggingface.co/spaces/{args.repo_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
