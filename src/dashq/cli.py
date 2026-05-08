from __future__ import annotations

import argparse
import json
from pathlib import Path

from huggingface_hub import snapshot_download


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect a DASH-Q quantized checkpoint.")
    parser.add_argument("repo_id_or_path")
    parser.add_argument("--cache-dir", default="/workspace/cache/models")
    parser.add_argument("--revision", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    path = Path(args.repo_id_or_path)
    if not path.exists():
        path = Path(
            snapshot_download(
                repo_id=args.repo_id_or_path,
                cache_dir=args.cache_dir,
                revision=args.revision,
            )
        )
    config_path = path / "dashq_config.json"
    if not config_path.exists():
        raise SystemExit(f"Missing dashq_config.json: {path}")
    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    print(f"checkpoint: {path}")
    print(f"base_model: {config.get('base_model')}")
    print(f"model_class: {config.get('model_class')}")
    print(f"method: {config.get('method')}")
    print(f"quantized_modules: {len(config.get('quantized_modules', {}))}")


if __name__ == "__main__":
    main()
