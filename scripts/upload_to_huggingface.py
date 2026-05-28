"""
upload_to_huggingface.py
========================
Upload CASTLE checkpoint, tokenizer, config, and source to HuggingFace Hub.

Usage (run on the training server where checkpoint_best.pt exists):
  pip install huggingface_hub
  huggingface-cli login   # or set HF_TOKEN env var

  python scripts/upload_to_huggingface.py \
    --repo_id syauqie/castle-gec \
    --checkpoint checkpoints/castle9/checkpoint_best.pt

What gets uploaded:
  - checkpoint_best.pt          (model weights)
  - configs/castle_base.yaml    (hyperparameters)
  - data/tokenizer/             (WordPiece tokenizer)
  - src/                        (source code)
  - README.md (from MODELCARD.md)
"""

import argparse
import os
from pathlib import Path
from huggingface_hub import HfApi, create_repo

def upload(repo_id: str, checkpoint: str, private: bool = False):
    api = HfApi()

    # Create repo if it doesn't exist
    print(f"Creating/verifying repo: {repo_id}")
    create_repo(repo_id, repo_type="model", private=private, exist_ok=True)

    base = Path(__file__).parent.parent  # castle-repo root

    files_to_upload = []

    # 1. Model checkpoint
    ckpt = Path(checkpoint)
    assert ckpt.exists(), f"Checkpoint not found: {ckpt}"
    files_to_upload.append((str(ckpt), "checkpoint_best.pt"))

    # 2. Config
    cfg = base / "configs" / "castle_base.yaml"
    files_to_upload.append((str(cfg), "configs/castle_base.yaml"))

    # 3. Model card (use MODELCARD.md as README.md)
    modelcard = base / "MODELCARD.md"
    files_to_upload.append((str(modelcard), "README.md"))

    # 4. Tokenizer files
    tokenizer_dir = base / "data" / "tokenizer"
    if tokenizer_dir.exists():
        for f in tokenizer_dir.iterdir():
            files_to_upload.append((str(f), f"data/tokenizer/{f.name}"))
    else:
        print("WARNING: data/tokenizer/ not found — tokenizer won't be uploaded")

    # 5. Source code
    src_dir = base / "src"
    for f in src_dir.glob("*.py"):
        files_to_upload.append((str(f), f"src/{f.name}"))

    # Upload
    for local_path, repo_path in files_to_upload:
        if not Path(local_path).exists():
            print(f"  SKIP (not found): {local_path}")
            continue
        print(f"  Uploading: {repo_path}")
        api.upload_file(
            path_or_fileobj=local_path,
            path_in_repo=repo_path,
            repo_id=repo_id,
            repo_type="model",
        )

    print(f"\nDone! View at: https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_id", default="syauqie/castle-gec",
                        help="HuggingFace repo ID, e.g. username/model-name")
    parser.add_argument("--checkpoint", default="checkpoints/castle9/checkpoint_best.pt",
                        help="Path to checkpoint_best.pt on this machine")
    parser.add_argument("--private", action="store_true",
                        help="Create as private repo (default: public)")
    args = parser.parse_args()
    upload(args.repo_id, args.checkpoint, args.private)
