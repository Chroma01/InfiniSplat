"""Download only the two DA3 inference bundle files from a model Hub repository."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo-id', required=True, help='Published v2 model repository; v1 checkpoints are incompatible')
    parser.add_argument('--revision', default='main', help='Prefer a commit ID for reproducible downloads')
    parser.add_argument('--output', type=Path, default=Path('weights/da3'))
    args = parser.parse_args()
    if args.output.exists():
        parser.error('--output must be a new directory')
    from huggingface_hub import hf_hub_download
    # Hub caches stay outside the exported model directory.
    config_path = Path(hf_hub_download(args.repo_id, 'config.json', revision=args.revision))
    config = json.loads(config_path.read_text())
    if config.get('backend') != 'da3' or config.get('format_version') not in (1, 2):
        parser.error('Expected an InfiniSplat v2 DA3 inference bundle')
    weights_path = Path(hf_hub_download(args.repo_id, 'model.safetensors', revision=args.revision))
    digest = hashlib.sha256()
    with weights_path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    if digest.hexdigest() != config.get('weights_sha256'):
        parser.error('Weight integrity check failed')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.mkdir(exist_ok=False)
    try:
        shutil.copyfile(config_path, args.output / 'config.json')
        shutil.copyfile(weights_path, args.output / 'model.safetensors')
    except BaseException:
        shutil.rmtree(args.output)
        raise
    print(args.output)


if __name__ == '__main__':
    main()
