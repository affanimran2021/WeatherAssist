"""Resize SIGNS images; split only the supplied TRAINING partition into train/val.

Keep the supplied test partition untouched. Reruns require a fresh output directory.
"""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import random
from PIL import Image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data_dir', default='data/SIGNS')
    parser.add_argument('--output_dir', default='data/64x64_SIGNS')
    parser.add_argument('--seed', type=int, default=4110)
    parser.add_argument('--image_size', type=int, default=64)
    parser.add_argument('--val_fraction', type=float, default=0.2)
    args = parser.parse_args()
    if not 0 < args.val_fraction < 1 or args.image_size < 16:
        raise ValueError('Choose 0 < val_fraction < 1 and image_size >= 16.')
    source, destination = Path(args.data_dir), Path(args.output_dir)
    if destination.exists() and any(p.name != '.gitkeep' for p in destination.iterdir()):
        raise FileExistsError('Use a fresh output directory to prevent stale split membership.')
    by_class = defaultdict(list)
    for path in sorted((source/'train_signs').iterdir()):
        if path.suffix.lower() in {'.jpg', '.jpeg', '.png'}:
            by_class[int(path.name.split('_', 1)[0])].append(path)
    test = sorted(p for p in (source/'test_signs').iterdir()
                  if p.suffix.lower() in {'.jpg', '.jpeg', '.png'})
    if not by_class or not test:
        raise ValueError('Both train_signs and test_signs must contain images.')
    rng = random.Random(args.seed)
    splits = {'train': [], 'val': [], 'test': test}
    for label, paths in sorted(by_class.items()):
        if len(paths) < 2:
            raise ValueError(f'Class {label} needs at least two training images.')
        rng.shuffle(paths)
        size = max(1, min(len(paths)-1, round(len(paths)*args.val_fraction)))
        splits['val'].extend(paths[:size])
        splits['train'].extend(paths[size:])
    records = {}
    for split, paths in splits.items():
        output = destination/f'{split}_signs'
        output.mkdir(parents=True, exist_ok=True)
        records[split] = []
        for path in paths:
            with Image.open(path) as image:
                image.convert('RGB').resize((args.image_size, args.image_size),
                    Image.Resampling.BILINEAR).save(output/path.name)
            records[split].append(str(path.relative_to(source)))
    payload = json.dumps(records, sort_keys=True).encode()
    manifest = dict(seed=args.seed, image_size=args.image_size,
                    val_fraction=args.val_fraction, files=records,
                    split_sha256=hashlib.sha256(payload).hexdigest())
    (destination/'split_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n', encoding='utf-8')
    print({split: len(paths) for split, paths in splits.items()})


if __name__ == '__main__':
    main()
