"""Generate tiny synthetic colour/stripe images to verify the code without a download.

These are NOT hand signs or a valid course-project dataset.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output_dir', default='data/toy')
    parser.add_argument('--seed', type=int, default=4110)
    args = parser.parse_args()
    root = Path(args.output_dir)
    if root.exists() and any(root.iterdir()):
        raise FileExistsError('Use a new toy-data directory.')
    rng = np.random.default_rng(args.seed)
    records = {}
    for split, per_class in [('train', 8), ('val', 3), ('test', 3)]:
        folder = root/f'{split}_signs'
        folder.mkdir(parents=True, exist_ok=True)
        records[split] = []
        for label in range(6):
            for index in range(per_class):
                pixels = rng.integers(0, 25, (64,64,3), dtype=np.uint8)
                pixels[:, :, label % 3] += 120
                if label >= 3:
                    pixels[::4, :, :] = 255
                name = f'{label}_{split}_{index:03}.png'
                Image.fromarray(pixels).save(folder/name)
                records[split].append(f'{split}_signs/{name}')
    (root/'split_manifest.json').write_text(json.dumps(
        {'source':'synthetic smoke data only','seed':args.seed,'files':records},indent=2)+'\n', encoding='utf-8')
    print(f'Toy images written to {root}; do not report these as project results.')


if __name__ == '__main__':
    main()
