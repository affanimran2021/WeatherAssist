"""Controlled learning-rate sweep. Each configuration gets its own run folder."""
import argparse
import json
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent_dir', default='experiments/learning_rate')
    parser.add_argument('--data_dir', default='data/small')
    parser.add_argument('--learning_rates', nargs='+', type=float, default=[1e-4, 1e-3, 1e-2])
    parser.add_argument('--device', default=None)
    args = parser.parse_args()
    parent = Path(args.parent_dir)
    reference = json.loads((parent / 'params.json').read_text(encoding='utf-8'))
    for rate in args.learning_rates:
        run = parent / f'learning_rate_{rate:g}'
        if (run / 'params.json').exists():
            raise FileExistsError(f'Experiment already configured: {run}. Use a fresh parent folder.')
        run.mkdir(parents=True, exist_ok=True)
        config = dict(reference, learning_rate=rate)
        (run / 'params.json').write_text(json.dumps(config, indent=2)+'\n', encoding='utf-8')
        command = [sys.executable, str(Path(__file__).with_name('train.py')),
                   '--data_dir', args.data_dir, '--model_dir', str(run)]
        if args.device:
            command.extend(['--device', args.device])
        subprocess.run(command, check=True)


if __name__ == '__main__':
    main()
