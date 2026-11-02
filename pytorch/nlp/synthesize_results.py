"""Collect validation results only. Never select hyperparameters using test metrics."""
import argparse
import csv
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent_dir', default='experiments')
    args = parser.parse_args()
    parent = Path(args.parent_dir)
    rows = []
    for path in sorted(parent.rglob('metrics_val_best_weights.json')):
        metrics = json.loads(path.read_text(encoding='utf-8'))
        rows.append({'run': str(path.parent.relative_to(parent)), **metrics})
    if not rows:
        print('No validation results yet. Train at least one experiment first.')
        return
    fields = ['run'] + sorted(set().union(*(row.keys() for row in rows)) - {'run'})
    with (parent / 'results.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    lines = ['| ' + ' | '.join(fields) + ' |', '| ' + ' | '.join(['---']*len(fields)) + ' |']
    lines += ['| ' + ' | '.join(str(row.get(key, '')) for key in fields) + ' |' for row in rows]
    text = '\n'.join(lines) + '\n'
    (parent / 'results.md').write_text(text, encoding='utf-8')
    print(text)


if __name__ == '__main__':
    main()
