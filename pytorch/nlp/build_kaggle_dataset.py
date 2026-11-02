"""Convert the original token-per-row NER CSV to aligned train/val/test text.

This preserves sentence order (70/15/15). For a real project, check duplicates
and use document/source-aware splits if document IDs are available.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path


def load_dataset(path_csv, encoding='windows-1252'):
    dataset, words, tags = [], [], []
    with open(path_csv, encoding=encoding, newline='') as handle:
        reader = csv.DictReader(handle)
        if not {'Sentence #','Word','Tag'}.issubset(set(reader.fieldnames or [])):
            raise ValueError('Expected CSV columns: Sentence #, Word, POS, Tag.')
        for row in reader:
            if row['Sentence #'].strip() and words:
                dataset.append((words,tags))
                words,tags = [],[]
            word, tag = row['Word'].strip(), row['Tag'].strip()
            if not word or not tag or any(c.isspace() for c in word):
                raise ValueError('Every row needs one whitespace-free token and one tag.')
            words.append(word)
            tags.append(tag)
    if words:  # Do not silently discard the final sentence.
        dataset.append((words,tags))
    return dataset


def save_dataset(dataset, save_dir):
    destination = Path(save_dir)
    destination.mkdir(parents=True, exist_ok=True)
    for filename, column in [('sentences.txt',0),('labels.txt',1)]:
        (destination/filename).write_text(''.join(' '.join(row[column])+'\n' for row in dataset), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--csv', default='data/kaggle/ner_dataset.csv')
    parser.add_argument('--output_dir', default='data/kaggle')
    parser.add_argument('--encoding', default='windows-1252')
    args = parser.parse_args()
    rows = load_dataset(args.csv,args.encoding)
    if len(rows) < 7:
        raise ValueError('At least seven sentences are needed for three nonempty splits.')
    root = Path(args.output_dir)
    if any((root/split).exists() for split in ['train','val','test']):
        raise FileExistsError('Split folders already exist; use a fresh output directory.')
    first, second = int(.7*len(rows)), int(.85*len(rows))
    ranges = {'train':(0,first),'val':(first,second),'test':(second,len(rows))}
    for split,(start,end) in ranges.items():
        save_dataset(rows[start:end],root/split)
    source_hash = hashlib.sha256(Path(args.csv).read_bytes()).hexdigest()
    (root/'split_manifest.json').write_text(json.dumps(
        {'source_sha256':source_hash,'strategy':'ordered sentence split',
         'sentence_index_ranges_end_exclusive':ranges},indent=2)+'\n',encoding='utf-8')
    # Fixed NER ontology, not labels discovered by inspecting evaluation data.
    schema = ['O']+[prefix+'-'+kind for kind in ['geo','gpe','org','per','tim','art','eve','nat'] for prefix in ['B','I']]
    (root/'label_schema.json').write_text(json.dumps(schema,indent=2)+'\n',encoding='utf-8')
    print({split:end-start for split,(start,end) in ranges.items()})


if __name__ == '__main__':
    main()
