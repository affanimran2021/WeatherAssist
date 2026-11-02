"""Build a training-only word vocabulary and a declared or train-only label set."""
import argparse
from collections import Counter
import json
from pathlib import Path

PAD_WORD, UNK_WORD = '<pad>', 'UNK'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data_dir', default='data/small')
    parser.add_argument('--min_count_word', type=int, default=1)
    parser.add_argument('--label_schema', default=None,
                        help='JSON list of task-defined labels, fixed independently of evaluation data')
    args = parser.parse_args()
    root = Path(args.data_dir)
    sentences = (root / 'train/sentences.txt').read_text(encoding='utf-8').splitlines()
    labels = (root / 'train/labels.txt').read_text(encoding='utf-8').splitlines()
    if not sentences or len(sentences) != len(labels):
        raise ValueError('Training files must be nonempty and aligned.')
    words, tags = Counter(), set()
    for sentence, tag_line in zip(sentences, labels):
        tokens, names = sentence.split(), tag_line.split()
        if not tokens or len(tokens) != len(names):
            raise ValueError('Every training token needs a label.')
        words.update(tokens)
        tags.update(names)
    vocabulary = [PAD_WORD, UNK_WORD] + sorted(w for w, count in words.items()
                      if count >= args.min_count_word and w not in {PAD_WORD, UNK_WORD})
    schema_path = Path(args.label_schema) if args.label_schema else root/'label_schema.json'
    if schema_path.is_file():
        tag_list = json.loads(schema_path.read_text(encoding='utf-8'))
        if not isinstance(tag_list, list) or not tag_list or len(set(tag_list)) != len(tag_list):
            raise ValueError('Label schema must be a nonempty list of unique labels.')
        if not tags.issubset(set(tag_list)):
            raise ValueError('Training labels are missing from the declared schema.')
        label_source = 'declared task schema'
    elif args.label_schema:
        raise FileNotFoundError(schema_path)
    else:
        tag_list = sorted(tags)
        label_source = 'train only'
    (root / 'words.txt').write_text('\n'.join(vocabulary) + '\n', encoding='utf-8')
    (root / 'tags.txt').write_text('\n'.join(tag_list) + '\n', encoding='utf-8')
    metadata = dict(vocab_size=len(vocabulary), number_of_tags=len(tag_list),
                    pad_word=PAD_WORD, unk_word=UNK_WORD,
                    vocabulary_source='train only', label_source=label_source)
    (root / 'dataset_params.json').write_text(json.dumps(metadata, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(metadata, indent=2))


if __name__ == '__main__':
    main()
