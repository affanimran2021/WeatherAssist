"""Read aligned tokens/tags and include every sentence in a padded batch."""
import json
from pathlib import Path
import random
import torch


class DataLoader:
    def __init__(self, data_dir, params):
        root = Path(data_dir)
        metadata = json.loads((root / 'dataset_params.json').read_text(encoding='utf-8'))
        self.vocab = {w: i for i, w in enumerate((root / 'words.txt').read_text(encoding='utf-8').splitlines())}
        self.tag_map = {w: i for i, w in enumerate((root / 'tags.txt').read_text(encoding='utf-8').splitlines())}
        params.__dict__.update(metadata)
        self.unk_ind = self.vocab[metadata['unk_word']]
        self.pad_ind = self.vocab[metadata['pad_word']]
        params.pad_index = self.pad_ind

    def load_data(self, types, data_dir):
        result = {}
        for split in types:
            root = Path(data_dir) / split
            sentences = (root / 'sentences.txt').read_text(encoding='utf-8').splitlines()
            tags = (root / 'labels.txt').read_text(encoding='utf-8').splitlines()
            if not sentences or len(sentences) != len(tags):
                raise ValueError(f'{split}: expected nonempty, aligned sentence/label files')
            inputs, labels = [], []
            for index, (sentence, tag_line) in enumerate(zip(sentences, tags), 1):
                words, names = sentence.split(), tag_line.split()
                if not words or len(words) != len(names):
                    raise ValueError(f'{split} line {index}: tokens and labels do not align')
                unknown = set(names) - self.tag_map.keys()
                if unknown:
                    raise ValueError(f'{split} has labels absent from training: {sorted(unknown)}. '
                                     'Define a task label schema explicitly; do not learn it from test data.')
                inputs.append([self.vocab.get(w, self.unk_ind) for w in words])
                labels.append([self.tag_map[t] for t in names])
            result[split] = {'data': inputs, 'labels': labels, 'size': len(inputs)}
        return result

    def data_iterator(self, data, params, shuffle=False):
        order = list(range(data['size']))
        if shuffle:
            random.shuffle(order)  # Seed once at the start of training, not once per epoch.
        for start in range(0, data['size'], params.batch_size):
            indices = order[start:start + params.batch_size]
            length = max(len(data['data'][i]) for i in indices)
            tokens = torch.full((len(indices), length), self.pad_ind, dtype=torch.long)
            tags = torch.full((len(indices), length), -1, dtype=torch.long)
            for row, index in enumerate(indices):
                size = len(data['data'][index])
                tokens[row, :size] = torch.tensor(data['data'][index], dtype=torch.long)
                tags[row, :size] = torch.tensor(data['labels'][index], dtype=torch.long)
            yield tokens, tags
