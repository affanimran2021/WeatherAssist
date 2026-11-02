"""Tag one space-tokenized sentence using the vocabulary saved with the run."""
import argparse
import json
from pathlib import Path
import torch
import utils
from model.net import Net


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--text', required=True)
    parser.add_argument('--model_dir', default='experiments/base_model')
    parser.add_argument('--device', default='auto')
    args = parser.parse_args()
    root = Path(args.model_dir)
    device = utils.get_device(args.device)
    state = torch.load(root/'best.pth.tar', map_location='cpu', weights_only=True)
    params = utils.Params(root/'params.json')
    params.__dict__.update(state['params'])
    vocab = {w: i for i, w in enumerate((root/'words.txt').read_text(encoding='utf-8').splitlines())}
    tags = (root/'tags.txt').read_text(encoding='utf-8').splitlines()
    tokens = args.text.split()
    if not tokens:
        raise ValueError('Provide a nonempty sentence.')
    ids = torch.tensor([[vocab.get(w, vocab[params.unk_word]) for w in tokens]], device=device)
    model = Net(params).to(device)
    model.load_state_dict(state['state_dict'])
    model.eval()
    with torch.no_grad():
        predictions = model(ids).argmax(dim=-1).cpu().tolist()
    print(json.dumps([{'token': word, 'tag': tags[i]} for word, i in zip(tokens, predictions)]))


if __name__ == '__main__':
    main()
