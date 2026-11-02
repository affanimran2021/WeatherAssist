"""Evaluate a selected checkpoint. Use validation for selection; reserve test for final evaluation."""
import argparse
from pathlib import Path
import torch
import utils
from model.net import Net, loss_fn
from model.data_loader import fetch_dataloader


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data_dir', default='data/64x64_SIGNS')
    parser.add_argument('--model_dir', default='experiments/base_model')
    parser.add_argument('--restore_file', default='best', choices=['best', 'last'])
    parser.add_argument('--split', default='test', choices=['val', 'test'])
    parser.add_argument('--device', default='auto')
    args = parser.parse_args()
    checkpoint = Path(args.model_dir) / (args.restore_file + '.pth.tar')
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    params = utils.Params(Path(args.model_dir) / 'params.json')
    params.__dict__.update(state['params'])  # Architecture must match the trained checkpoint.
    params.device = str(utils.get_device(args.device))
    utils.seed_everything(params.seed)
    loaders = fetch_dataloader([args.split], args.data_dir, params)
    batches = loaders[args.split]
    model = Net(params).to(params.device)
    model.load_state_dict(state['state_dict'])
    metrics = utils.run_epoch(model, loss_fn, batches, params.device)
    metrics.update(epoch=state['epoch'], split=args.split)
    destination = Path(args.model_dir) / f'metrics_{args.split}_{args.restore_file}.json'
    utils.save_dict_to_json(metrics, destination)
    print(metrics)


if __name__ == '__main__':
    main()
